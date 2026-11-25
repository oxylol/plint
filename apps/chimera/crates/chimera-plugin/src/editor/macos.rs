//! The editor on macOS: a WKWebView in the host's NSView. Sizes are in points, and macOS does the
//! scaling, so the host's scale factor is declined.

use std::cell::{Cell, RefCell};
use std::ffi::c_void;
use std::ptr::{NonNull, null, null_mut};
use std::rc::Rc;
use std::sync::Arc;

use nih_plug::prelude::ParentWindowHandle;
use objc2::runtime::AnyObject;
use objc2::{class, msg_send, sel};
use objc2_core_foundation::{CGPoint, CGRect};
use raw_window_handle::{AppKitWindowHandle, HandleError, HasWindowHandle, RawWindowHandle, WindowHandle};
use wry::{WebView, WebViewExtMacOS};

use super::HOST_CHROME;
use crate::bridge::{Bridge, EditAction, page_zoom};

/// macOS scales for us: sizes are in points.
pub const HOST_SCALES: bool = false;

/// The host's NSView.
pub type View = NonNull<c_void>;

/// NSViewWidthSizable | NSViewHeightSizable
const FOLLOW_SUPERVIEW: usize = 2 | 16;

pub fn view(parent: ParentWindowHandle) -> Option<View> {
    let ParentWindowHandle::AppKitNsView(view) = parent else { return None };
    NonNull::new(view)
}

/// The host's NSView, as a window handle wry can attach to.
pub struct HostView(pub View);

impl HasWindowHandle for HostView {
    fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
        let raw = RawWindowHandle::AppKit(AppKitWindowHandle::new(self.0));
        // SAFETY: the host keeps its view alive while the editor is open (we drop the webview first)
        Ok(unsafe { WindowHandle::borrow_raw(raw) })
    }
}

/// The size of the host's view (points).
pub fn view_size(view: View, _scale: f64) -> (u32, u32) {
    // SAFETY: an NSView, alive while the editor is open, and `frame` returns a CGRect
    let frame: CGRect = unsafe { msg_send![view.as_ptr() as *const AnyObject, frame] };
    (frame.size.width.round() as u32, frame.size.height.round() as u32)
}

/// The pointer on screen (points, y downwards like the other platforms: AppKit's goes up).
pub fn pointer(_scale: f64) -> (f64, f64) {
    // SAFETY: a class method returning an NSPoint
    let at: CGPoint = unsafe { msg_send![class!(NSEvent), mouseLocation] };
    (at.x, -at.y)
}

/// A size that fits on the screen in use, host's window included, even below the usual minimum: a
/// window hanging off the screen would have its grip there too, with no way to make it smaller. The
/// page zooms out to suit.
pub fn fit_screen((width, height): (u32, u32), _scale: f64) -> (u32, u32) {
    // SAFETY: class methods on the main thread, where hosts ask for the editor's size
    let screen: *const AnyObject = unsafe { msg_send![class!(NSScreen), mainScreen] };
    if screen.is_null() {
        return (width, height);
    }
    // SAFETY: an NSScreen, and `visibleFrame` returns a CGRect
    let room: CGRect = unsafe { msg_send![screen, visibleFrame] };
    (width.min(room.size.width as u32).max(1), height.min((room.size.height - HOST_CHROME).max(0.0) as u32).max(1))
}

/// `_WKLayoutModeDynamicSizeComputedFromViewScale`: the page is laid out at the view's size divided by
/// the view scale, and drawn scaled.
const LAYOUT_FROM_VIEW_SCALE: usize = 2;

/// Whether this WKWebView has the view scale (WebKit's, since macOS 10.11, but not public).
fn has_view_scale(wk: &AnyObject) -> bool {
    // SAFETY: `respondsToSelector:` on an NSObject
    unsafe { msg_send![wk, respondsToSelector: sel!(_setViewScale:)] && msg_send![wk, respondsToSelector: sel!(_setLayoutMode:)] }
}

/// wry fixes a child's size; this one follows the host's view whenever that changes. The page fits a
/// smaller window by the view scale, not `pageZoom`: under a page zoom, WebKit keeps text of 9 px and
/// more at 9 px or larger (its "smart minimum" font size), so at the smallest windows the labels outgrew
/// their buttons and the page outgrew the window. A view scale scales the drawing, so text shrinks with
/// everything else.
pub fn after_build(webview: &WebView, size: (u32, u32)) {
    let wk = webview.webview();
    // SAFETY: `setAutoresizingMask:` takes an NSUInteger, on the main thread we're on
    let _: () = unsafe { msg_send![&*wk, setAutoresizingMask: FOLLOW_SUPERVIEW] };
    if has_view_scale(&wk) {
        // SAFETY: an `_WKLayoutMode` (NSUInteger), on the main thread
        let _: () = unsafe { msg_send![&*wk, _setLayoutMode: LAYOUT_FROM_VIEW_SCALE] };
    }
    fit(webview, size);
}

/// The page's scale for a view of this size.
fn fit(webview: &WebView, size: (u32, u32)) {
    let wk = webview.webview();
    if has_view_scale(&wk) {
        // WebKit raises an exception for a scale that isn't positive, which would take the host down
        let scale: f64 = page_zoom(size).clamp(0.05, 1.0);
        // SAFETY: a CGFloat in range, on the main thread
        let _: () = unsafe { msg_send![&*wk, _setViewScale: scale] };
    } else {
        let _ = webview.zoom(page_zoom(size));
    }
}

/// A clipboard shortcut, as the Edit menu would send it: to the webview, which acts on the field with
/// the focus. Pasting this way isn't the page reading the clipboard, so WebKit doesn't ask first.
pub fn edit(webview: &WebView, action: EditAction) {
    let wk = webview.webview();
    let none: *const AnyObject = null();
    // SAFETY: NSResponder actions WKWebView implements, taking a sender, on the main thread we're on
    unsafe {
        let _: () = match action {
            EditAction::Cut => msg_send![&*wk, cut: none],
            EditAction::Copy => msg_send![&*wk, copy: none],
            EditAction::Paste => msg_send![&*wk, paste: none],
            EditAction::SelectAll => msg_send![&*wk, selectAll: none],
        };
    }
}

/// The view changed size: the webview follows it by itself, and the page's scale follows here.
pub fn follow(webview: &WebView, _view: View, size: (u32, u32), _scale: f64) {
    fit(webview, size);
}

/// macOS keeps the page's scale itself.
pub fn monitor_dpi(_view: View) -> u32 {
    0
}

// CoreFoundation's run loop timer. A view of ours watching for its window would need a class of ours,
// and bundles of one binary in one process can't all register the same class name.
#[repr(C)]
struct CFRunLoopTimerContext {
    version: isize,
    info: *mut c_void,
    retain: *const c_void,
    release: *const c_void,
    copy_description: *const c_void,
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    static kCFRunLoopCommonModes: *const c_void;
    fn CFAbsoluteTimeGetCurrent() -> f64;
    fn CFRunLoopGetMain() -> *mut c_void;
    fn CFRunLoopTimerCreate(allocator: *const c_void, fire_date: f64, interval: f64, flags: usize, order: isize, callout: extern "C" fn(*mut c_void, *mut c_void), context: *mut CFRunLoopTimerContext) -> *mut c_void;
    fn CFRunLoopAddTimer(run_loop: *mut c_void, timer: *mut c_void, mode: *const c_void);
    fn CFRunLoopTimerInvalidate(timer: *mut c_void);
    fn CFRelease(object: *const c_void);
}

/// How often to look for the host's window (seconds).
const WINDOW_POLL: f64 = 0.05;

fn in_window(view: View) -> bool {
    // SAFETY: an NSView, alive while the editor is open, and `window` returns an NSWindow or nil
    let window: *const AnyObject = unsafe { msg_send![view.as_ptr() as *const AnyObject, window] };
    !window.is_null()
}

/// Waiting for the host to put its view in a window. wry panics without one
/// (`ns_view.window().unwrap()`), and clap-wrapper's AU hands over its view before the host has done
/// that, in `uiViewForAudioUnit`. Building in a temporary window and moving across doesn't work: the
/// webview never draws in the window it's moved to.
struct Waiting {
    view: View,
    size: (u32, u32),
    bridge: Arc<Bridge>,
    webview: Rc<RefCell<Option<WebView>>>,
    timer: Cell<*mut c_void>,
}

extern "C" fn look_for_window(_timer: *mut c_void, info: *mut c_void) {
    // SAFETY: `info` is the `Waiting` the `Keep` owns, which invalidates this timer before dropping it
    let waiting = unsafe { &*(info as *const Waiting) };
    if in_window(waiting.view) {
        // SAFETY: a timer of ours, on the run loop it's on
        unsafe { CFRunLoopTimerInvalidate(waiting.timer.get()) };
        *waiting.webview.borrow_mut() = super::build(waiting.view, waiting.size, &waiting.bridge, None);
    }
}

/// What the open editor holds on to here: the timer, if the webview is still waiting for a window.
pub struct Keep(Option<Box<Waiting>>);

impl Drop for Keep {
    fn drop(&mut self) {
        if let Some(waiting) = &self.0 {
            // SAFETY: a timer of ours, on the main thread, and no use of it after this
            unsafe {
                CFRunLoopTimerInvalidate(waiting.timer.get());
                CFRelease(waiting.timer.get());
            }
        }
    }
}

/// Builds the webview into `webview`, now or once the host's view is in a window.
pub fn open(view: View, size: (u32, u32), _scale: f64, bridge: Arc<Bridge>, webview: Rc<RefCell<Option<WebView>>>) -> Keep {
    if in_window(view) {
        *webview.borrow_mut() = super::build(view, size, &bridge, None);
        return Keep(None);
    }
    let waiting = Box::new(Waiting { view, size, bridge, webview, timer: Cell::new(null_mut()) });
    let mut timer_context = CFRunLoopTimerContext { version: 0, info: &*waiting as *const Waiting as *mut c_void, retain: null(), release: null(), copy_description: null() };
    // SAFETY: the context is copied, and its `info` lives in the box `Keep` owns, whose drop
    // invalidates the timer
    unsafe {
        let timer = CFRunLoopTimerCreate(null(), CFAbsoluteTimeGetCurrent() + WINDOW_POLL, WINDOW_POLL, 0, 0, look_for_window, &mut timer_context);
        waiting.timer.set(timer);
        CFRunLoopAddTimer(CFRunLoopGetMain(), timer, kCFRunLoopCommonModes);
    }
    Keep(Some(waiting))
}
