//! The editor on Windows: WebView2 in the host's window. The host sizes windows in physical pixels
//! and tells the editor its scale, so sizes here are logical and multiplied by it at the edges.

use std::cell::RefCell;
use std::mem::{size_of, zeroed};
use std::num::NonZeroIsize;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

use nih_plug::prelude::ParentWindowHandle;
use raw_window_handle::{HandleError, HasWindowHandle, RawWindowHandle, Win32WindowHandle, WindowHandle};
use windows_sys::Win32::Foundation::{HWND, POINT, RECT};
use windows_sys::Win32::Graphics::Gdi::{GetMonitorInfoW, MONITOR_DEFAULTTONEAREST, MONITORINFO, MonitorFromPoint};
use windows_sys::Win32::UI::HiDpi::GetDpiForWindow;
use windows_sys::Win32::UI::WindowsAndMessaging::{GetClientRect, GetCursorPos};
use wry::dpi::{PhysicalPosition, PhysicalSize};
use wry::{Rect, WebContext, WebView};

use super::HOST_CHROME;
use crate::bridge::{Bridge, EditAction, page_zoom};

/// The host passes its scale on, and sizes to it are physical.
pub const HOST_SCALES: bool = true;

/// The host's window.
pub type View = NonZeroIsize;

pub fn view(parent: ParentWindowHandle) -> Option<View> {
    let ParentWindowHandle::Win32Hwnd(hwnd) = parent else { return None };
    NonZeroIsize::new(hwnd as isize)
}

/// The host's window, as a window handle wry can attach to.
pub struct HostView(pub View);

impl HasWindowHandle for HostView {
    fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
        let raw = RawWindowHandle::Win32(Win32WindowHandle::new(self.0));
        // SAFETY: the host keeps its window alive while the editor is open (we drop the webview first)
        Ok(unsafe { WindowHandle::borrow_raw(raw) })
    }
}

fn client(view: View) -> (u32, u32) {
    // SAFETY: a window, alive while the editor is open, and a RECT to fill
    let mut rect: RECT = unsafe { zeroed() };
    unsafe { GetClientRect(view.get() as HWND, &mut rect) };
    ((rect.right - rect.left).max(0) as u32, (rect.bottom - rect.top).max(0) as u32)
}

/// The size of the host's window (logical pixels).
pub fn view_size(view: View, scale: f64) -> (u32, u32) {
    let (w, h) = client(view);
    ((w as f64 / scale).round() as u32, (h as f64 / scale).round() as u32)
}

/// The pointer on screen (logical pixels, y downwards).
pub fn pointer(scale: f64) -> (f64, f64) {
    let mut at = POINT { x: 0, y: 0 };
    // SAFETY: a POINT to fill
    unsafe { GetCursorPos(&mut at) };
    (at.x as f64 / scale, at.y as f64 / scale)
}

/// A size that fits on the screen in use, host's window included, even below the usual minimum: a
/// window hanging off the screen would have its grip there too. The screen is the pointer's: the
/// editor is asked its size before it has a window, just after it was opened from there.
pub fn fit_screen((width, height): (u32, u32), scale: f64) -> (u32, u32) {
    let mut at = POINT { x: 0, y: 0 };
    // SAFETY: plain Win32 calls with structs to fill, sized as MONITORINFO wants
    let room = unsafe {
        GetCursorPos(&mut at);
        let monitor = MonitorFromPoint(at, MONITOR_DEFAULTTONEAREST);
        let mut info: MONITORINFO = zeroed();
        info.cbSize = size_of::<MONITORINFO>() as u32;
        if GetMonitorInfoW(monitor, &mut info) == 0 {
            return (width, height);
        }
        info.rcWork
    };
    let (room_w, room_h) = ((room.right - room.left) as f64 / scale, (room.bottom - room.top) as f64 / scale);
    (width.min(room_w.max(0.0) as u32).max(1), height.min((room_h - HOST_CHROME).max(0.0) as u32).max(1))
}

pub fn after_build(webview: &WebView, size: (u32, u32)) {
    let _ = webview.zoom(page_zoom(size));
}

/// WebView2 takes Ctrl+C and the rest itself, and the page only asks on macOS.
pub fn edit(_webview: &WebView, _action: EditAction) {}

/// The monitor's DPI, which WebView2 scales the page by (96 is 100 %).
pub fn monitor_dpi(view: View) -> u32 {
    // SAFETY: a window, alive while the editor is open
    unsafe { GetDpiForWindow(view.get() as HWND) }.max(1)
}

/// The host's window changed size, or moved to a monitor with another scale. WebView2 doesn't follow
/// its parent by itself, so it fills the window here, in physical pixels. It scales the page by the
/// monitor's DPI, and the host may have chosen another scale for the editor (100 % on a 150 % monitor,
/// say), so the zoom undoes the difference as well as fitting the layout.
pub fn follow(webview: &WebView, view: View, size: (u32, u32), scale: f64) {
    let (w, h) = client(view);
    let _ = webview.set_bounds(Rect { position: PhysicalPosition::new(0, 0).into(), size: PhysicalSize::new(w, h).into() });
    let _ = webview.zoom(page_zoom(size) * scale * 96.0 / monitor_dpi(view) as f64);
}

/// What the open editor holds on to here: the webview's context, which says where WebView2 keeps its
/// data. By default that's beside the host's executable, which a host in Program Files can't write to.
pub struct Keep(#[allow(dead_code)] WebContext);

pub fn open(view: View, size: (u32, u32), scale: f64, bridge: Arc<Bridge>, webview: Rc<RefCell<Option<WebView>>>) -> Keep {
    let folder = std::env::var_os("LOCALAPPDATA").map(|d| PathBuf::from(d).join("CHIMERA").join("WebView2"));
    let mut context = WebContext::new(folder);
    let built = super::build(view, size, &bridge, Some(&mut context));
    // wry turned the logical bounds into pixels by the monitor's DPI, not the host's scale
    if let Some(built) = &built {
        follow(built, view, size, scale);
    }
    *webview.borrow_mut() = built;
    Keep(context)
}
