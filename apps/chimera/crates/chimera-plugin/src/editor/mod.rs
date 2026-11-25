//! The editor: the system's webview in the host's window, showing the React UI embedded in the
//! plugin. WKWebView on macOS, WebView2 on Windows. [`crate::bridge`] does all the talking; this only
//! connects it, and `macos.rs` and `windows.rs` hold what differs between them.
//!
//! - The page and the state it polls come from a custom protocol (`chimera://localhost/…`, which wry
//!   serves as `http://chimera.localhost/…` on Windows), so no files or ports are involved and the UI
//!   pulls state at its own frame rate.
//! - Commands go the other way through `window.ipc.postMessage` (JSON).
//! - Both handlers run on the host's GUI thread.

use std::cell::RefCell;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::{Rc, Weak};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use nih_plug::prelude::*;
use wry::dpi::{LogicalPosition, LogicalSize};
use wry::http::{Request, Response};
use wry::{Rect, WebContext, WebView, WebViewBuilder};

use crate::bridge::{Bridge, EditAction, GripAction, GuiHost, ParamHost, WINDOW_DEFAULT, WINDOW_MAX, WINDOW_MIN};
use crate::core::Shared;
use crate::ChimeraParams;

#[cfg_attr(target_os = "macos", path = "macos.rs")]
#[cfg_attr(target_os = "windows", path = "windows.rs")]
mod platform;

/// Polls after the grip lets go before the host's view is taken as the answer (about half a second:
/// a CLAP host may resize a little after it's asked).
const SETTLE_POLLS: u32 = 30;
/// Room left on the screen for the host's title bar and toolbar above the editor (logical pixels).
const HOST_CHROME: f64 = 60.0;

pub struct WebEditor {
    params: Arc<ChimeraParams>,
    shared: Arc<Shared>,
    /// the host's scale for the editor, as f32 bits: Windows only, where sizes to the host are physical
    scale: Arc<AtomicU32>,
}

impl WebEditor {
    pub fn new(params: Arc<ChimeraParams>, shared: Arc<Shared>) -> Self {
        Self { params, shared, scale: Arc::new(AtomicU32::new(1f32.to_bits())) }
    }
}

fn scale_of(scale: &AtomicU32) -> f64 {
    f32::from_bits(scale.load(Ordering::Relaxed)) as f64
}

/// The host's connection, with the host's view for resizing.
struct WindowHost {
    host: GuiHost,
    params: Arc<ChimeraParams>,
    view: platform::View,
    scale: Arc<AtomicU32>,
    /// the webview, once built: weak, because the webview owns this struct
    webview: Weak<RefCell<Option<WebView>>>,
    /// the window's size and the pointer's place when the drag began (logical, y downwards)
    drag: Mutex<Option<((u32, u32), (f64, f64))>>,
    /// the view size and the monitor's scale the page's zoom was set for
    zoomed_for: Mutex<((u32, u32), u32)>,
    /// polls until the view's size is taken as the window's
    settling: Mutex<u32>,
}

// SAFETY: the bridge calls this from the host's GUI thread, while the editor is open.
unsafe impl Send for WindowHost {}
unsafe impl Sync for WindowHost {}

impl WindowHost {
    fn store(&self, size: (u32, u32)) {
        *self.params.window.write().unwrap_or_else(|e| e.into_inner()) = size;
    }

    fn size(&self) -> (u32, u32) {
        *self.params.window.read().unwrap_or_else(|e| e.into_inner())
    }

    fn view_size(&self) -> (u32, u32) {
        platform::view_size(self.view, scale_of(&self.scale))
    }

    fn resize(&self, asked: (u32, u32)) {
        if asked == self.size() {
            return;
        }
        // `request_resize` reports the editor's `size()`, so that changes first. Its answer means
        // nothing here: clap-wrapper's VST3 layer returns false on success and true on failure, and
        // its AU layer always returns false. `tick` looks at what really happened once the grip is let go.
        self.store(asked);
        let _ = self.host.context.request_resize();
    }

    fn settle(&self) {
        *self.settling.lock().unwrap_or_else(|e| e.into_inner()) = SETTLE_POLLS;
    }
}

impl ParamHost for WindowHost {
    fn begin(&self, id: &str) {
        self.host.begin(id);
    }
    fn set(&self, id: &str, plain: f32) {
        self.host.set(id, plain);
    }
    fn end(&self, id: &str) {
        self.host.end(id);
    }
    fn values(&self) -> Vec<(String, f32)> {
        self.host.values()
    }
    fn grip(&self, action: GripAction) {
        let mut drag = self.drag.lock().unwrap_or_else(|e| e.into_inner());
        match action {
            GripAction::Begin => {
                // the view's size, not the saved one: the window may be smaller to fit the screen.
                // The pointer is read here, not taken from the page, whose coordinates change with its zoom.
                *drag = Some((self.view_size(), platform::pointer(scale_of(&self.scale))));
                *self.settling.lock().unwrap_or_else(|e| e.into_inner()) = 0;
            }
            GripAction::Move => {
                let Some(((w, h), from)) = *drag else { return };
                let to = platform::pointer(scale_of(&self.scale));
                let width = w as f64 + to.0 - from.0;
                let height = h as f64 + to.1 - from.1;
                self.resize(((width.round().max(0.0) as u32).clamp(WINDOW_MIN.0, WINDOW_MAX.0), (height.round().max(0.0) as u32).clamp(WINDOW_MIN.1, WINDOW_MAX.1)));
            }
            GripAction::End => {
                *drag = None;
                self.settle();
            }
            GripAction::Reset => {
                *drag = None;
                self.resize(WINDOW_DEFAULT);
                self.settle();
            }
        }
    }
    fn edit(&self, action: EditAction) {
        let Some(cell) = self.webview.upgrade() else { return };
        // this runs inside the webview's own handlers, never while it's being built
        let Ok(webview) = cell.try_borrow() else { return };
        if let Some(webview) = webview.as_ref() {
            platform::edit(webview, action);
        }
    }
    /// The page's zoom follows the host's view, whichever way that came to change, and once the
    /// grip has settled the view's size is the window's: a host may have refused or clamped it.
    fn tick(&self) {
        let now = self.view_size();
        let mut settling = self.settling.lock().unwrap_or_else(|e| e.into_inner());
        if *settling > 0 {
            *settling -= 1;
            if *settling == 0 {
                self.store(now);
            }
        }
        drop(settling);
        let key = (now, platform::monitor_dpi(self.view));
        let mut zoomed_for = self.zoomed_for.lock().unwrap_or_else(|e| e.into_inner());
        if *zoomed_for != key {
            let Some(cell) = self.webview.upgrade() else { return };
            // this runs inside the webview's own handlers, never while it's being built
            let Ok(webview) = cell.try_borrow() else { return };
            if let Some(webview) = webview.as_ref() {
                *zoomed_for = key;
                platform::follow(webview, self.view, now, scale_of(&self.scale));
            }
        }
    }
}

/// The webview, in the host's view. None if it couldn't be made: the host gets an empty window rather
/// than going down with a panic.
fn build(view: platform::View, size: (u32, u32), bridge: &Arc<Bridge>, context: Option<&mut WebContext>) -> Option<WebView> {
    let (width, height) = size;
    let serving = bridge.clone();
    let protocol = move |_id: wry::WebViewId, req: Request<Vec<u8>>| {
        let reply = serving.request(req.uri().path());
        Response::builder().status(reply.status).header("Content-Type", reply.mime).header("Cache-Control", "no-store").body(reply.body).unwrap()
    };
    let commands = bridge.clone();
    let ipc = move |req: Request<String>| commands.command(req.body());

    let built = catch_unwind(AssertUnwindSafe(|| {
        let builder = match context {
            Some(context) => WebViewBuilder::new_with_web_context(context),
            None => WebViewBuilder::new(),
        };
        builder
            .with_bounds(Rect { position: LogicalPosition::new(0, 0).into(), size: LogicalSize::new(width, height).into() })
            .with_custom_protocol("chimera".into(), protocol)
            .with_ipc_handler(ipc)
            // a link must not replace the plugin's own page
            .with_navigation_handler(|url| url.starts_with("chimera://") || url.starts_with("http://chimera.localhost/"))
            // macOS: the first click on a window the host hasn't focused still works the control under it
            .with_accept_first_mouse(true)
            .with_devtools(true)
            .with_url("chimera://localhost/")
            .build_as_child(&platform::HostView(view))
    }));
    match built {
        Ok(Ok(webview)) => {
            platform::after_build(&webview, size);
            Some(webview)
        }
        Ok(Err(err)) => {
            nih_error!("CHIMERA: could not create the webview: {err}");
            None
        }
        Err(_) => {
            nih_error!("CHIMERA: creating the webview panicked");
            None
        }
    }
}

/// The open editor. Dropped (on the GUI thread) when the host closes the window; wry then removes
/// the webview from the host's view.
struct Open {
    bridge: Arc<Bridge>,
    /// before `_keep`, which may hold what the webview was built with
    _webview: Rc<RefCell<Option<WebView>>>,
    _keep: platform::Keep,
}

// SAFETY: nih-plug creates and drops the editor handle on the GUI thread only.
unsafe impl Send for Open {}

impl Drop for Open {
    fn drop(&mut self) {
        self.bridge.close();
    }
}

impl Editor for WebEditor {
    fn spawn(&self, parent: ParentWindowHandle, context: Arc<dyn GuiContext>) -> Box<dyn std::any::Any + Send> {
        let Some(view) = platform::view(parent) else { return Box::new(()) };
        let size = self.size();
        let webview = Rc::new(RefCell::new(None));
        let host = WindowHost { host: GuiHost::new(self.params.clone(), context), params: self.params.clone(), view, scale: self.scale.clone(), webview: Rc::downgrade(&webview), drag: Mutex::new(None), zoomed_for: Mutex::new((size, platform::monitor_dpi(view))), settling: Mutex::new(0) };
        let bridge = Arc::new(Bridge::new(self.params.clone(), self.shared.clone(), Box::new(host)));
        let keep = platform::open(view, size, scale_of(&self.scale), bridge.clone(), webview.clone());
        Box::new(Open { bridge, _webview: webview, _keep: keep })
    }

    fn size(&self) -> (u32, u32) {
        platform::fit_screen(*self.params.window.read().unwrap_or_else(|e| e.into_inner()), scale_of(&self.scale))
    }

    fn set_scale_factor(&self, factor: f32) -> bool {
        if platform::HOST_SCALES {
            self.scale.store(factor.to_bits(), Ordering::Relaxed);
        }
        platform::HOST_SCALES
    }

    // the page polls the parameters, so host-side changes need no push
    fn param_value_changed(&self, _id: &str, _normalized_value: f32) {}
    fn param_modulation_changed(&self, _id: &str, _modulation_offset: f32) {}
    fn param_values_changed(&self) {}
}
