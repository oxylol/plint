//! The plugin's page over HTTP, for a web browser: the editor on Linux (`browser.rs`) and
//! `examples/ui_harness.rs`. The page talks to the plugin as it does in a webview, through
//! `window.ipc.postMessage`, which a script put in the page turns into a POST to `/ipc`.

use tiny_http::{Header, Request, Response};

use crate::bridge::Bridge;

/// Makes the page think it's in the plugin: `window.ipc` is what wry defines in a webview. It sends the
/// token the page was opened with (`?k=`), as `src/ui/pluginState.ts` does for its own requests.
pub const IPC_SHIM: &str = "<script>window.ipc={postMessage:(m)=>fetch('/ipc',{method:'POST',body:m,keepalive:true,headers:{'X-Chimera-Token':new URLSearchParams(location.search).get('k')||''}})}</script>";

pub fn header(name: &str, value: &str) -> Header {
    Header::from_bytes(name.as_bytes(), value.as_bytes()).expect("an ASCII header")
}

/// Answers one request: `/ipc` takes a command, and anything else is what the bridge serves, with
/// `shim` put at the top of the page.
pub fn answer(bridge: &Bridge, mut req: Request, shim: &str, extra: Option<Header>) {
    let path = req.url().split('?').next().unwrap_or("/").to_string();
    if path == "/ipc" {
        let mut body = String::new();
        let _ = req.as_reader().read_to_string(&mut body);
        bridge.command(&body);
        let _ = req.respond(Response::empty(204));
        return;
    }
    let reply = bridge.request(&path);
    let mut body = reply.body.into_owned();
    if path == "/" {
        body = String::from_utf8_lossy(&body).replacen("<head>", &format!("<head>{shim}"), 1).into_bytes();
    }
    let mut response = Response::from_data(body).with_status_code(reply.status).with_header(header("Content-Type", reply.mime)).with_header(header("Cache-Control", "no-store"));
    if let Some(extra) = extra {
        response = response.with_header(extra);
    }
    let _ = req.respond(response);
}
