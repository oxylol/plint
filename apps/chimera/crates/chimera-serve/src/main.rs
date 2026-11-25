//! `chimera-serve`: CHIMERA as a text-to-speech service (Plint's addition to the fork).
//!
//!   POST /api/speak          a JSON script (see `script.rs`) → audio/wav
//!   POST /api/speak?plan=1   the same script → its phones and gestures as JSON, itself a script
//!   GET  /api/speak?text=…   plain text → audio/wav (also body=, rate=, f0=)
//!   GET  /api/health         {"ok":true,…}
//!   GET  anything else       the web app's files from CHIMERA_SITE
//!
//! Environment: PORT (5270), HOST (127.0.0.1), CHIMERA_SITE (the built page, `dist`),
//! CHIMERA_DICT (cmudict.txt.gz; `<site>/data/cmudict.txt.gz` when absent), CHIMERA_JOBS (renders
//! at once, 2).

mod script;

use chimera::phonetics::english::EnglishG2P;
use script::{Planned, Script};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

/// The largest request body (a gestural score for two minutes is well under this).
const MAX_BODY: u64 = 4 << 20;

/// At most `n` renders at once: each takes a core for about half the length of what it says.
struct Gate {
    free: Mutex<usize>,
    cv: Condvar,
}

impl Gate {
    fn run<T>(&self, f: impl FnOnce() -> T) -> T {
        let mut free = self.free.lock().unwrap();
        while *free == 0 {
            free = self.cv.wait(free).unwrap();
        }
        *free -= 1;
        drop(free);
        let out = f();
        *self.free.lock().unwrap() += 1;
        self.cv.notify_one();
        out
    }
}

struct App {
    g2p: EnglishG2P,
    /// lines of the pronouncing dictionary that were loaded (0: spelling rules only)
    words: usize,
    site: PathBuf,
    gate: Gate,
}

fn header(k: &str, v: &str) -> Header {
    Header::from_bytes(k.as_bytes(), v.as_bytes()).unwrap()
}

fn json(code: u16, body: String) -> Response<std::io::Cursor<Vec<u8>>> {
    Response::from_data(body.into_bytes())
        .with_status_code(StatusCode(code))
        .with_header(header("Content-Type", "application/json; charset=utf-8"))
        .with_header(header("Cache-Control", "no-store"))
}

fn error(code: u16, msg: &str) -> Response<std::io::Cursor<Vec<u8>>> {
    json(code, serde_json::json!({ "error": msg }).to_string())
}

/// `%41` and `+` in a query value.
fn unescape(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        let hex = (b[i] == b'%' && i + 3 <= b.len())
            .then(|| std::str::from_utf8(&b[i + 1..i + 3]).ok().and_then(|h| u8::from_str_radix(h, 16).ok()))
            .flatten();
        match (b[i], hex) {
            (_, Some(x)) => {
                out.push(x);
                i += 2;
            }
            (b'+', _) => out.push(b' '),
            (c, _) => out.push(c),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn query(q: &str) -> Vec<(String, String)> {
    q.split('&')
        .filter(|kv| !kv.is_empty())
        .map(|kv| {
            let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
            (unescape(k), unescape(v))
        })
        .collect()
}

fn mime(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()).unwrap_or("") {
        "html" => "text/html; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "json" | "map" => "application/json",
        "wasm" => "application/wasm",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "ico" => "image/x-icon",
        "woff2" => "font/woff2",
        "woff" => "font/woff",
        "ttf" => "font/ttf",
        "wav" => "audio/wav",
        "txt" | "md" => "text/plain; charset=utf-8",
        "gz" => "application/gzip",
        _ => "application/octet-stream",
    }
}

/// A file of the web app. Only plain path parts are followed, so nothing outside the site is read.
fn site_file(app: &App, url_path: &str) -> Response<std::io::Cursor<Vec<u8>>> {
    let mut path = app.site.clone();
    for part in unescape(url_path).split('/') {
        if part.is_empty() || part == "." {
            continue;
        }
        if part == ".." || part.contains('\\') || part.contains('\0') {
            return error(404, "not found");
        }
        path.push(part);
    }
    if path.is_dir() {
        path.push("index.html");
    }
    match std::fs::read(&path) {
        Ok(bytes) => {
            let cache = if url_path.starts_with("/assets/") {
                "public, max-age=31536000, immutable"
            } else if url_path.starts_with("/fonts/") || url_path.starts_with("/data/") {
                "public, max-age=86400"
            } else {
                "no-cache"
            };
            Response::from_data(bytes)
                .with_header(header("Content-Type", mime(&path)))
                .with_header(header("Cache-Control", cache))
        }
        Err(_) => error(404, "not found"),
    }
}

fn speak(app: &App, s: &Script, plan_only: bool) -> Response<std::io::Cursor<Vec<u8>>> {
    let p: Planned = match script::plan(&app.g2p, s) {
        Ok(p) => p,
        Err(e) => return error(400, &e),
    };
    if plan_only {
        return json(200, serde_json::to_string(&p).unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}")));
    }
    let t = Instant::now();
    match app.gate.run(|| script::render(&p)) {
        Ok(out) => {
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            println!("speak  {:.2} s of sound in {:.0} ms (waiting included)", p.seconds, ms);
            Response::from_data(script::wav(&out))
                .with_header(header("Content-Type", "audio/wav"))
                .with_header(header("Cache-Control", "no-store"))
                .with_header(header("X-Chimera-Seconds", &format!("{:.3}", p.seconds)))
                .with_header(header("X-Chimera-Render-Ms", &format!("{ms:.0}")))
        }
        Err(e) => error(400, &e),
    }
}

fn handle(app: &App, mut req: Request) {
    let url = req.url().to_string();
    let (path, q) = url.split_once('?').unwrap_or((&url, ""));
    let q = query(q);
    let get = |k: &str| q.iter().find(|(a, _)| a == k).map(|(_, v)| v.as_str());
    let plan_only = matches!(get("plan"), Some(v) if v != "0");
    let resp = match (req.method(), path) {
        (Method::Get, "/api/health") => json(200, serde_json::json!({ "ok": true, "version": env!("CARGO_PKG_VERSION"), "words": app.words }).to_string()),
        (Method::Get, "/api/speak") => {
            let num = |k: &str| get(k).and_then(|v| v.parse::<f64>().ok());
            let s = Script {
                text: get("text").map(str::to_string),
                body: get("body").map(Into::into),
                rate: num("rate"),
                f0: num("f0"),
                accent: num("accent"),
                ..Default::default()
            };
            speak(app, &s, plan_only)
        }
        (Method::Post, "/api/speak") => {
            let mut body = String::new();
            match req.as_reader().take(MAX_BODY + 1).read_to_string(&mut body) {
                Ok(n) if n as u64 > MAX_BODY => error(413, "the script is larger than 4 MB"),
                Err(_) => error(400, "the body is not UTF-8 text"),
                Ok(_) => match serde_json::from_str::<Script>(&body) {
                    Ok(s) => speak(app, &s, plan_only),
                    Err(e) => error(400, &format!("not a script: {e}")),
                },
            }
        }
        (_, p) if p.starts_with("/api/") => error(404, "no such call: POST /api/speak, GET /api/speak?text=, GET /api/health"),
        (Method::Get | Method::Head, p) => site_file(app, p),
        _ => error(405, "method not allowed"),
    };
    let _ = req.respond(resp);
}

fn main() {
    let env = |k: &str, d: &str| std::env::var(k).unwrap_or_else(|_| d.to_string());
    let site = PathBuf::from(env("CHIMERA_SITE", "dist"));
    let dict = std::env::var("CHIMERA_DICT").map(PathBuf::from).unwrap_or_else(|_| site.join("data/cmudict.txt.gz"));
    let mut g2p = EnglishG2P::new();
    let mut words = 0;
    match std::fs::read(&dict) {
        Ok(gz) => {
            let mut text = String::new();
            if flate2::read::GzDecoder::new(&gz[..]).read_to_string(&mut text).is_ok() {
                g2p.load_dict(&text);
                words = text.lines().count();
            }
        }
        // without the dictionary every word goes through the letter-to-sound rules
        Err(e) => eprintln!("warning: {}: {e}; reading by spelling rules only", dict.display()),
    }
    chimera::phonetics::planner::init();
    let jobs: usize = env("CHIMERA_JOBS", "2").parse().unwrap_or(2).max(1);
    let app = Arc::new(App { g2p, words, site, gate: Gate { free: Mutex::new(jobs), cv: Condvar::new() } });
    let addr = format!("{}:{}", env("HOST", "127.0.0.1"), env("PORT", "5270"));
    let server = match Server::http(&addr) {
        Ok(s) => Arc::new(s),
        Err(e) => {
            eprintln!("{addr}: {e}");
            std::process::exit(1);
        }
    };
    println!("chimera-serve on http://{addr}  site {}  {} words  {jobs} renders at once", app.site.display(), app.words);
    let workers: Vec<_> = (0..jobs + 4)
        .map(|_| {
            let (server, app) = (server.clone(), app.clone());
            std::thread::spawn(move || {
                while let Ok(req) = server.recv() {
                    handle(&app, req);
                }
            })
        })
        .collect();
    for w in workers {
        let _ = w.join();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_values_are_unescaped() {
        assert_eq!(query("text=Hello%2C+world%21&body=female"), vec![("text".into(), "Hello, world!".into()), ("body".into(), "female".into())]);
        assert_eq!(unescape("100%"), "100%");
        assert_eq!(unescape("%zz"), "%zz");
        assert_eq!(unescape("%C3%A9"), "é");
    }
}
