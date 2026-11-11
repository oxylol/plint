// fetch wrapper: JSON in and out, the CSRF header, and a sign-in prompt on 401.
let onUnauthorized = () => {};
export function setUnauthorized(fn) { onUnauthorized = fn; }

export async function api(path, opts = {}) {
  const r = await fetch(`/api/${path}`, {
    method: opts.method || (opts.body !== undefined ? "POST" : "GET"),
    headers: { "content-type": "application/json", "x-plint": "1" },
    body: opts.body !== undefined ? JSON.stringify(opts.body) : undefined,
  });
  const data = await r.json().catch(() => ({}));
  if (r.status === 401 && !path.startsWith("login")) onUnauthorized();
  if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
  return data;
}

export async function upload(path, file, onProgress) {
  return new Promise((resolve, reject) => {
    const x = new XMLHttpRequest();
    x.open("POST", `/api/${path}`);
    x.setRequestHeader("x-plint", "1");
    x.setRequestHeader("x-filename", encodeURIComponent(file.name || "file"));
    x.setRequestHeader("content-type", file.type || "application/octet-stream");
    x.upload.onprogress = (e) => e.lengthComputable && onProgress?.(e.loaded / e.total);
    x.onload = () => {
      let data = {};
      try { data = JSON.parse(x.responseText); } catch {}
      x.status < 300 ? resolve(data) : reject(new Error(data.error || `Upload failed (${x.status})`));
    };
    x.onerror = () => reject(new Error("Upload failed"));
    x.send(file);
  });
}

export async function text(path) {
  const r = await fetch(`/api/${path}`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.text();
}
