#!/bin/bash
# CNC Twin UI tests (tests/ui/README.md): the unit tests of the UI's pure modules (node), then the
# Playwright test in the mcr.microsoft.com/playwright:v1.48.2-jammy image against a throwaway API
# server on port 5995 (bin/tsafe) that serves synthetic runs, example projects and fake jobs.
# Exit status 0 when every check passed. Screens and results: $UI_WORK/out (default /tmp/cnctwin-ui).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
work="${UI_WORK:-/tmp/cnctwin-ui}"
deps="${UI_DEPS:-/tmp/cnctwin-ui-node}"
port="${UI_PORT:-5995}"
image="mcr.microsoft.com/playwright:v1.48.2-jammy"
server_pid=""

stop_server() {
  # the server runs under sudo (bin/tsafe), which passes SIGTERM on to it
  if [ -n "$server_pid" ]; then
    kill -TERM "$server_pid" 2>/dev/null || true
    for _ in $(seq 1 40); do kill -0 "$server_pid" 2>/dev/null || break; sleep 0.25; done
    if kill -0 "$server_pid" 2>/dev/null; then
      pkill -KILL -f "^python tests/ui/serve_ui.py --data $work/data" 2>/dev/null || true
    fi
    wait "$server_pid" 2>/dev/null || true
    server_pid=""
  fi
}
trap stop_server EXIT

if ss -ltn 2>/dev/null | grep -q ":$port "; then
  echo "port $port is in use: stop what listens there or set UI_PORT (5990-5999)" >&2
  exit 2
fi
rm -rf "$work"
mkdir -p "$work/out" "$deps"
chmod 777 "$work/out"

# 1. the Playwright library (the browsers are in the image)
if [ ! -f "$deps/node_modules/playwright/index.mjs" ]; then
  [ -f "$deps/package.json" ] || echo '{"name": "cnctwin-ui-tests", "private": true}' > "$deps/package.json"
  (cd "$deps" && PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 npm i --no-audit --no-fund playwright@1.48.2)
fi

# 2. unit tests of the pure modules
echo "== unit tests (node)"
if command -v node > /dev/null; then
  node "$here/unit_test.mjs" "$root/web/js"
else
  docker run --rm --user 1002:1002 --memory 1g --cpus 1 -e HOME=/tmp -v "$root/web/js:/web/js:ro" -v "$here:/tests:ro" \
    "$image" node /tests/unit_test.mjs /web/js
fi

# 3. the served UI: web/, with the 3D view test double when web/js/view3d is absent
cp -r "$root/web" "$work/web"
stub=0
if [ ! -f "$root/web/js/view3d/index.js" ]; then
  mkdir -p "$work/web/js/view3d"
  cp "$here/view3d_stub/index.js" "$work/web/js/view3d/index.js"
  stub=1
  echo "== web/js/view3d is absent: serving the test double tests/ui/view3d_stub"
fi

# 4. data, an upload, the server
cd "$root"
bin/tsafe python -c "
import sys, zipfile, pathlib
sys.path.insert(0, 'tests/ui')
import ui_fixtures
ui_fixtures.build_data_dir(pathlib.Path('$work/data'))
src = pathlib.Path('$work/upload-src')
ui_fixtures.synthetic.write_synthetic_project(src, seed=6, name='UI upload')
with zipfile.ZipFile('$work/out/upload.zip', 'w') as archive:
    for path in sorted(src.iterdir()):
        archive.write(path, path.name)
"
TSAFE_MEM="${UI_SERVER_MEM:-1500M}" bin/tsafe python tests/ui/serve_ui.py --data "$work/data" --port "$port" --web "$work/web" \
  > "$work/server.log" 2>&1 &
server_pid=$!
for _ in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$port/api/health" > /dev/null && break
  sleep 0.5
done
curl -sf "http://127.0.0.1:$port/api/health" > /dev/null || { echo "the server did not start:" >&2; cat "$work/server.log" >&2; exit 1; }

# 5. the browser test
echo "== Playwright (Chromium) against http://127.0.0.1:$port"
status=0
docker run --rm --network host --user 1002:1002 --memory 1g --cpus 1 -e HOME=/tmp \
  -e UI_BASE="http://127.0.0.1:$port" -e VIEW3D_STUB="$stub" -e UPLOAD_ZIP=/out/upload.zip -e UI_OUT=/out \
  -v "$deps:/deps:ro" -v "$here:/tests:ro" -v "$work/out:/out" \
  "$image" bash -c 'node /tests/ui_test.mjs; a=$?; node /tests/ui_phone_test.mjs; b=$?; [ "$a" -eq 0 ] && [ "$b" -eq 0 ]' || status=$?
stop_server
if [ "$status" -ne 0 ]; then
  echo "== server log (tail)"; tail -40 "$work/server.log" || true
fi
exit "$status"
