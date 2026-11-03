# UI tests

Tests of the browser UI (`web/`, docs/ui.md). Three layers:

| file | what | runs with |
|---|---|---|
| `unit_test.mjs` | the pure modules: units, scales, colours, the API client and its decoders (frames.bin, .npy/.npz, inflate against zlib), series building, the run index, the inspection judgement, grid orientation | node 18+ |
| `ui_test.mjs`, `ui_phone_test.mjs` (+ `ui_lib.mjs`) | Playwright (Chromium): every project and run, every tab, every canvas not blank, every channel plotted, every layer, the transport and the 3D view contract, finding links and cause chains, upload, check, a live run, comparison, uncertainty study, calibration; the phone viewport (390 px); no console errors | the `mcr.microsoft.com/playwright:v1.48.2-jammy` image |
| `test_ui_fixtures.py` | the test data: structure and consistency of the synthetic study and calibration overlay; the fake jobs | pytest under bin/tsafe |

The browser tests run against `serve_ui.py`: the real API and UI (`cnctwin.api.create_app`) on port
5995 with a throwaway DATA_DIR built by `ui_fixtures.py` (three synthetic runs of
tests/fixtures/make_synthetic_run.py, the first with an uncertainty study attached; two example
projects) and the job entry points of `fake_jobs.py` (simulate writes a synthetic run step by step;
check, uncertainty and calibrate write documents of the documented formats; comparisons use the real
`cnctwin.report.compare`). When `web/js/view3d/index.js` is absent, the served copy of `web/` gets
`view3d_stub/index.js`, a test double that records every call the shell makes (the tests then check
them); with the real 3D view present it is used as it is.

## Run everything

```
cd /opt/plint/apps/cnc-twin && tests/ui/run_ui_tests.sh
```

It installs `playwright@1.48.2` (library only, `PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1`) into
`/tmp/cnctwin-ui-node` once, runs the unit tests, builds the data under `/tmp/cnctwin-ui`, starts the
server under `bin/tsafe` (TSAFE_MEM 1500M; jobs capped with RLIMIT_DATA 1 GiB), runs both browser
files in one container (`--memory 1g --cpus 1 --user 1002:1002 --rm --network host`), stops the
server and exits 0 when every check passed. Screens (`desktop-<tab>.png`, `phone-<tab>.png`, the
comparison and calibration pages) and `desktop-results.json` / `phone-results.json` land in
`/tmp/cnctwin-ui/out`. About 70 s on this box. Settings: `UI_PORT` (5990-5999, default 5995),
`UI_WORK`, `UI_DEPS`, `UI_SERVER_MEM`.

## Step by step

```
# unit tests (any node 18+)
node tests/ui/unit_test.mjs web/js

# the Playwright library, once
mkdir -p /tmp/cnctwin-ui-node && cd /tmp/cnctwin-ui-node && echo '{"private": true}' > package.json \
  && PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 npm i --no-audit --no-fund playwright@1.48.2

# the server (Ctrl-C stops it); --build writes the synthetic data
cd /opt/plint/apps/cnc-twin
mkdir -p /tmp/cnctwin-ui/out && cp -r web /tmp/cnctwin-ui/web
mkdir -p /tmp/cnctwin-ui/web/js/view3d && [ -f web/js/view3d/index.js ] || cp tests/ui/view3d_stub/index.js /tmp/cnctwin-ui/web/js/view3d/
TSAFE_MEM=1500M bin/tsafe python tests/ui/serve_ui.py --data /tmp/cnctwin-ui/data --port 5995 --web /tmp/cnctwin-ui/web --build

# the browser tests (another shell); VIEW3D_STUB=1 when the stub is served; UPLOAD_ZIP: any zipped project
docker run --rm --network host --user 1002:1002 --memory 1g --cpus 1 -e HOME=/tmp \
  -e UI_BASE=http://127.0.0.1:5995 -e VIEW3D_STUB=1 -e UPLOAD_ZIP=/out/upload.zip -e UI_OUT=/out \
  -v /tmp/cnctwin-ui-node:/deps:ro -v /opt/plint/apps/cnc-twin/tests/ui:/tests:ro -v /tmp/cnctwin-ui/out:/out \
  mcr.microsoft.com/playwright:v1.48.2-jammy bash -c 'node /tests/ui_test.mjs && node /tests/ui_phone_test.mjs'

# the fixtures (pytest); CNCTWIN_UI_TESTS=1 adds the browser suite through run_ui_tests.sh
bin/tsafe python -m pytest -q tests/ui
```

The upload test needs `/tmp/cnctwin-ui/out/upload.zip` (run_ui_tests.sh writes it from
`write_synthetic_project`). Open `http://127.0.0.1:5995/` in a browser on this box to look at the
UI with the test data.
