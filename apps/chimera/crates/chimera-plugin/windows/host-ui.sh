# What ../host-test/run.sh needs on Windows, in Git Bash: REAPER from its installer, and the screen,
# windows, dialogs and mouse through PowerShell (ui.ps1).

WITH_EDITOR=1
CHECK_FORMATS=CLAP,VST3
# REAPER's toolbar above the editor, in the client area of its window (pixels)
CHROME=30
GRIP_INSET=16
RATE=48000
NAG=1

reaper_version=785
reaper_sha=4d30d0e6df3e05d4df594ed7b1ce6465e95c9f3b15a3457ca95730c3f26fc2a9
reaper_exe="/c/Program Files/REAPER (x64)/reaper.exe"
config="$(cygpath -u "$APPDATA")/REAPER"

ui() {
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(cygpath -w "$here/windows/ui.ps1")" "$@" | tr -d '\r'
}

native() { cygpath -w "$1"; }

screen_size() { ui screen; }

shot() { # name
  ui shot "$(cygpath -w "$out/$1.jpg")" || echo "no screenshot: $1"
}

# the editor's window, its client area on screen: x y width height
bounds() { ui bounds; }

click() { ui click "$1" "$2"; }

drag() { ui drag "$1" "$2" "$3" "$4"; }

front() { ui front || true; }

dialogs() { ui dialogs || true; }

nag_open() { [ -n "$(ui nag)" ]; }

install_reaper() {
  curl -fsSL -o "$work/reaper-install.exe" "https://www.reaper.fm/files/7.x/reaper${reaper_version}_x64-install.exe"
  echo "$reaper_sha  $work/reaper-install.exe" | sha256sum -c -
  # Git Bash would turn /S into a path, and the installer would wait for someone to click through it
  MSYS_NO_PATHCONV=1 "$work/reaper-install.exe" /S
  for _ in $(seq 1 60); do
    [ -e "$reaper_exe" ] && break
    sleep 1
  done
  [ -e "$reaper_exe" ] || { echo "REAPER didn't install"; return 1; }
}

fresh_config() {
  rm -rf "$config"
  mkdir -p "$config/Scripts"
  cp "$here/host-test/host_test.lua" "$config/Scripts/__startup.lua"
}

start_reaper() {
  env -u GH_TOKEN "$reaper_exe" -nosplash >>"$out/reaper.log" 2>&1 &
  reaper_pid=$!
}

reaper_running() {
  tasklist //FI "IMAGENAME eq reaper.exe" 2>/dev/null | grep -qi reaper.exe
}

stop_reaper() {
  taskkill //F //IM reaper.exe >/dev/null 2>&1 || true
  wait "$reaper_pid" 2>/dev/null || true
}

runs() {
  run_reaper CLAP,VST3
  run_reaper CLAP,VST3 "$out/saved-CLAP.rpp"
}

report() {
  ui windows || true
  tail -20 "$out/reaper.log" || true
  grep -il chimera "$config"/*.ini 2>/dev/null | while read -r f; do
    echo "--- $f"
    grep -i -B1 -A1 chimera "$f" | cut -c1-160
  done || true
}
