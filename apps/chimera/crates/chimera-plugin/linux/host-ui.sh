# What ../host-test/run.sh needs on Linux: REAPER from its tarball, under a virtual display. The
# editor on Linux is the page in a web browser (docs/plugin.md, "Linux"), which this doesn't drive,
# so nothing here looks at the screen or clicks.

WITH_EDITOR=0
CHECK_FORMATS=CLAP,VST3
CHROME=0
GRIP_INSET=16
# as on the other systems: no sound device here, so nothing prepares plugins at another rate
RATE=48000
NAG=0

reaper_version=785
reaper_sha=b962c40a51535d2faf9796f56270f9e331c9d424b5741472277d7bfd6c4893ac
config="$HOME/.config/REAPER"

native() { echo "$1"; }
screen_size() { echo "virtual, 1600 x 1000"; }
shot() { :; }
bounds() { :; }
click() { :; }
drag() { :; }
front() { :; }
dialogs() { :; }
nag_open() { return 1; }

install_reaper() {
  curl -fsSL -o "$work/reaper.tar.xz" "https://www.reaper.fm/files/7.x/reaper${reaper_version}_linux_x86_64.tar.xz"
  echo "$reaper_sha  $work/reaper.tar.xz" | sha256sum -c -
  tar -xf "$work/reaper.tar.xz" -C "$work"
}

fresh_config() {
  rm -rf "$config"
  mkdir -p "$config/Scripts"
  cp "$here/host-test/host_test.lua" "$config/Scripts/__startup.lua"
}

start_reaper() {
  env -u GH_TOKEN xvfb-run -a -s "-screen 0 1600x1000x24" "$work/reaper_linux_x86_64/REAPER/reaper" -nosplash >>"$out/reaper.log" 2>&1 &
  reaper_pid=$!
}

reaper_running() {
  kill -0 "$reaper_pid" 2>/dev/null
}

stop_reaper() {
  kill "$reaper_pid" 2>/dev/null || true
  # xvfb-run's child; a pattern that can't match this shell's own command line
  pkill -f "reaper_linux_x86_64/REAPER/reape[r]" 2>/dev/null || true
  wait "$reaper_pid" 2>/dev/null || true
}

runs() {
  run_reaper CLAP,VST3
}

report() {
  tail -20 "$out/reaper.log" || true
}
