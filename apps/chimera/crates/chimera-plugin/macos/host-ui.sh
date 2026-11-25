# What ../host-test/run.sh needs on macOS: REAPER from its disk image, screenshots with screencapture,
# windows and dialogs through System Events, and the mouse through cliclick. GitHub's Macs grant those
# to scripts. Screenshots also go to the log as base64, in case the workflow doesn't upload them.

WITH_EDITOR=1
CHECK_FORMATS=CLAP,VST3,AU
# REAPER's title bar and toolbar above the editor, in a window's bounds (points)
CHROME=54
# 12 points in from the corner, macOS still takes a drag for resizing the host's window
GRIP_INSET=16
# what this Mac's sound device runs at, so what REAPER prepares plugins at
RATE=48000
NAG=1

reaper_version=785
reaper_sha=4501a1280ef5ca33e1ad32974917281bfb31514bcfc043fe3e586cba6cba4686
config="$HOME/Library/Application Support/REAPER"

native() { echo "$1"; }

screen_size() {
  osascript -e 'tell application "Finder" to get bounds of window of desktop' 2>&1 || true
}

shot() { # name
  screencapture -x "$out/$1.png" || { echo "no screenshot: $1"; return 0; }
  sips -Z 1400 -s format jpeg -s formatOptions 60 "$out/$1.png" --out "$out/$1.jpg" >/dev/null
  rm -f "$out/$1.png"
  echo "=== screenshot $1 begin"
  base64 -b 200 -i "$out/$1.jpg"
  echo "=== screenshot $1 end"
}

# the editor's window: x y width height
bounds() {
  osascript -e 'tell application "System Events" to tell process "REAPER"
    repeat with w in windows
      if name of w contains "CHIMERA" then
        set {x, y} to position of w
        set {cx, cy} to size of w
        return (x as text) & " " & (y as text) & " " & (cx as text) & " " & (cy as text)
      end if
    end repeat
  end tell' 2>&1 || true
}

click() { # x y
  cliclick "c:$1,$2"
}

drag() { # from x y, to x y
  cliclick "m:$1,$2" w:200 "dd:$1,$2" w:200 "dm:$(($1 - 30)),$(($2 - 20))" w:200 "dm:$3,$4" w:200 "du:$3,$4"
}

front() {
  osascript -e 'tell application "System Events" to tell process "REAPER" to set frontmost to true' 2>&1 || true
}

# REAPER's first-run questions come before it runs any script. Answers one, and says what it asked.
dialogs() {
  osascript -e 'tell application "System Events" to tell process "REAPER"
    repeat with w in windows
      -- a REAPER newer than the one installed says so, over everything (7.86, November 2026)
      if name of w is "REAPER New Version Notification" then
        click button "Close" of w
        return "Close to: a new version"
      end if
      set names to name of buttons of w
      repeat with answer in {"No", "Still Evaluating", "Still evaluating", "I agree", "OK"}
        if names contains (answer as text) then
          set said to value of static texts of w
          click button (answer as text) of w
          return (answer as text) & " to: " & (said as text)
        end if
      end repeat
    end repeat
  end tell' 2>/dev/null || true
}

nag_open() {
  osascript -e 'tell application "System Events" to tell process "REAPER" to get name of every window' 2>/dev/null | grep -q 'About REAPER'
}

install_reaper() {
  HOMEBREW_NO_AUTO_UPDATE=1 brew install --quiet cliclick
  curl -fsSL -o "$work/reaper.dmg" "https://www.reaper.fm/files/7.x/reaper${reaper_version}_universal.dmg"
  echo "$reaper_sha  $work/reaper.dmg" | shasum -a 256 -c -
  yes | hdiutil attach -nobrowse -noverify -mountpoint "$work/mnt" "$work/reaper.dmg" >/dev/null || true
  cp -R "$work/mnt/REAPER.app" "$work/"
  hdiutil detach "$work/mnt" >/dev/null || true
}

fresh_config() {
  rm -rf "$config"
  mkdir -p "$config/Scripts"
  cp "$here/host-test/host_test.lua" "$config/Scripts/__startup.lua"
}

start_reaper() {
  env -u GH_TOKEN "$work/REAPER.app/Contents/MacOS/REAPER" -nosplash >>"$out/reaper.log" 2>&1 &
  reaper_pid=$!
}

reaper_running() {
  kill -0 "$reaper_pid" 2>/dev/null
}

stop_reaper() {
  kill -9 "$reaper_pid" 2>/dev/null || true
  wait "$reaper_pid" 2>/dev/null || true
}

runs() {
  # AU alone, as in a host that only takes AU: all three bundles hold one binary, and in one process
  # their Objective-C classes share names, which a host's lookup of the AU's view trips over.
  local plugins="$HOME/Library/Audio/Plug-Ins"
  mkdir -p "$work/hidden"
  mv "$plugins/CLAP/CHIMERA.clap" "$plugins/VST3/CHIMERA.vst3" "$work/hidden/"
  run_reaper AU
  run_reaper AU "$out/saved-AU.rpp"
  mv "$work/hidden/CHIMERA.clap" "$plugins/CLAP/"
  mv "$work/hidden/CHIMERA.vst3" "$plugins/VST3/"
  run_reaper CLAP,VST3
  run_reaper CLAP,VST3 "$out/saved-CLAP.rpp"
}

report() {
  tail -20 "$out/reaper.log" || true
  grep -il chimera "$config"/*.ini 2>/dev/null | while read -r f; do
    echo "--- $f"
    grep -i -B1 -A1 chimera "$f" | cut -c1-160
  done || true
}
