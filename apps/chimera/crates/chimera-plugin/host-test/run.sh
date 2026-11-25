#!/bin/bash
# CHIMERA in a real host: REAPER loads the plugin that `<system>/build.sh --install` put in place,
# renders what each mode does (host_test.lua, judged by check.ts), and opens each editor for this
# script to look at and click. GitHub's runners have a display that scripts can capture and click.
# What differs between systems (installing REAPER, the screen, the mouse, dialogs) is in
# `<system>/host-ui.sh`. See docs/plugin.md, "Testing in a host".
set -euo pipefail
# from this file, not git: validate.sh runs it from a folder of downloaded tools
here=$(cd "$(dirname "$0")/.." && pwd)
root=$(cd "$here/../.." && pwd)
out="$root/target/host-test"
work=$(mktemp -d)
rm -rf "$out"
mkdir -p "$out"

case "$(uname -s)" in
  Darwin) system=macos ;;
  MINGW* | MSYS* | CYGWIN*) system=windows ;;
  *) system=linux ;;
esac
# shellcheck source=/dev/null
source "$here/$system/host-ui.sh"

# Where the tabs' middles are in the page: from its right edge, as the top bar keeps them there, and
# from its top (measured in the UI harness). The page is scaled below LAYOUT_MIN (page_zoom in
# src/bridge.rs), and since it's drawn scaled, text included, these hold at any window size.
# tools/plugin-e2e.ts checks them against the page, so a change to the top bar fails there first.
PLAY_FROM_RIGHT=496
BEATBOX_FROM_RIGHT=351
TAB_Y=26
LAYOUT_W=1120
LAYOUT_H=800
# how far the grip is dragged in: check.ts's GRIP_DRAG
DRAG_X=100
DRAG_Y=60

# set once an evaluation copy's nag has been answered
nag_done=0

answer_dialogs() {
  local answered
  answered=$(dialogs)
  [ -z "$answered" ] && return 0
  echo "answered $answered"
  case "$answered" in *Evaluating*) nag_done=1 ;; esac
}

# An evaluation copy's nag comes up about half a minute in, and covers everything until its countdown
# lets dialogs() answer it.
wait_for_nag() {
  local i seen=0
  [ "$NAG" = 1 ] || return 0
  for i in $(seq 1 90); do
    [ "$nag_done" = 1 ] && return 0
    answer_dialogs
    if nag_open; then
      seen=1
    elif [ "$seen" = 1 ]; then
      return 0
    fi
    sleep 1
  done
  echo "::warning::REAPER's nag didn't come and go"
}

# A tab in the page, wherever the window is and whatever its zoom: x from the page's right edge, window's x y w h
click_tab() {
  local zoom view_h tab_x tab_y
  view_h=$(($5 - CHROME))
  zoom=$(awk -v w="$4" -v h="$view_h" -v lw="$LAYOUT_W" -v lh="$LAYOUT_H" 'BEGIN { z = w / lw; if (h / lh < z) z = h / lh; if (z > 1) z = 1; print z }')
  tab_x=$(awk -v x="$2" -v w="$4" -v z="$zoom" -v t="$1" 'BEGIN { printf "%d", x + w - z * t }')
  tab_y=$(awk -v y="$3" -v z="$zoom" -v c="$CHROME" -v t="$TAB_Y" 'BEGIN { printf "%d", y + c + z * t }')
  echo "zoom $zoom, clicking at $tab_x,$tab_y"
  click "$tab_x" "$tab_y"
}

editor() { # format
  local fmt=$1 x y w h gx gy
  front
  shot "$fmt-open"
  read -r x y w h <<<"$(bounds)" || true
  echo "$fmt window: ${x:-?} ${y:-?} ${w:-?} ${h:-?}"
  case "${h:-}" in '' | *[!0-9]*) return 0 ;; esac
  # REAPER's generic sliders come in a small window of their own
  if [ "$w" -lt 700 ]; then
    echo "::warning::$fmt shows REAPER's generic sliders, not the plugin's view"
    return 0
  fi
  click_tab "$BEATBOX_FROM_RIGHT" "$x" "$y" "$w" "$h"
  sleep 3
  shot "$fmt-beatbox"
  gx=$((x + w - GRIP_INSET))
  gy=$((y + h - GRIP_INSET))
  drag "$gx" "$gy" $((gx - DRAG_X)) $((gy - DRAG_Y))
  sleep 3
  { echo "$x $y $w $h"; bounds; } >"$out/bounds-$fmt.txt"
  echo "$fmt window after the grip: $(tail -1 "$out/bounds-$fmt.txt")"
  shot "$fmt-resized"
}

run_reaper() { # formats, and a project to reopen instead of starting afresh
  local deadline marker step i bx by bw bh
  rm -f "$out/finished"
  # A first run, unless it's coming back to a project. A launch after REAPER was killed got stuck with
  # nothing on screen to answer, so each one here gets to quit by itself.
  [ -n "${2:-}" ] || fresh_config
  nag_done=0
  export CHIMERA_TEST_OUT CHIMERA_TEST_FORMATS CHIMERA_TEST_EDITOR CHIMERA_TEST_RATE CHIMERA_TEST_REOPEN
  CHIMERA_TEST_OUT=$(native "$out")
  CHIMERA_TEST_FORMATS=$1
  CHIMERA_TEST_EDITOR=$WITH_EDITOR
  CHIMERA_TEST_RATE=$RATE
  CHIMERA_TEST_REOPEN=$([ -n "${2:-}" ] && native "$2" || true)
  start_reaper
  deadline=$((SECONDS + 420))
  while [ ! -e "$out/finished" ]; do
    if [ "$SECONDS" -gt "$deadline" ]; then
      echo "::warning::REAPER didn't finish in time"
      shot "stuck-$1"
      break
    fi
    if ! reaper_running; then
      echo "::warning::REAPER stopped by itself"
      break
    fi
    answer_dialogs
    for marker in "$out"/at-*; do
      [ -e "$marker" ] || continue
      step=${marker##*/at-}
      rm "$marker"
      case "$step" in
        # a second launch has neither the first-run questions nor, it turns out, the nag
        ready) if [ -z "${2:-}" ]; then wait_for_nag; fi ;;
        editor-*) editor "${step#editor-}" ;;
        shot-*) shot "${step#shot-}" ;;
        click-play-*)
          read -r bx by bw bh <<<"$(bounds)" || true
          case "${bh:-}" in '' | *[!0-9]*) echo "no window to click" ;; *) click_tab "$PLAY_FROM_RIGHT" "$bx" "$by" "$bw" "$bh" ;; esac
          ;;
        reopened-*)
          bounds >"$out/$step.txt"
          echo "$step window: $(cat "$out/$step.txt")"
          shot "$step"
          ;;
      esac
      touch "$out/go-$step"
    done
    sleep 0.5
  done
  for i in $(seq 1 20); do
    reaper_running || break
    sleep 1
  done
  stop_reaper
  rm -f "$out"/go-*
}

echo "screen: $(screen_size)"
install_reaper
runs
report

editor_flag=()
[ "$WITH_EDITOR" = 1 ] && editor_flag=(--editor --chrome "$CHROME")
bun "$here/host-test/check.ts" "$out" "$CHECK_FORMATS" ${editor_flag[@]+"${editor_flag[@]}"}
