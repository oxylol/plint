import { useEffect, useRef, type ReactNode } from "react";
import { useStore } from "../store.ts";

/**
 * The top of each tab's right column: the one bright button for what the tab is for, in the same
 * place on every tab, and the few settings that belong beside it. Its button starts the audio too
 * (there's no separate start step), and failing to start shows here, as does a device too slow
 * to run the voice in real time. Space presses it, unless the focus is in a field or on another
 * control.
 */
export function ActionBar(props: { label: string; onSpace: () => void; children: ReactNode }) {
  const err = useStore((s) => s.startError);
  const overloaded = useStore((s) => s.overloaded);
  const onSpace = useRef(props.onSpace);
  onSpace.current = props.onSpace;
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== "Space" || e.repeat || e.metaKey || e.ctrlKey || e.altKey || e.shiftKey) return;
      // Space belongs to whatever has the focus: typing, a button, a dial, the body pad
      const t = e.target as Element | null;
      if (t && t !== document.body && t.closest("input, textarea, select, button, a, [contenteditable], [tabindex]")) return;
      e.preventDefault();
      onSpace.current();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  return (
    <section className="screen action-bar" aria-label={props.label}>
      <div className="screen-body">
        <div className="act-row">{props.children}</div>
        {err && (
          <p className="act-error" role="alert">
            Audio could not start: {err}. Press again to retry.
          </p>
        )}
        {overloaded && (
          <p className="act-error" role="status">
            This device can't keep up, so the voice breaks up or goes silent. Plugging in, or closing other tabs, may help.
          </p>
        )}
      </div>
    </section>
  );
}

/**
 * The bright button (amber: it makes sound). While its sound plays it reads Stop. While the audio
 * starts, every main button waits, and the one pressed (`pressed`, default all) says so.
 */
export function PrimaryButton(props: { label: string; playing?: boolean; busy?: string | null; pressed?: boolean; icon?: "play" | "breath"; className?: string; title?: string; onClick: () => void }) {
  const starting = useStore((s) => s.starting);
  const text = starting && props.pressed !== false ? "Starting…" : props.busy ? props.busy : props.playing ? "Stop" : props.label;
  return (
    <button type="button" className={`act-primary${props.playing ? " on" : ""}${props.className ? " " + props.className : ""}`} onClick={props.onClick} disabled={starting || !!props.busy} title={props.title}>
      <span className={`act-icon ${props.playing ? "stop" : (props.icon ?? "play")}`} aria-hidden="true" />
      {text}
    </button>
  );
}
