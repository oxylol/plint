import type { ButtonHTMLAttributes, ReactNode } from "react";

/**
 * One screen on the wall: a black panel with a thin frame and a single-stroke title.
 * The body always has a positive size (see .screen-body in styles.css); canvases fill it.
 * A screen whose content names itself (the head) has no title: `lead` takes its place in the
 * header, and `label` names the screen for assistive tech.
 * `watch` marks a screen that only shows (nothing in it to touch): on a phone it goes below every
 * screen you touch (AGENTS.md; the layout lint checks the order).
 */
export function Screen(props: { title?: string; sub?: string; lead?: ReactNode; focus?: boolean; watch?: boolean; className?: string; tools?: ReactNode; children: ReactNode; label?: string }) {
  const { title, sub, lead, focus, watch, className, tools, children, label } = props;
  return (
    <section className={`screen${focus ? " focus" : ""}${watch ? " watch" : ""}${className ? " " + className : ""}`} aria-label={label ?? title}>
      {(title || lead || sub || tools) && (
        <header className="screen-head">
          {title && <h2 className="screen-title">{title}</h2>}
          {lead}
          {sub && <span className="screen-sub">{sub}</span>}
          {tools && <div className="screen-tools">{tools}</div>}
        </header>
      )}
      <div className="screen-body">{children}</div>
    </section>
  );
}

/** A framed caps button; `on` lights it and sets aria-pressed (a toggle) unless it's a plain action. */
export function Chip(props: ButtonHTMLAttributes<HTMLButtonElement> & { on?: boolean; toggle?: boolean; tone?: "slot0" | "slot1" | "slot2" | "slot3" | "amber" }) {
  const { on, toggle = true, tone, className, children, ...rest } = props;
  return (
    <button type="button" {...rest} className={`chip${on ? " on" : ""}${tone ? " " + tone : ""}${className ? " " + className : ""}`} aria-pressed={toggle ? !!on : undefined}>
      {children}
    </button>
  );
}
