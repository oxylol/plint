import { useEffect, useRef } from "react";

export interface CanvasFrame {
  ctx: CanvasRenderingContext2D;
  w: number; // css px
  h: number;
  dpr: number;
  dt: number;
  now: number;
}

/**
 * Scale a context so one unit is one CSS pixel of its canvas, w x h. The backing store is rounded
 * to whole pixels, so the ratio is the store's to the box, not devicePixelRatio: a box 54.5 px tall
 * gets a 55 px store, and drawing at dpr shrank everything 1% (an 11 px label showed at 10.9).
 */
export function cssScale(ctx: CanvasRenderingContext2D, w: number, h: number) {
  ctx.setTransform(ctx.canvas.width / w, 0, 0, ctx.canvas.height / h, 0, 0);
}

/** Canvas with devicePixelRatio scaling, resize observer and a rAF draw loop (ctx pre-scaled to css px). */
export function useCanvasLoop(draw: (f: CanvasFrame) => void) {
  const ref = useRef<HTMLCanvasElement>(null);
  const drawRef = useRef(draw);
  drawRef.current = draw;
  useEffect(() => {
    const canvas = ref.current!;
    const ctx = canvas.getContext("2d")!;
    let w = 0, h = 0, dpr = 1;
    const ro = new ResizeObserver(() => {
      const r = canvas.getBoundingClientRect();
      dpr = Math.min(2, window.devicePixelRatio || 1);
      w = r.width;
      h = r.height;
      canvas.width = Math.max(1, Math.round(w * dpr));
      canvas.height = Math.max(1, Math.round(h * dpr));
    });
    ro.observe(canvas);
    let raf = 0;
    let last = performance.now();
    const loop = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      if (w > 0 && h > 0) {
        cssScale(ctx, w, h);
        drawRef.current({ ctx, w, h, dpr, dt, now });
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, []);
  return ref;
}
