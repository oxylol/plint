/* The run page's clock: simulated time t (s), playing at real speed (1x) or a multiple, seeking,
 * and stepping block by block. Everything that shows time (the 3D view, DRO, program viewer,
 * plots, scrubber) listens to its 'time' event; nothing else moves time.
 *
 * Playback advances t by (wall-clock seconds since the last frame, from the moment play is pressed)
 * x speed on each animation frame, so 1x is real time whatever the frame rate; one frame's step is
 * capped at 0.5 s of wall time; it stops at the end. Listeners run at most once per animation frame
 * however often seek() is called.
 */

export const SPEEDS = [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 25, 100];
/** Longest wall-clock step of one frame: a tab that was hidden or stalled resumes, it does not jump. */
const MAX_FRAME_S = 0.5;

export class Timebase extends EventTarget {
  constructor({ duration = 0, t = 0 } = {}) {
    super();
    this.duration = Math.max(0, duration);
    this.t = Math.max(0, Math.min(t, this.duration));
    this.playing = false;
    this.speed = 1;
    this._frame = null;
    this._last = null;
    this._pending = false;
    this._loop = this._loop.bind(this);
  }

  setDuration(duration) {
    this.duration = Math.max(0, duration);
    if (this.t > this.duration) this.seek(this.duration);
    this.dispatchEvent(new CustomEvent('duration', { detail: this.duration }));
  }

  /** Move to time t (clamped); `source` names who moved it (listeners may ignore their own). */
  seek(t, source = null) {
    const next = Math.max(0, Math.min(Number(t) || 0, this.duration));
    this.t = next;
    this.source = source;
    this._emit();
  }

  _emit() {
    if (this._pending) return;
    this._pending = true;
    requestAnimationFrame(() => {
      this._pending = false;
      this.dispatchEvent(new CustomEvent('time', { detail: { t: this.t, source: this.source, playing: this.playing } }));
    });
  }

  play() {
    if (this.playing || this.duration <= 0) return;
    if (this.t >= this.duration - 1e-9) this.t = 0;
    this.playing = true;
    this._last = performance.now();
    this._frame = requestAnimationFrame(this._loop);
    this.dispatchEvent(new CustomEvent('state', { detail: { playing: true } }));
  }

  pause() {
    if (!this.playing) return;
    this.playing = false;
    if (this._frame) cancelAnimationFrame(this._frame);
    this._frame = null;
    this.dispatchEvent(new CustomEvent('state', { detail: { playing: false } }));
    this._emit();
  }

  toggle() { if (this.playing) this.pause(); else this.play(); }

  setSpeed(speed) {
    this.speed = Math.max(1e-4, Number(speed) || 1);
    this.dispatchEvent(new CustomEvent('speed', { detail: this.speed }));
  }

  _loop(now) {
    if (!this.playing) return;
    if (this._last !== null) {
      const dt = Math.min(MAX_FRAME_S, Math.max(0, now - this._last) / 1000) * this.speed;
      this.t = Math.min(this.duration, this.t + dt);
      this.source = 'play';
      this.dispatchEvent(new CustomEvent('time', { detail: { t: this.t, source: 'play', playing: true } }));
      if (this.t >= this.duration) { this.pause(); return; }
    }
    this._last = now;
    this._frame = requestAnimationFrame(this._loop);
  }

  destroy() {
    this.pause();
  }
}
