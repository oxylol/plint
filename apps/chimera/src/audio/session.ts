/**
 * iOS plays Web Audio in the "ambient" audio session, which the ring/silent switch mutes: on an
 * iPhone in silent mode the page makes no sound at all, and the volume buttons don't change that.
 * The "playback" session plays through the switch, like a video or a music app. Safari 16.4+
 * exposes it as `navigator.audioSession`; elsewhere this does nothing.
 *
 * Call it before audio starts (from the tap that starts it).
 */
type AudioSessionLike = { type: string };

export function preferPlaybackSession(nav: unknown = typeof navigator === "undefined" ? undefined : navigator): void {
  const session = (nav as { audioSession?: AudioSessionLike } | undefined)?.audioSession;
  if (!session || session.type === "playback") return;
  try {
    session.type = "playback";
  } catch {
    // a browser that refuses: the default session still plays wherever the switch isn't on silent
  }
}
