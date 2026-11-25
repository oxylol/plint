/**
 * iPhone in silent mode: Web Audio in iOS's default "ambient" session is muted by the ring/silent
 * switch, so Play made no sound whatever the volume. Audio asks for the "playback" session first.
 */
import { afterEach, describe, expect, test } from "bun:test";
import { preferPlaybackSession } from "../src/audio/session.ts";
import { Transport } from "../src/show/transport.ts";

describe("preferPlaybackSession", () => {
  test("switches iOS's ambient session to playback, which the silent switch doesn't mute", () => {
    const nav = { audioSession: { type: "auto" } };
    preferPlaybackSession(nav);
    expect(nav.audioSession.type).toBe("playback");
  });

  test("does nothing where there is no audioSession (every browser but Safari)", () => {
    expect(() => preferPlaybackSession({})).not.toThrow();
    expect(() => preferPlaybackSession(undefined)).not.toThrow();
  });

  test("a browser that refuses the session doesn't stop the audio from starting", () => {
    const nav = {
      audioSession: {
        get type() {
          return "auto";
        },
        set type(_: string) {
          throw new Error("NotAllowedError");
        },
      },
    };
    expect(() => preferPlaybackSession(nav)).not.toThrow();
  });
});

describe("the duet page's Play", () => {
  const g = globalThis as { navigator?: unknown };
  const saved = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  afterEach(() => {
    if (saved) Object.defineProperty(globalThis, "navigator", saved);
    else delete g.navigator;
  });

  test("asks for the playback session in the tap, before the audio resumes", async () => {
    const session = { type: "auto" };
    let typeAtResume = "";
    Object.defineProperty(globalThis, "navigator", { value: { audioSession: session }, configurable: true });
    const ctx = {
      state: "suspended",
      onstatechange: null,
      destination: {},
      currentTime: 0,
      createBufferSource: () => ({ connect() {}, disconnect() {}, start() {}, stop() {}, onended: null, buffer: null }),
      decodeAudioData: async () => ({ duration: 10 }),
      resume: async () => {
        typeAtResume = session.type;
      },
      close: async () => {},
    };
    const t = new Transport(ctx as unknown as AudioContext);
    await t.load(new ArrayBuffer(0));
    t.play();
    expect(typeAtResume).toBe("playback");
  });
});
