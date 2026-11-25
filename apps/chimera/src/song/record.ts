/**
 * A video of the head singing: the face screen's canvas copied each frame into a 9:16 frame (black
 * above and below, which the vector look already is) with the wordmark at the foot, and the voice
 * from the audio graph, through the browser's MediaRecorder. MP4 where the browser records it
 * (Safari, recent Chrome: what Instagram and TikTok take), else WebM. While a clip records, the
 * face screens leave out their measurements (`recording`), so what you see is what you get.
 */
import { label } from "../ui/render/vector.ts";
import { V } from "../ui/colors.ts";

const W = 720, H = 1280;
/** the band at the foot that holds the wordmark */
const FOOT = 150;
const TYPES = ["video/mp4;codecs=avc1.42E01E,mp4a.40.2", "video/mp4", "video/webm;codecs=vp9,opus", "video/webm;codecs=vp8,opus", "video/webm"];

function pickType(): string | null {
  if (typeof MediaRecorder === "undefined" || typeof HTMLCanvasElement === "undefined" || !("captureStream" in HTMLCanvasElement.prototype)) return null;
  return TYPES.find((t) => MediaRecorder.isTypeSupported(t)) ?? null;
}

/**
 * A clip is recording: the face screens draw the head without readouts or measurements. Owned by
 * the clip that set it, so an earlier clip finishing late can't clear a newer one's.
 */
export const recording = {
  owner: null as Clip | null,
  get on(): boolean {
    return this.owner !== null;
  },
};

/** whether this browser can record a clip at all */
export const canRecord = (): boolean => pickType() !== null;

export class Clip {
  private frame = document.createElement("canvas");
  private rec!: MediaRecorder;
  private chunks: Blob[] = [];
  private raf = 0;
  private type: string;
  /** the frame's capture track: ours to stop (the audio track is the engine's, shared) */
  private video: MediaStreamTrack[] = [];
  /** settles once the recorder has stopped, however it stopped (stop(), cancel(), or a failure) */
  private ended!: Promise<void>;
  /** the recorder reported an error: what it holds isn't a whole clip */
  private failed = false;

  /**
   * Starts recording `source()`, read every frame (the face's canvas changes when 2D/3D is
   * toggled); throws (having let go of everything it took) if the browser can't.
   */
  constructor(source: () => HTMLCanvasElement | null, audio: MediaStream) {
    this.type = pickType()!;
    recording.owner = this;
    try {
      this.begin(source, audio);
    } catch (e) {
      this.release();
      throw e;
    }
  }

  private begin(source: () => HTMLCanvasElement | null, audio: MediaStream): void {
    this.frame.width = W;
    this.frame.height = H;
    const ctx = this.frame.getContext("2d")!;
    const draw = () => {
      ctx.globalCompositeOperation = "source-over";
      ctx.globalAlpha = 1;
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, W, H);
      const src = source();
      const sw = src?.width ?? 0, sh = src?.height ?? 0;
      if (src && sw > 0 && sh > 0) {
        const k = Math.min(W / sw, (H - FOOT) / sh);
        ctx.drawImage(src, (W - sw * k) / 2, (H - FOOT - sh * k) / 2, sw * k, sh * k);
      }
      label(ctx, "CHIMERA", W / 2, H - FOOT / 2, { size: 30, color: V.blue, weight: "m", ls: 0.34, align: "center" });
      this.raf = requestAnimationFrame(draw);
    };
    draw();
    this.video = this.frame.captureStream(30).getVideoTracks();
    this.rec = new MediaRecorder(new MediaStream([...this.video, ...audio.getAudioTracks()]), { mimeType: this.type, videoBitsPerSecond: 5_000_000 });
    this.rec.ondataavailable = (e) => e.data.size && this.chunks.push(e.data);
    // a recorder can also stop by itself (an encoder or resource failure): it lets go of the
    // capture and shows the measurements again then, not when the song ends
    this.rec.onerror = () => (this.failed = true);
    this.ended = new Promise((resolve) => {
      this.rec.onstop = () => {
        this.release();
        resolve();
      };
    });
    this.rec.start(500);
  }

  /** Finish: the clip as a file named `name` (the extension is added); rejects if the recorder failed. */
  async stop(name: string): Promise<File> {
    cancelAnimationFrame(this.raf);
    // the face screens show their measurements again now, not when the browser hands the file over
    if (recording.owner === this) recording.owner = null;
    if (this.rec.state !== "inactive") this.rec.stop();
    await this.ended;
    if (this.failed || !this.chunks.length) throw new Error("the recording failed");
    const type = this.type.split(";")[0];
    return new File(this.chunks, `${name}.${type === "video/mp4" ? "mp4" : "webm"}`, { type });
  }

  /** Throw it away. */
  cancel(): void {
    if (this.rec.state !== "inactive") this.rec.stop();
    this.release();
  }

  private release(): void {
    if (recording.owner === this) recording.owner = null;
    cancelAnimationFrame(this.raf);
    for (const t of this.video) t.stop();
  }
}
