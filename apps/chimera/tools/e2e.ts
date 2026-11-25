/**
 * Headless-Chromium smoke test: loads the app, starts audio, interacts, collects console errors,
 * measures worklet load, and saves screenshots.
 * Usage: bun tools/e2e.ts [url] [outdir]
 */
import { chromium, type Page } from "playwright-core";
import { mkdirSync } from "node:fs";

const url = process.argv[2] ?? "http://127.0.0.1:5318/";
const outDir = process.argv[3] ?? "out/e2e";
mkdirSync(outDir, { recursive: true });

const browser = await chromium.launch({
  headless: true,
  args: ["--autoplay-policy=no-user-gesture-required", "--use-fake-ui-for-media-stream"],
});
const errors: string[] = [];
const logs: string[] = [];
async function run(viewport: { width: number; height: number }, tag: string) {
  const page = await browser.newPage({ viewport, deviceScaleFactor: 1 });
  page.on("console", (m) => {
    const t = `${m.type()}: ${m.text()}`;
    if (m.type() === "error") errors.push(`[${tag}] ${t}`);
    else logs.push(`[${tag}] ${t}`);
  });
  page.on("pageerror", (e) => errors.push(`[${tag}] pageerror: ${e.message}`));
  await page.goto(url, { waitUntil: "networkidle" });
  await page.waitForTimeout(400);
  await page.screenshot({ path: `${outDir}/${tag}-0-idle.png` });
  // the app opens on Sing; this run starts in Play
  await page.getByRole("radio", { name: "Play" }).click();
  if (tag === "desktop") await grabTongueFirst(page, tag);
  else {
    // each tab's main button starts the audio: Play's breathes
    await page.getByRole("button", { name: /^Breathe$/ }).click();
    await page.waitForTimeout(1800);
  }
  await page.screenshot({ path: `${outDir}/${tag}-1-running.png` });
  // drag the tongue body handle via the canvas: find it by probing (center-ish of viewport)
  const box = await page.locator(".anatomy canvas").boundingBox();
  if (box) {
    // move over likely tongue area and drag up/forward
    const sx = box.x + box.width * 0.47, sy = box.y + box.height * 0.42;
    await page.mouse.move(sx, sy);
    await page.mouse.down();
    await page.mouse.move(sx + 40, sy - 30, { steps: 10 });
    await page.mouse.up();
  }
  // vowel map drag
  const vm = await page.locator(".vowel-map").boundingBox();
  if (vm) {
    await page.mouse.move(vm.x + vm.width * 0.2, vm.y + vm.height * 0.2);
    await page.mouse.down();
    await page.mouse.move(vm.x + vm.width * 0.8, vm.y + vm.height * 0.8, { steps: 20 });
    await page.mouse.up();
  }
  await page.keyboard.down("d");
  await page.waitForTimeout(900);
  await page.screenshot({ path: `${outDir}/${tag}-2-played.png` });
  await page.keyboard.up("d");
  // Z and X move the on-screen keys at once (they waited for the next note, and a click on them
  // meanwhile played the old octave)
  const firstKey = page.locator(".keys .key.white").first();
  const lowest = await firstKey.getAttribute("aria-label");
  await page.keyboard.press("x");
  const raised = await firstKey.getAttribute("aria-label");
  await page.keyboard.press("z");
  const back = await firstKey.getAttribute("aria-label");
  logs.push(`[${tag}] octave keys: ${lowest} → x → ${raised} → z → ${back}`);
  const num = (l: string | null) => Number(l?.replace("Note ", ""));
  if (num(raised) !== num(lowest) + 12 || back !== lowest) errors.push(`[${tag}] Z/X didn't move the on-screen keys: ${lowest} → ${raised} → ${back}`);
  // a key held with the mouse while X redraws it away is still released when the mouse lifts
  await firstKey.scrollIntoViewIfNeeded();
  const kb = (await firstKey.boundingBox())!;
  await page.mouse.move(kb.x + kb.width / 2, kb.y + kb.height - 10);
  await page.mouse.down();
  const pressed = await page.locator(".keys .key.down").count();
  await page.keyboard.press("x");
  await page.mouse.up();
  await page.keyboard.press("z");
  const stuck = await page.locator(".keys .key.down").count();
  logs.push(`[${tag}] octave keys: ${pressed} held with the mouse across X, ${stuck} still down after release`);
  if (pressed !== 1) errors.push(`[${tag}] the mouse didn't hold the lowest on-screen key (${pressed} down)`);
  if (stuck) errors.push(`[${tag}] a key held with the mouse while X was pressed stayed down after the mouse lifted`);
  const loadText = await page.locator(".load").textContent().catch(() => null);
  logs.push(`[${tag}] load: ${loadText}`);
  // speak mode
  await page.getByRole("radio", { name: "Speak" }).click();
  await page.waitForTimeout(200);
  await page.getByRole("button", { name: /^Speak$/ }).click();
  await page.waitForTimeout(1500);
  await page.screenshot({ path: `${outDir}/${tag}-3-speak.png`, fullPage: true });
  const trans = await page.locator(".transcription").textContent();
  logs.push(`[${tag}] transcription: ${trans}`);
  // speech breathes by itself: Play's held breath must not drone on once the phrase is over
  const speechRms: number[] = [];
  for (let i = 0; i < 12; i++) {
    await page.waitForTimeout(250);
    speechRms.push(
      await page.evaluate(() => {
        // @ts-expect-error debug handle
        return window.__chimeraEngine?.latest?.t?.outRms ?? -1;
      }),
    );
  }
  logs.push(`[${tag}] speech rms then silence ${speechRms.map((r) => r.toFixed(4)).join(",")}`);
  if (!(Math.max(...speechRms.slice(-3)) < 1e-3)) errors.push(`[${tag}] voice still sounding after the phrase: ${speechRms.slice(-3).join(",")}`);
  // the Examples menu puts a phrase in the box as text (not IPA), then reads Examples again, so
  // the same phrase can be picked a second time
  const examples = page.getByRole("combobox", { name: "Examples" });
  const sayBox = page.locator(".say textarea");
  const ipaChip = page.getByRole("button", { name: "Text is IPA" });
  const SHELLS = "She sells sea shells by the sea shore.";
  await ipaChip.click();
  await examples.selectOption({ label: SHELLS });
  const picked = { box: await sayBox.inputValue(), menu: await examples.inputValue(), ipa: await ipaChip.getAttribute("aria-pressed") };
  await sayBox.fill("Something else.");
  await examples.selectOption({ label: SHELLS });
  const again = await sayBox.inputValue();
  logs.push(`[${tag}] example picked: ${JSON.stringify(picked)}, picked again: "${again}"`);
  if (picked.box !== SHELLS || picked.menu !== "" || picked.ipa !== "false" || again !== SHELLS) errors.push(`[${tag}] the Examples menu didn't put the phrase in as text and reset (${JSON.stringify(picked)}, again "${again}")`);
  await page.locator("button.ipa", { hasText: "ʃ" }).first().click();
  await page.waitForTimeout(700);
  await page.screenshot({ path: `${outDir}/${tag}-4-pad.png` });
  // IPA mode tells the person which symbols the voice will not say (it used to drop them without a
  // word: upstream issue 52), as they type; plain text and clean IPA get no notice
  const notice = page.locator(".say-skipped");
  const box_ = page.getByRole("textbox", { name: /to speak/ });
  const noticeText = async () => ((await notice.textContent()) ?? "").trim();
  if ((await notice.getAttribute("aria-live")) !== "polite") errors.push(`[${tag}] the skipped-symbols notice is not a polite live region`);
  if (await noticeText()) errors.push(`[${tag}] the skipped-symbols notice shows with IPA mode off: ${await noticeText()}`);
  await ipaChip.click(); // IPA mode on (a toggle button)
  await box_.fill("ɕiɛ˥˩");
  const shown = await noticeText();
  logs.push(`[${tag}] skipped-symbols notice: ${shown}`);
  if (!/Not said/.test(shown) || !shown.includes("˥") || !shown.includes("˩")) errors.push(`[${tag}] typing IPA with tone letters showed no notice naming them: "${shown}"`);
  await page.screenshot({ path: `${outDir}/${tag}-4-speak-skipped.png`, fullPage: true });
  await box_.fill("ˈhɛloʊ");
  if (await noticeText()) errors.push(`[${tag}] the notice stayed for IPA the voice says: ${await noticeText()}`);
  await box_.fill("ɕiɛ˥˩");
  await box_.fill("");
  if (await noticeText()) errors.push(`[${tag}] clearing the box left the notice: ${await noticeText()}`);
  await box_.fill("ɕiɛ˥˩");
  await ipaChip.click(); // and off
  if (await noticeText()) errors.push(`[${tag}] turning IPA mode off left the notice: ${await noticeText()}`);
  // one language for Speak and Sing: TR chosen in Speak is lit in Sing, and still chosen when Speak
  // comes back (it was Speak's own state, reset whenever the tab remounted)
  const tr = (scope: string) => page.locator(scope).getByRole("button", { name: "Turkish" });
  await tr(".right-speak").click();
  await page.getByRole("radio", { name: "Sing" }).click();
  await page.waitForTimeout(200);
  const inSing = await tr(".s-lyrics").getAttribute("aria-pressed");
  await page.getByRole("radio", { name: "Speak" }).click();
  await page.waitForTimeout(200);
  const speakAgain = { chip: await tr(".right-speak").getAttribute("aria-pressed"), box: await sayBox.inputValue() };
  logs.push(`[${tag}] language: TR in Speak, in Sing ${inSing}; Speak again ${JSON.stringify(speakAgain)}`);
  if (inSing !== "true" || speakAgain.chip !== "true" || speakAgain.box !== "Merhaba, nasılsın?") errors.push(`[${tag}] Speak and Sing don't share the language (Sing ${inSing}, Speak again ${JSON.stringify(speakAgain)})`);
  await page.locator(".right-speak").getByRole("button", { name: "English" }).click();
  // sing mode: a key sings the marked syllable and moves the cursor; legato; a demo song
  await page.getByRole("radio", { name: "Sing" }).click();
  await page.waitForTimeout(600);
  const rms = () =>
    page.evaluate(() => {
      // @ts-expect-error debug handle
      return window.__chimeraEngine?.latest?.t?.outRms ?? -1;
    });
  const nextSyl = () => page.locator(".syl.next").textContent().catch(() => null);
  const sylBefore = await nextSyl();
  await page.keyboard.down("d");
  await page.waitForTimeout(450);
  const singing = await rms();
  await page.keyboard.down("g"); // legato into the next syllable
  await page.waitForTimeout(350);
  await page.keyboard.up("d");
  await page.waitForTimeout(250);
  await page.keyboard.up("g");
  await page.waitForTimeout(900);
  const after = await rms();
  const sylAfter = await nextSyl();
  logs.push(`[${tag}] sing: next syllable ${sylBefore} -> ${sylAfter}, rms while held ${singing.toFixed(4)}, then silence ${after.toFixed(4)}`);
  if (!(singing > 0.01)) errors.push(`[${tag}] a sung key made no sound (${singing})`);
  if (!(after < 1e-3)) errors.push(`[${tag}] voice still sounding after the sung notes: ${after}`);
  if (sylBefore === sylAfter) errors.push(`[${tag}] the syllable cursor did not move (${sylBefore})`);
  // a menu isn't an instrument: a letter typed in it picks an entry, it doesn't play a note
  await page.getByRole("combobox", { name: "Tune" }).focus();
  await page.keyboard.down("d");
  const menuNotes = await page.locator(".keys .key.down").count();
  await page.keyboard.up("d");
  if (menuNotes) errors.push(`[${tag}] a key typed in the tune menu played a note`);
  await page.getByRole("combobox", { name: "Tune" }).selectOption({ label: "Dreamy" });
  await page.getByRole("button", { name: /^Sing$/ }).click();
  await page.waitForTimeout(2200);
  const demoRms = await rms();
  const demoNow = await page.locator(".syl.now").textContent().catch(() => null);
  await page.screenshot({ path: `${outDir}/${tag}-5-sing.png`, fullPage: tag !== "desktop" });
  await page.getByRole("button", { name: /^Stop$/ }).click();
  await page.waitForTimeout(700);
  const demoStopped = await rms();
  logs.push(`[${tag}] sing a tune: rms ${demoRms.toFixed(4)} on "${demoNow}", after stop ${demoStopped.toFixed(4)}`);
  if (!(demoRms > 0.01) || !demoNow) errors.push(`[${tag}] the tune is not singing the lyrics (rms ${demoRms}, syllable ${demoNow})`);
  if (!(demoStopped < 1e-3)) errors.push(`[${tag}] the tune still sounding after stop: ${demoStopped}`);
  // bestiary: every creature, its first call, a key in held-voice mode, and the sustain toggle
  await page.getByRole("radio", { name: "Bestiary" }).click();
  await page.waitForTimeout(300);
  const names = await page.locator(".specimen-name").allTextContents();
  for (const [i, name] of names.entries()) {
    await page.locator(".specimen").nth(i).click();
    await page.waitForTimeout(250);
    await page.locator(".act-call").first().click();
    await page.waitForTimeout(1300);
    const slug = name.toLowerCase().replace(/[^a-z]+/g, "-");
    if (tag === "desktop" || i === 0) await page.screenshot({ path: `${outDir}/${tag}-5-${slug}.png`, fullPage: tag !== "desktop" });
    const state = await page.evaluate(() => {
      // @ts-expect-error debug handle
      const e = window.__chimeraEngine;
      const t = e?.latestCreature?.t;
      return t ? { id: t.id, rms: t.outRms, call: t.call, load: e.load } : null;
    });
    logs.push(`[${tag}] creature ${name}: ${JSON.stringify(state)}`);
  }
  // the sounding call's button reads Stop, and stops it (start one if the last has ended)
  const stop = page.getByRole("button", { name: /^Stop$/ });
  if (!(await stop.count())) {
    await page.locator(".act-call").first().click();
    await page.waitForTimeout(500);
  }
  await stop.click();
  await page.waitForTimeout(400);
  const callAfterStop = await page.evaluate(() => {
    // @ts-expect-error debug handle
    return window.__chimeraEngine?.latestCreature?.t?.call ?? null;
  });
  logs.push(`[${tag}] creature call after Stop: ${callAfterStop}`);
  if (callAfterStop !== null) errors.push(`[${tag}] the call kept sounding after Stop (${callAfterStop})`);
  await page.keyboard.down("g");
  await page.waitForTimeout(600);
  const held = await page.evaluate(() => {
    // @ts-expect-error debug handle
    const t = window.__chimeraEngine?.latestCreature?.t;
    return t ? { gate: t.gate, rms: t.outRms } : null;
  });
  await page.keyboard.up("g");
  logs.push(`[${tag}] held key: ${JSON.stringify(held)}`);
  // beatbox: preset, play, playhead + level from telemetry, a pad, stop
  await page.getByRole("radio", { name: "Beatbox" }).click();
  await page.waitForTimeout(300);
  await page.getByRole("combobox", { name: "Pattern" }).selectOption("Breakbeat");
  await page.getByRole("button", { name: /^Play beat$/ }).click();
  const beatSamples: { step: number; rms: number }[] = [];
  for (let i = 0; i < 8; i++) {
    await page.waitForTimeout(250);
    beatSamples.push(
      await page.evaluate(() => {
        // @ts-expect-error debug handle
        const t = window.__chimeraEngine?.latest;
        return { step: t?.beatStep ?? -2, rms: t?.t?.outRms ?? -1 };
      }),
    );
  }
  await page.screenshot({ path: `${outDir}/${tag}-6-beatbox.png`, fullPage: tag !== "desktop" });
  logs.push(`[${tag}] beatbox steps ${beatSamples.map((b) => b.step).join(",")} rms ${beatSamples.map((b) => b.rms.toFixed(3)).join(",")}`);
  await page.locator(".pad").first().dispatchEvent("pointerdown");
  await page.waitForTimeout(300);
  await page.getByRole("button", { name: /^Stop$/ }).click();
  await page.waitForTimeout(300);
  // Space presses the tab's main button (here: the beat on, then off), unless typing
  await page.locator("body").click({ position: { x: 2, y: 2 } });
  await page.keyboard.press("Space");
  await page.waitForTimeout(700);
  const spaceOn = await page.getByRole("button", { name: /^Stop$/ }).count();
  await page.keyboard.press("Space");
  await page.waitForTimeout(300);
  const spaceOff = await page.getByRole("button", { name: /^Play beat$/ }).count();
  logs.push(`[${tag}] beatbox space: playing ${spaceOn}, stopped ${spaceOff}`);
  if (spaceOn !== 1 || spaceOff !== 1) errors.push(`[${tag}] Space didn't start and stop the beat (${spaceOn}, ${spaceOff})`);
  // back to the human voice
  await page.getByRole("radio", { name: "Play" }).click();
  await page.waitForTimeout(600);
  const human = await page.evaluate(() => {
    // @ts-expect-error debug handle
    const e = window.__chimeraEngine;
    return { human: !!e?.latest, creature: !!e?.latestCreature };
  });
  logs.push(`[${tag}] back to play: ${JSON.stringify(human)}`);
  // a device that keeps up keeps the full rate through the whole run (the fallback is for slow ones)
  const rate = await page.evaluate(() => {
    // @ts-expect-error debug handle
    return window.__chimeraEngine?.ctx?.sampleRate ?? -1;
  });
  logs.push(`[${tag}] rate after the run: ${rate} Hz`);
  if (rate !== 48000) errors.push(`[${tag}] this machine fell back to ${rate} Hz (it keeps up at 48 kHz)`);
  await page.close();
}

// a start that failed (the worklet didn't load) can be retried from the same button: the retry
// builds the audio again instead of resuming the empty context and reporting success
{
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  page.on("pageerror", (e) => errors.push(`[retry] pageerror: ${e.message}`));
  // the worklet's fetch isn't the page's (route() never sees it): fail the first addModule instead
  await page.addInitScript(() => {
    const add = AudioWorklet.prototype.addModule;
    let calls = 0;
    AudioWorklet.prototype.addModule = function (this: AudioWorklet, ...a: Parameters<typeof add>) {
      (window as unknown as { __worklets: number }).__worklets = ++calls;
      return calls === 1 ? Promise.reject(new Error("worklet blocked by the test")) : add.apply(this, a);
    };
  });
  await page.goto(url, { waitUntil: "networkidle" });
  await page.getByRole("radio", { name: "Play" }).click();
  await page.getByRole("button", { name: /^Breathe$/ }).click();
  const failed = await page.getByText(/Audio could not start/).waitFor({ timeout: 10000 }).then(() => true, () => false);
  await page.getByRole("button", { name: /^Breathe$/ }).click();
  await page.waitForTimeout(1500);
  const running = await page.evaluate(() => {
    // @ts-expect-error debug handle
    return !!window.__chimeraEngine?.latest;
  });
  const worklets = await page.evaluate(() => (window as unknown as { __worklets?: number }).__worklets ?? 0);
  logs.push(`[retry] first start failed: ${failed}, worklet requests ${worklets}, running after retry: ${running}`);
  if (!failed) errors.push("[retry] blocking the worklet didn't make the first start fail");
  if (!running) errors.push(`[retry] the retry after a failed start didn't start the audio (${worklets} worklet requests)`);
  await page.close();
}

// the very first key in Sing mode, before audio has started, still sings once audio is up
{
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  page.on("pageerror", (e) => errors.push(`[first-key] pageerror: ${e.message}`));
  await page.goto(url, { waitUntil: "networkidle" });
  await page.getByRole("radio", { name: "Sing" }).click();
  await page.keyboard.down("d");
  await page.waitForTimeout(1200);
  const level = await page.evaluate(() => {
    // @ts-expect-error debug handle
    return window.__chimeraEngine?.latest?.t?.outRms ?? -1;
  });
  await page.keyboard.up("d");
  logs.push(`[first-key] sing before start: rms ${level.toFixed(4)}`);
  if (!(level > 0.01)) errors.push(`[first-key] the first sung key, pressed before audio started, made no sound (${level})`);
  await page.close();
}

// an arrow key moves every encoder, on every page of Sing and Beatbox: on one whose value is
// rounded (Key, Tempo), an arrow once moved a thousandth of the dial and the rounding put it back
{
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  page.on("pageerror", (e) => errors.push(`[arrow-keys] pageerror: ${e.message}`));
  await page.goto(url, { waitUntil: "networkidle" });
  const stuck: string[] = [];
  let turned = 0;
  for (const mode of ["Sing", "Beatbox"]) {
    await page.getByRole("radio", { name: mode }).click();
    const tabs = page.locator(".right .enc-tab");
    for (let t = 0; t < Math.max(1, await tabs.count()); t++) {
      if (await tabs.count()) await tabs.nth(t).click();
      const inputs = page.locator(".right .enc-input");
      for (let i = 0; i < (await inputs.count()); i++) {
        const input = inputs.nth(i);
        const was = await input.inputValue();
        await input.focus();
        await page.keyboard.press("ArrowRight");
        // at the top of its range it can only go down
        if ((await input.inputValue()) === was) await page.keyboard.press("ArrowLeft");
        const now = await input.inputValue();
        turned++;
        if (now === was) stuck.push(`${mode} ${await input.getAttribute("aria-valuetext")}`);
      }
    }
  }
  logs.push(`[arrow-keys] ${turned} encoders turned by an arrow key, ${stuck.length} stuck`);
  if (!turned) errors.push("[arrow-keys] found no encoders to turn");
  if (stuck.length) errors.push(`[arrow-keys] an arrow key didn't move: ${stuck.join(", ")}`);
  await page.close();
}

// the voice runs at 48 kHz on a 192 kHz device: at the device's rate it took ~3x the CPU and
// starved the worklet (issue #33). The patched constructor plays a device whose default is 192 kHz.
{
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  page.on("pageerror", (e) => errors.push(`[device-rate] pageerror: ${e.message}`));
  await page.addInitScript(() => {
    const Device = window.AudioContext;
    window.AudioContext = class extends Device {
      constructor(o: AudioContextOptions = {}) {
        super({ sampleRate: 192000, ...o });
      }
    };
  });
  await page.goto(url, { waitUntil: "networkidle" });
  await page.getByRole("radio", { name: "Play" }).click();
  await page.getByRole("button", { name: /^Breathe$/ }).click();
  await page.waitForTimeout(1500);
  const rate = await page.evaluate(() => {
    // @ts-expect-error debug handle
    return window.__chimeraEngine?.ctx?.sampleRate ?? -1;
  });
  logs.push(`[device-rate] a 192 kHz device runs the voice at ${rate} Hz`);
  if (rate !== 48000) errors.push(`[device-rate] the voice ran at ${rate} Hz on a 192 kHz device (want 48000)`);
  await page.close();
}

// a device too slow for 48 kHz falls back to 32 kHz and keeps sounding; one too slow for that says
// so (issues #86, #113). The patched node adds to the load the worklet reports: at 48 kHz only
// (it keeps up at 32), or at every rate. "-gesture": the browser holds the 32 kHz context, built
// without a gesture, until there is one (its resume() doesn't settle till then): the rebuild must
// still finish, and the next tap must bring the sound back.
for (const slowAt of ["48000", "any", "48000-gesture"]) {
  const tag = `slow-${slowAt}`;
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  page.on("pageerror", (e) => errors.push(`[${tag}] pageerror: ${e.message}`));
  await page.addInitScript((slowAt) => {
    if (slowAt.endsWith("-gesture")) {
      let gestured = false;
      window.addEventListener("pointerdown", () => (gestured = true), { capture: true });
      const Device = window.AudioContext, resume = Device.prototype.resume;
      window.AudioContext = class extends Device {
        constructor(o?: AudioContextOptions) {
          super(o);
          // (the autoplay switch starts a context after its constructor: hold it at each start)
          // (only a tap after it counts: the one that started the audio came before)
          if (this.sampleRate === 32000) gestured = false;
          if (this.sampleRate === 32000) this.addEventListener("statechange", () => !gestured && this.state === "running" && void this.suspend());
        }
        override resume(): Promise<void> {
          return this.sampleRate !== 32000 || gestured ? resume.call(this) : new Promise<void>(() => {});
        }
      };
    }
    const Node = window.AudioWorkletNode;
    const onmessage = Object.getOwnPropertyDescriptor(MessagePort.prototype, "onmessage")!;
    window.AudioWorkletNode = class extends Node {
      constructor(ctx: BaseAudioContext, name: string, o?: AudioWorkletNodeOptions) {
        super(ctx, name, o);
        const port = this.port, slow = slowAt === "any" || String(ctx.sampleRate) === slowAt.split("-")[0];
        let h: ((e: MessageEvent) => void) | null = null;
        Object.defineProperty(port, "onmessage", {
          get: () => h,
          set(f: ((e: MessageEvent) => void) | null) {
            h = f;
            onmessage.set!.call(port, f && ((e: MessageEvent) => f(slow && "load" in e.data ? ({ data: { ...e.data, load: e.data.load + 1.5 } } as MessageEvent) : e)));
          },
        });
      }
    };
  }, slowAt);
  await page.goto(url, { waitUntil: "networkidle" });
  await page.getByRole("radio", { name: "Play" }).click();
  await page.getByRole("button", { name: /^Breathe$/ }).click();
  // the start's warm-up and the averaging, at each rate
  await page.waitForTimeout(10000);
  const state = await page.evaluate(() => {
    // @ts-expect-error debug handle
    const e = window.__chimeraEngine;
    return { rate: e?.ctx?.sampleRate ?? -1, rms: e?.latest?.t?.outRms ?? -1 };
  });
  if (slowAt.endsWith("-gesture")) {
    const held = await page.evaluate(() => {
      // @ts-expect-error debug handle
      return window.__chimeraEngine?.ctx?.state;
    });
    await page.locator(".wordmark").click();
    await page.waitForTimeout(1500);
    state.rms = await page.evaluate(() => {
      // @ts-expect-error debug handle
      return window.__chimeraEngine?.latest?.t?.outRms ?? -1;
    });
    logs.push(`[${tag}] the 32 kHz context was ${held} until a tap`);
    if (held !== "suspended") errors.push(`[${tag}] the held context was ${held} before the tap (the test's patch didn't hold it)`);
  }
  const warned = await page.getByText(/This device can't keep up/).isVisible();
  const badge = await page.getByText(/· 32 kHz/).isVisible();
  logs.push(`[${tag}] slow device: runs at ${state.rate} Hz, rms ${state.rms.toFixed(4)}, 32 kHz shown ${badge}, warning ${warned}`);
  if (state.rate !== 32000) errors.push(`[${tag}] a device too slow for 48 kHz ran at ${state.rate} Hz (want 32000)`);
  if (!badge) errors.push(`[${tag}] the top bar doesn't say the voice runs at 32 kHz`);
  if (!(state.rms > 0.01)) errors.push(`[${tag}] the breath stopped sounding across the fallback (rms ${state.rms})`);
  if (warned !== (slowAt === "any")) errors.push(`[${tag}] the can't-keep-up warning ${warned ? "showed" : "didn't show"}`);
  await page.close();
}

// Sing's Record video: sings the lyrics to the tune again into a clip to save; the face screen leaves out
// its readouts while it records
{
  const page = await browser.newPage({ viewport: { width: 390, height: 844 }, acceptDownloads: true });
  page.on("console", (m) => m.type() === "error" && errors.push(`[video] ${m.text()}`));
  page.on("pageerror", (e) => errors.push(`[video] pageerror: ${e.message}`));
  await page.goto(url, { waitUntil: "networkidle" });
  await page.locator(".lyrics textarea").fill("one two three");
  await page.getByRole("combobox", { name: "Tune" }).selectOption({ label: "Playful" });
  await page.getByRole("button", { name: /^Record video$/ }).click();
  await page.getByRole("button", { name: /^Cancel$/ }).waitFor({ timeout: 10000 });
  await page.waitForTimeout(800);
  await page.screenshot({ path: `${outDir}/video-recording.png` });
  const save = page.getByRole("link", { name: /^Save$/ });
  const saved = await save.waitFor({ timeout: 20000 }).then(() => true, () => false);
  if (!saved) errors.push("[video] Record video never offered a clip to save");
  else {
    const [dl] = await Promise.all([page.waitForEvent("download"), save.click()]);
    const size = (await (await dl.createReadStream()).toArray()).reduce((n, b: Buffer) => n + b.length, 0);
    logs.push(`[video] clip ${dl.suggestedFilename()}: ${size} bytes`);
    if (size < 10000) errors.push(`[video] the clip is nearly empty (${size} bytes)`);
  }
  // the words and the tune come back on reload
  await page.reload({ waitUntil: "networkidle" });
  const words = await page.locator(".lyrics textarea").inputValue();
  const tune = await page.getByRole("combobox", { name: "Tune" }).inputValue();
  if (words !== "one two three" || tune !== "playful") errors.push(`[video] the song didn't come back on reload: "${words}" to ${tune}`);
  await page.close();
}

// two taps on Record video while the audio is still starting make one clip: the first start gives way (it
// used to leave a second recorder running that Cancel couldn't reach)
{
  const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
  page.on("pageerror", (e) => errors.push(`[video-twice] pageerror: ${e.message}`));
  await page.addInitScript(() => {
    const Rec = window.MediaRecorder;
    const w = window as unknown as { __recorders: number };
    w.__recorders = 0;
    window.MediaRecorder = class extends Rec {
      constructor(...a: ConstructorParameters<typeof Rec>) {
        super(...a);
        w.__recorders++;
      }
    };
  });
  await page.goto(url, { waitUntil: "networkidle" });
  await page.locator(".lyrics textarea").fill("one two");
  // both taps land before the first has started the audio (a second Playwright click would wait,
  // and then find Cancel)
  await page.evaluate(() => {
    const b = [...document.querySelectorAll<HTMLButtonElement>(".action-bar button")].find((x) => x.textContent === "Record video")!;
    b.click();
    b.click();
  });
  await page.getByRole("link", { name: /^Save$/ }).waitFor({ timeout: 20000 }).catch(() => errors.push("[video-twice] no clip after two taps"));
  const recorders = await page.evaluate(() => (window as unknown as { __recorders: number }).__recorders);
  logs.push(`[video-twice] two taps on Record video: ${recorders} recorder(s)`);
  if (recorders !== 1) errors.push(`[video-twice] two taps on Record video made ${recorders} recorders (want 1)`);
  await page.close();
}

// Record video pressed while a song is singing: the clip starts once that song has died away (it
// used to open on the stopped song's tail)
{
  const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
  page.on("pageerror", (e) => errors.push(`[video-tail] pageerror: ${e.message}`));
  await page.addInitScript(() => {
    const w = window as unknown as { __startLevel: number | null; __chimeraEngine?: { level(): number } };
    w.__startLevel = null;
    const start = MediaRecorder.prototype.start;
    MediaRecorder.prototype.start = function (this: MediaRecorder, ...a: [number?]) {
      w.__startLevel = w.__chimeraEngine?.level() ?? null;
      return start.apply(this, a);
    };
  });
  await page.goto(url, { waitUntil: "networkidle" });
  await page.locator(".lyrics textarea").fill("one two three four five six seven eight");
  await page.getByRole("button", { name: /^Sing$/ }).click();
  await page.waitForTimeout(2500);
  const singing = await page.evaluate(() => (window as unknown as { __chimeraEngine: { level(): number } }).__chimeraEngine.level());
  await page.getByRole("button", { name: /^Record video$/ }).click();
  await page.getByRole("button", { name: /^Cancel$/ }).waitFor({ timeout: 10000 });
  const atStart = await page.evaluate(() => (window as unknown as { __startLevel: number | null }).__startLevel);
  logs.push(`[video-tail] output rms while singing ${singing.toFixed(4)}, as the clip starts ${atStart?.toFixed(4)}`);
  if (singing < 0.005) errors.push(`[video-tail] the song wasn't sounding when Record video was pressed (rms ${singing})`);
  if (atStart === null || atStart >= 0.003) errors.push(`[video-tail] the clip started on the last song's tail (rms ${atStart})`);
  await page.close();
}

// a recorder that fails partway (an encoder or resource failure): it lets go of the capture and the
// face's measurements come back then, no clip is offered, and nothing throws unhandled (the clip used
// to wait for the song's end, then reject, leaving its capture track running)
{
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  page.on("pageerror", (e) => errors.push(`[video-breaks] pageerror: ${e.message}`));
  await page.addInitScript(() => {
    const w = window as unknown as { __unhandled: number; __tracks: MediaStreamTrack[]; __texts: string[] };
    w.__unhandled = 0;
    w.__tracks = [];
    w.__texts = [];
    window.addEventListener("unhandledrejection", () => w.__unhandled++);
    const start = MediaRecorder.prototype.start;
    MediaRecorder.prototype.start = function (this: MediaRecorder, ...a: [number?]) {
      w.__tracks.push(...this.stream.getVideoTracks());
      start.apply(this, a);
      setTimeout(() => {
        this.dispatchEvent(new Event("error"));
        if (this.state !== "inactive") this.stop();
      }, 800);
    };
    const fill = CanvasRenderingContext2D.prototype.fillText;
    CanvasRenderingContext2D.prototype.fillText = function (this: CanvasRenderingContext2D, t: string, ...a: [number, number]) {
      if (w.__texts.length < 5000) w.__texts.push(t);
      return fill.call(this, t, ...a);
    };
  });
  await page.goto(url, { waitUntil: "networkidle" });
  await page.locator(".lyrics textarea").fill("one two three four five six seven eight nine ten");
  await page.getByRole("button", { name: /^Record video$/ }).click();
  await page.getByRole("button", { name: /^Cancel$/ }).waitFor({ timeout: 10000 });
  await page.waitForTimeout(1500); // the recorder has failed; the song is still singing
  await page.evaluate(() => ((window as unknown as { __texts: string[] }).__texts = []));
  await page.waitForTimeout(500);
  const mid = await page.evaluate(() => {
    const w = window as unknown as { __tracks: MediaStreamTrack[]; __texts: string[] };
    return { live: w.__tracks.filter((t) => t.readyState === "live").length, readouts: w.__texts.some((t) => /^RING O/.test(t)) };
  });
  await page.getByRole("button", { name: /^Record video$/ }).waitFor({ timeout: 30000 });
  await page.waitForTimeout(500);
  const save = await page.getByRole("link", { name: /^Save$/ }).count();
  const unhandled = await page.evaluate(() => (window as unknown as { __unhandled: number }).__unhandled);
  logs.push(`[video-breaks] recorder failed mid-song: capture tracks live ${mid.live}, readouts back ${mid.readouts}; after the song Save ${save}, unhandled rejections ${unhandled}`);
  if (mid.live) errors.push(`[video-breaks] the capture kept running after the recorder failed (${mid.live} live track(s))`);
  if (!mid.readouts) errors.push("[video-breaks] the face kept hiding its measurements after the recorder failed");
  if (save) errors.push("[video-breaks] a failed recording was offered to save");
  if (unhandled) errors.push(`[video-breaks] ${unhandled} unhandled rejection(s)`);
  await page.close();
}

// switching 2D/3D while recording: the clip follows the face that's showing (it kept copying the
// replaced canvas, so the video froze at the switch while the sound went on)
{
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  page.on("pageerror", (e) => errors.push(`[video-3d] pageerror: ${e.message}`));
  await page.addInitScript(() => {
    const w = window as unknown as { __draws: boolean[] };
    w.__draws = [];
    const draw = CanvasRenderingContext2D.prototype.drawImage as (...a: unknown[]) => void;
    CanvasRenderingContext2D.prototype.drawImage = function (this: CanvasRenderingContext2D, ...a: unknown[]) {
      // the clip's frame is 720x1280: note whether what it copies is still on the page
      if (this.canvas.width === 720 && this.canvas.height === 1280 && a[0] instanceof HTMLCanvasElement) w.__draws.push(a[0].isConnected);
      return draw.apply(this, a);
    } as typeof CanvasRenderingContext2D.prototype.drawImage;
  });
  await page.goto(url, { waitUntil: "networkidle" });
  await page.locator(".lyrics textarea").fill("one two three four five six seven eight nine ten");
  await page.getByRole("button", { name: /^Record video$/ }).click();
  await page.getByRole("button", { name: /^Cancel$/ }).waitFor({ timeout: 10000 });
  await page.waitForTimeout(500);
  await page.getByRole("checkbox", { name: "3D" }).check();
  await page.waitForTimeout(1000);
  await page.evaluate(() => ((window as unknown as { __draws: boolean[] }).__draws = []));
  await page.waitForTimeout(500);
  const draws = await page.evaluate(() => (window as unknown as { __draws: boolean[] }).__draws);
  const detached = draws.filter((c) => !c).length;
  logs.push(`[video-3d] after switching to 3D mid-clip: ${draws.length} frames, ${detached} from a canvas no longer on the page`);
  if (!draws.length) errors.push("[video-3d] the clip drew no frames after switching to 3D");
  if (detached) errors.push(`[video-3d] the clip kept copying the replaced 2D canvas (${detached} of ${draws.length} frames)`);
  await page.getByRole("button", { name: /^Cancel$/ }).click();
  await page.close();
}

// a recorder that fails to start (a type the browser claims but can't record): no clip, the Sing
// button comes back, and the face shows its readouts again (the clean view used to stay on)
{
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  page.on("pageerror", (e) => errors.push(`[video-fails] pageerror: ${e.message}`));
  await page.addInitScript(() => {
    MediaRecorder.prototype.start = () => {
      throw new Error("start refused by the test");
    };
    const w = window as unknown as { __texts: string[] };
    w.__texts = [];
    const fill = CanvasRenderingContext2D.prototype.fillText;
    CanvasRenderingContext2D.prototype.fillText = function (this: CanvasRenderingContext2D, t: string, ...a: [number, number]) {
      if (w.__texts.length < 5000) w.__texts.push(t);
      return fill.call(this, t, ...a);
    };
  });
  await page.goto(url, { waitUntil: "networkidle" });
  await page.getByRole("button", { name: /^Record video$/ }).click();
  await page.waitForTimeout(2500);
  await page.evaluate(() => ((window as unknown as { __texts: string[] }).__texts = []));
  await page.waitForTimeout(500);
  const readouts = await page.evaluate(() => (window as unknown as { __texts: string[] }).__texts.some((t) => /^RING O/.test(t)));
  const sing = await page.getByRole("button", { name: /^(Sing|Stop)$/ }).count();
  logs.push(`[video-fails] after a recorder that won't start: readouts ${readouts}, Sing button ${sing}`);
  if (!readouts) errors.push("[video-fails] the face kept hiding its readouts after a recording failed to start");
  if (!sing) errors.push("[video-fails] the Sing button didn't come back after a recording failed to start");
  await page.close();
}

/**
 * Play before any audio (upstream issue 89): grabbing the tongue starts the audio and breathes, and
 * the engine keeps the tongue in the mouth. Since 704ed2c took away the start overlay over the face,
 * a drag before the audio existed moved only the drawing, up through the skull, in silence.
 */
async function grabTongueFirst(page: Page, tag: string) {
  const box = (await page.locator(".anatomy canvas").boundingBox())!;
  const anatomy = page.locator(".anatomy");
  // the tongue handle, found by hovering a grid finer than its 22 px hit radius
  const findTongue = async () => {
    for (let y = box.y + 6; y < box.y + box.height; y += 16)
      for (let x = box.x + 6; x < box.x + box.width; x += 16) {
        await page.mouse.move(x, y);
        if ((await anatomy.getAttribute("data-hover")) === "tongue") return { x, y };
      }
    return null;
  };
  const from = await findTongue();
  if (!from) return void errors.push(`[${tag}] found no tongue handle on the face`);
  const to = { x: from.x, y: box.y + 4 }; // above the skull
  await page.mouse.move(from.x, from.y);
  await page.mouse.down();
  // heard with the tongue where it was: pressed to the palate it closes the mouth, and is silent
  await page
    .waitForFunction(() => {
      // @ts-expect-error debug handle
      const e = window.__chimeraEngine;
      return e?.ctx?.state === "running" && (e?.latest?.t?.outRms ?? 0) > 0.01;
    }, null, { timeout: 10000 })
    .catch(() => {});
  const sound = await page.evaluate(() => {
    // @ts-expect-error debug handle
    const e = window.__chimeraEngine;
    return { ctx: e?.ctx?.state ?? null, rms: e?.latest?.t?.outRms ?? -1 };
  });
  await page.mouse.move(to.x, to.y, { steps: 15 });
  await page.waitForTimeout(600);
  await page.mouse.up();
  const at = await findTongue();
  logs.push(`[${tag}] tongue grabbed first: audio ${sound.ctx}, rms ${sound.rms.toFixed(4)}; handle ${JSON.stringify(from)} dragged to ${JSON.stringify(to)}, rests at ${JSON.stringify(at)}`);
  if (sound.ctx !== "running" || !(sound.rms > 0.01)) errors.push(`[${tag}] grabbing the tongue before any audio made no sound (audio ${sound.ctx}, rms ${sound.rms})`);
  // the palate is a couple of centimetres above the tongue: a third of the way to the top is out of the mouth
  if (!at || at.y < to.y + (from.y - to.y) * 0.67) errors.push(`[${tag}] the tongue left the mouth: dragged from ${JSON.stringify(from)} to ${JSON.stringify(to)}, rests at ${JSON.stringify(at)}`);
}

await run({ width: 1440, height: 980 }, "desktop");
await run({ width: 390, height: 844 }, "mobile");
await browser.close();
console.log("console errors:", errors.length ? "\n" + errors.join("\n") : "none");
console.log(logs.filter((l) => /load|rate|slow|warn|transcription|example|creature|held|back to|beatbox|silence|sing|octave|retry|tongue|song|video|arrow/i.test(l)).slice(0, 52).join("\n"));
// a run that collected errors fails, so CI and agents can't read it as a pass
if (errors.length) process.exit(1);
