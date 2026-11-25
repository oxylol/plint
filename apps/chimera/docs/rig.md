# The body rig

A body for the 3D head to wear, for people (and agents) making videos with the voice: the app's
own head (`src/ui/render/anatomy3d.ts`) drawn live on a posable low-poly body, its face moving as
the voice does. It's a small library, not an animation tool: there's no timeline and no renderer
of its own. You pose it, skin it and draw it in your own three.js scene, frame by frame.

Try it: `bun run dev`, then `/rig.html` (pick a pose, open the mouth, pick how the head is drawn,
or play a motion file from another tool).

## Files

| | |
|---|---|
| `src/rig/rig.ts` | the rig: loading, posing by joint angles, skinning |
| `src/rig/pose.ts` | the `Pose` type (anatomical angles), ranges of motion, presets |
| `src/rig/head.ts` | `wearHead`: the head's skin grown from the neck, the head's lines on the head bone, the face moving |
| `src/rig/motion.ts` | `Motion`: a clip from another tool (Mixamo's FBX, BVH, glTF) carried over onto the rig |
| `src/rig/view.ts` | the wireframe in CHIMERA's blue (and heat and weights views for finding faults) |
| `src/rig/body.rig.bin` | the mesh, skeleton and weights (generated: `bun tools/rig/gen-body.ts`, never edited) |
| `src/rig/demo.ts`, `rig.html` | the dev page (not in the app's build) |

## Use

```ts
import * as THREE from "three";
import { initChimera } from "./src/wasm/chimera.ts";
import rigUrl from "./src/rig/body.rig.bin?url";

await initChimera();
// the head reads the engine's tables as it loads, so it (and what imports it) comes after the engine
const [{ Rig }, { pose }, { wearHead }, { RigView }, { VoiceModel }, { PRESETS: BODIES }] = await Promise.all([
  import("./src/rig/rig.ts"), import("./src/rig/pose.ts"), import("./src/rig/head.ts"),
  import("./src/rig/view.ts"), import("./src/ui/analysis.ts"), import("./src/ui/body.ts"),
]);
const rig = await Rig.load(rigUrl);
const model = new VoiceModel(BODIES.male, 48000);    // yours: keep it for the rig's life, dispose it after
const head = wearHead(rig, model);                    // before the view and any buffers: it adds the head's skin to the mesh
const view = new RigView(rig);
const skinned = new Float32Array(3 * rig.n);          // rig.n counts the head's skin now
scene.add(view.group, head.mount);                    // under the same parent
renderer.outputColorSpace = THREE.LinearSRGBColorSpace; // the head's colours are the app's display values

// every frame
rig.pose(pose({ arms: { R: { raise: 90, elbow: 40 } }, neck: { twist: 30 } }));
rig.skin(skinned);                                    // centres of rotation; rig.skin(out, "lbs") for plain blending
head.update(camera, artic, tele, dt);                 // the voice now: its articulation, its telemetry (or null)
head.face(skinned, view.posed);                       // the jaw, cheeks and lips on top
view.update();
head.resize(width, height);                           // on resize: the head's line widths are in px

// done with it (or changing voice body: a new rig, head and view)
scene.remove(view.group, head.mount);
view.dispose();
head.dispose();
model.dispose();
```

Or move it as a clip from another tool does, instead of posing it:

```ts
import { Motion } from "./src/rig/motion.ts";
const motion = await Motion.load(rig, await file.arrayBuffer(), file.name); // .fbx, .bvh, .glb or .gltf
// every frame, instead of rig.pose(…)
motion.pose(seconds);                                 // loops; motion.duration, motion.clip
```

`artic` and `tele` come from wherever the voice runs: the live worklet's telemetry messages
(`VoiceModel.articFromArray(msg.t.artic)`), or a recording replayed with `TeleTrack`
(`chimera render notes … --tele voice.tele`, `src/show/tele.ts`). Without a voice, pass an
articulation of your own (`REST_ARTIC` from `src/ui/render/anatomy3d.ts` is the mouth closed)
and `null`. A different body (a child's voice, say) is a new rig and a new head: the head's skin
is fitted to the model once.

## How it works

- **Body:** MakeHuman's CC0 low-poly body (proxy741, hand-laid loops), on MPFB2's Rigify human
  skeleton and weights: 33 bones, each limb bone split in two for twist, and Rigify's helpers
  (the shoulder's cap and the hip turn half way with the limb; the elbow and knee swell as they
  bend).
- **Poses** are angles as an anatomist names them, from anatomical zero (standing, arms
  hanging, thumbs forward). The shoulder is the ISB "globe": `raise`, `plane` (0 out to the side,
  90 forward) and `rotate`. Left and right take the same angles; the rig mirrors them. The rig
  adds what bodies do on their own: the collarbone rises about 10° as the arm goes overhead, the
  arm turns out as it rises to the side, the pelvis rolls back past about 55° of hip flexion,
  twist is shared along each limb, and a head turn is spread up the whole neck.
- **Skinning:** optimised centres of rotation (Le & Hodgins 2016), on the CPU: no candy-wrapper
  twist, no bulge. Plain linear blending is there too (what other tools do).
- **Head:** a quad grid grown from the neck's rim over the head's own surface (`skinAt`), closed
  by a patch over the crown and run on over the mouth, square for square. It's cut open along
  the lips' line, as wide as the mouth can spread; each frame the cut is laid on the lips the
  head draws (`lipEdges`), corner to corner (a rounded mouth gathers the skin in), and the jaw
  and cheeks carry the rest of the face (`skinNow`). The head's
  own lines (eyes, nose, lips, the vocal tract inside) are drawn by Anatomy3D on the head bone,
  with `drawing.mounted` (the body turns the head) and the lip patch's own skin lines left out.
- **Looks** (`HEAD_LOOKS`): `vocal` (the default: the grid and the face in front, the insides at
  half, for singing and speaking), `outside` (the insides faint, for acting), `skinned` (the
  app's insides at full under a half-bright grid) and `classic` (the app's head as it draws
  itself).

## Motion from other tools

`Motion.load` reads the file with three.js's loader for it (`readMotion`: FBX, BVH, a GLB or a
.gltf with nothing in separate files), plays a clip of it on its own skeleton (the longest, or
`{ clip: name or index }`) and carries each frame over onto the rig. For a scene you've loaded
yourself (a .gltf with separate files, a VRM), `new Motion(rig, root, clip)`.

- **Matching:** the limbs and head by name (Mixamo, CMU and DAZ BVHs, VRM and VRoid, Unreal,
  Rigify, 3ds Max Biped: `rigName`), each the bone the clip itself moves by that name;
  `motion.bones` says what was found. The trunk is found by where it sits between them, and a
  spine or neck of any number of bones is spread along the rig's by length (the chest the arms
  hang from, and the head, always match). For names it misses, or two bones that could be one
  of the rig's and the clip moves both: `{ map: { "DEF-spine.006": "head" } }` (it comes first).
- **Carrying over:** the source as loaded is its reference: each limb's turn then is swung until
  it points as the rig's does at anatomical zero (the hand also turned so its palm faces as the
  rig's, where it has fingers). Each frame, a source bone's turn since its reference is the rig
  bone's, in a world matched by the source's up axis and its hips' facing; so the source's bone
  axes, units, facing and up axis don't matter. A T or an A pose reference both work, palms down
  or in; a limb's roll can't be seen in a straight limb, so a reference with the arms turned
  otherwise (palms forward) comes out turned by as much. A mirrored or unevenly scaled source is
  refused, and so is one that doesn't stand as loaded (its height can't be told). Limb
  twist is shared along the rig's twist bones, as its posing does (without it a turned wrist
  pinches), except near half a turn, where it can't be told which way round it went. Only rotations carry over: the source's bones' lengths and any movement of a joint
  other than the hips are the rig's own.
- **Where it stands:** the hips move as the source's do, measured from the source's origin and
  scaled by leg length (hip joints over ankles). The floor is an estimate: where the source's
  lower ankle usually is over the clip (a low percentile), put where the rig's stands. A clip
  that never touches down stands on its lowest. Its feet can slide a little where its
  proportions differ from the rig's (there's no foot locking).
- A Motion plays its clip on the source's scene (`motion.root`) and reads it relative to that
  scene's parent: put it beside the rig by its parent (`motion.C` and `motion.scale` turn and
  size it as the rig), not by moving `root` itself. For another clip, load the file again.

Mixamo: download "FBX Binary", with or without skin. Mixamo's files can't be redistributed, so
none are in the repo; the tests write the rig's own poses as a BVH and read them back.

## Frames

Body cm: x forward, y up, z out of the body's right side. The skeleton's bind pose is the mesh's
(an A pose: arms out and down about 40°). Each bone has a parent (`rig.bones[i].parent`), its
rest joint `rig.H[i]`, and after `pose` its turn from rest `rig.S[i]` and joint `rig.P[i]`
(x' = S (x − H) + P). The head is drawn in its own cm, `head.s` times larger than the body.

## Checks

- `bun test tests/rig.test.ts` (part of `bun run test`): the mesh's open edges, skinning at
  rest, anatomical zero, the lips' cut lying on the drawn lips (shut; open narrow and wide, the
  head turned), weights summing to 1, the face's motion staying off the crown and the rim, the
  head following its parent's transform; and motion: bone names across tools, every preset
  written as a BVH (from a standing and a T-pose reference, turned, in other units) and read
  back to within 0.1 cm and the hands to 0.5°, the hips' height, a skeleton without toes, a
  one-frame clip, seeking, and the source moved by its parent.
- `/rig.html`: look at it. A metric isn't the picture.

## Provenance

`body.rig.bin` is CC0: MakeHuman's base mesh, targets and proxies
([makehuman](https://github.com/makehumancommunity/makehuman), `LICENSE.ASSETS.md`; the proxy
from the CC0 system assets pack) and MPFB2's Rigify human rig and weights
([mpfb2](https://github.com/makehumancommunity/mpfb2), `LICENSE.md`, assets). The generator pins
each source by commit, or by hash where the source has no version.

## Not yet

Fingers (folded into the hand), dragging hands and feet (IK), keeping imported feet planted,
exporting the body (GLB). Deep bends (elbow or knee near 150°, a deep
squat) still show some strain.
