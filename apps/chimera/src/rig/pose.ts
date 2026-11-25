/**
 * A pose, as a body moves: every joint's angles in degrees, named as an anatomist names them,
 * each from anatomical zero (standing straight, arms hanging at the sides with the thumbs
 * forward, feet flat). Positive is the direction in the name: `flex` bends forward, `abduct`
 * takes a limb out from the body, `rotate` turns a limb's front outward (external rotation),
 * `turn` turns a forearm's palm down and back (pronation).
 *
 * Left and right take the same angles and are mirrored by the rig.
 */
export interface Arm {
  /** the shoulder girdle: the collarbone's outer end raised (shrug) and brought forward. These
   *  add to what the rig moves on its own as the arm rises (the scapulohumeral rhythm) */
  shrug: number;
  reach: number;
  /** how far the arm is raised from hanging (0) to straight up (180) */
  raise: number;
  /** which way it rises: 0 out to the side, 90 straight forward, beyond 90 across the body,
   *  negative behind */
  plane: number;
  /** the upper arm turned about its length: positive outward (external rotation) */
  rotate: number;
  elbow: number;
  /** the forearm turned about its length: positive palm down (pronation), negative palm up */
  turn: number;
  /** the hand bent towards the palm (flexion; negative: back) and towards the thumb (radial
   *  deviation; negative: towards the little finger) */
  wrist: number;
  deviate: number;
}

export interface Leg {
  /** the thigh raised forward (hip flexion; negative: back) */
  flex: number;
  /** the thigh out to the side */
  abduct: number;
  /** the thigh turned about its length: positive knee and toes outward */
  rotate: number;
  knee: number;
  /** the foot: toes up (dorsiflexion; negative: pointed), its sole turned in (inversion) */
  ankle: number;
  invert: number;
  /** the toes raised */
  toes: number;
}

export interface Pose {
  /** the whole body: tipped forward, leaned to its right, turned to its left (degrees), and
   *  moved (cm: forward, up, right) */
  body: { tip: number; lean: number; turn: number; x: number; y: number; z: number };
  /** the trunk bending forward, to its right and turning to its left, shared down the spine */
  spine: { flex: number; bend: number; twist: number };
  /** the head and neck likewise, shared down the neck */
  neck: { flex: number; bend: number; twist: number };
  arms: { R: Arm; L: Arm };
  legs: { R: Leg; L: Leg };
}

export const arm = (a: Partial<Arm> = {}): Arm => ({ shrug: 0, reach: 0, raise: 0, plane: 0, rotate: 0, elbow: 0, turn: 0, wrist: 0, deviate: 0, ...a });
export const leg = (l: Partial<Leg> = {}): Leg => ({ flex: 0, abduct: 0, rotate: 0, knee: 0, ankle: 0, invert: 0, toes: 0, ...l });

export function neutral(): Pose {
  return {
    body: { tip: 0, lean: 0, turn: 0, x: 0, y: 0, z: 0 },
    spine: { flex: 0, bend: 0, twist: 0 },
    neck: { flex: 0, bend: 0, twist: 0 },
    arms: { R: arm(), L: arm() },
    legs: { R: leg(), L: leg() },
  };
}

/**
 * How far each joint goes, degrees [least, most]: the usual adult active range of motion
 * (AAOS / clinical goniometry tables), with a little more where a performer reaches further.
 * Sliders stop here; poses can go past them deliberately (the rig doesn't clamp).
 */
export const RANGE = {
  arm: {
    shrug: [-10, 40],
    reach: [-25, 30],
    raise: [0, 180],
    plane: [-60, 140],
    rotate: [-90, 90],
    elbow: [0, 150],
    turn: [-90, 90],
    wrist: [-70, 80],
    deviate: [-35, 20],
  },
  leg: {
    flex: [-30, 135],
    abduct: [-30, 60],
    rotate: [-45, 100],
    knee: [0, 150],
    ankle: [-50, 25],
    invert: [-20, 35],
    toes: [-40, 70],
  },
  spine: { flex: [-30, 80], bend: [-35, 35], twist: [-45, 45] },
  neck: { flex: [-60, 50], bend: [-45, 45], twist: [-80, 80] },
  body: { tip: [-90, 90], lean: [-90, 90], turn: [-180, 180], x: [-60, 60], y: [-100, 60], z: [-60, 60] },
} as const;

type DeepPartial<T> = { [K in keyof T]?: T[K] extends object ? DeepPartial<T[K]> : T[K] };
/** a pose from what differs from neutral; `both` sets the two arms or legs alike */
export function pose(p: DeepPartial<Pose> & { bothArms?: Partial<Arm>; bothLegs?: Partial<Leg> } = {}): Pose {
  const n = neutral();
  Object.assign(n.body, p.body);
  Object.assign(n.spine, p.spine);
  Object.assign(n.neck, p.neck);
  for (const s of ["R", "L"] as const) {
    Object.assign(n.arms[s], p.bothArms, p.arms?.[s]);
    Object.assign(n.legs[s], p.bothLegs, p.legs?.[s]);
  }
  return n;
}

/** poses that test the rig: each joint near its limits, and whole-body poses that combine them */
export const PRESETS: Record<string, Pose> = {
  neutral: neutral(),
  "A pose": pose({ bothArms: { raise: 40 } }),
  "T pose": pose({ bothArms: { raise: 90 } }),
  "arms up": pose({ bothArms: { raise: 175, plane: 20 } }),
  "arms forward": pose({ bothArms: { raise: 90, plane: 90 } }),
  "arms folded": pose({ bothArms: { raise: 75, plane: 95, rotate: -85, elbow: 95 } }),
  "hands behind head": pose({ bothArms: { raise: 140, plane: 10, elbow: 145, rotate: 70 }, neck: { flex: -5 } }),
  "flex biceps": pose({ bothArms: { raise: 90, elbow: 130, rotate: 80, turn: -60 } }),
  "reach behind": pose({ bothArms: { raise: 45, plane: -50, rotate: -60, elbow: 90, turn: 40 } }),
  "elbows bent": pose({ bothArms: { elbow: 150, turn: -80 } }),
  "palms twist": pose({ arms: { R: { raise: 60, plane: 70, turn: 90 }, L: { raise: 60, plane: 70, turn: -90 } } }),
  "wrists": pose({ arms: { R: { raise: 30, plane: 60, elbow: 70, wrist: 80, deviate: 20 }, L: { raise: 30, plane: 60, elbow: 70, wrist: -70, deviate: -35 } } }),
  "high kick": pose({ legs: { R: { flex: 120 }, L: {} }, bothArms: { raise: 30 } }),
  "splits": pose({ legs: { R: { flex: 90 }, L: { flex: -30 } } }),
  "side splits": pose({ bothLegs: { abduct: 60, rotate: 30 } }),
  "squat": pose({ body: { y: -45, tip: 25 }, spine: { flex: 20 }, bothLegs: { flex: 125, knee: 145, ankle: 25, abduct: 15, rotate: 15 }, bothArms: { raise: 80, plane: 90 } }),
  "kneel": pose({ body: { y: -50 }, legs: { R: { flex: 90, knee: 90 }, L: { flex: -10, knee: 100, toes: 60, ankle: -10 } } }),
  "lotus": pose({
    body: { y: -70 },
    bothLegs: { flex: 85, abduct: 40, rotate: 55, knee: 150, ankle: -20, invert: 30 },
    arms: { R: { raise: 30, plane: 60, elbow: 40, turn: 30 }, L: { raise: 35, plane: 45, elbow: 70, rotate: 40, turn: -80 } },
  }),
  "sit": pose({ body: { y: -42 }, bothLegs: { flex: 90, knee: 90 }, bothArms: { raise: 20, plane: 60, elbow: 60 } }),
  "run": pose({
    spine: { flex: 10, twist: 10 },
    arms: { R: { raise: 40, plane: 95, rotate: -20, elbow: 95 }, L: { raise: 35, plane: -75, rotate: -10, elbow: 70 } },
    legs: { R: { flex: 70, knee: 90 }, L: { flex: -20, knee: 40, ankle: -30 } },
  }),
  "fetal": pose({ spine: { flex: 70 }, neck: { flex: 45 }, bothLegs: { flex: 130, knee: 150, ankle: -40 }, bothArms: { raise: 70, plane: 110, elbow: 140 } }),
  "backbend": pose({ spine: { flex: -30 }, neck: { flex: -55 }, bothArms: { raise: 170, plane: 60 } }),
  "twist & look": pose({ spine: { twist: 45, bend: 15 }, neck: { twist: 75, bend: 20 }, arms: { R: { raise: 60, plane: 120, elbow: 40 }, L: { raise: 30, plane: -30, elbow: 30 } } }),
};
