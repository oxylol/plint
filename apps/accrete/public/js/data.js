// Accrete · what exists in the game: the kinds of bodies, what can be bought, the milestones, the sky, the laws
// and the constants. Numbers only; the rules that use them are in game.js. Works in the browser and in node.
(function (root) {
  "use strict";

  // ── Bodies ────────────────────────────────────────────────
  // cost grows by GROWTH for each one owned (Cookie Clicker's 1.15); rate is mass per second for one of them.
  // A kind is shown once the cloud has gathered `reveal` of its cost; the last two need a constellation.
  const GROWTH = 1.15;
  const REVEAL = 0.4;
  const KINDS = [
    { id: "mote",     name: "Mote",     plural: "Motes",     cost: 10,     rate: 0.25 },
    { id: "pebble",   name: "Pebble",   plural: "Pebbles",   cost: 1200,   rate: 12 },
    { id: "boulder",  name: "Boulder",  plural: "Boulders",  cost: 15000,  rate: 105 },
    { id: "asteroid", name: "Asteroid", plural: "Asteroids", cost: 2.7e5,  rate: 1300 },
    { id: "moon",     name: "Moon",     plural: "Moons",     cost: 3.7e6,  rate: 15500 },
    { id: "planet",   name: "Planet",   plural: "Planets",   cost: 8.5e7,  rate: 290000 },
    { id: "giant",    name: "Giant",    plural: "Giants",    cost: 9e8,    rate: 2.6e6, node: "giants" },
    { id: "dwarf",    name: "Dwarf",    plural: "Dwarfs",    cost: 1.4e10, rate: 3e7,   node: "dwarfs" },
  ];
  // Owning this many of a kind doubles that kind (Pecorella: "make it bumpy"). After the list: every 50 more.
  const BUMPS = [10, 25, 50, 100, 150, 200];
  const BUMP_STEP = 50;
  const BUMP_MULT = 2;

  // ── Bought with mass, gone at ignition (shepherds stay for good) ──
  // need: what has to be true before it can be bought. own:[kind,n] · mass:n (gathered in this cloud) · comets:n (ever)
  const UPGRADES = [
    // the core
    { id: "grip1", set: "core", name: "Grip",        text: "Each pull also gathers 2% of your mass per second.", cost: 100,  need: { mass: 40 } },
    { id: "spin1", set: "core", name: "Spin",        text: "The core pulls by itself: 1 pull a second.",            cost: 3000, need: { mass: 1200 } },
    { id: "grip2", set: "core", name: "Firm grip",   text: "Each pull gathers 5% of your mass per second.",       cost: 4e4,  need: { mass: 1.5e4 } },
    { id: "spin2", set: "core", name: "Fast spin",   text: "The core pulls by itself: 5 pulls a second. Holding is no longer needed.", cost: 2e5, need: { mass: 8e4 } },
    { id: "grip3", set: "core", name: "Iron grip",   text: "Each pull gathers 10% of your mass per second.",      cost: 5e6,  need: { mass: 2e6 } },
    // links: the next kind out makes this kind stronger
    { id: "dust",  set: "inner", name: "Static",        text: "Motes x2.",                             cost: 500,   need: { own: [0, 15] } },
    { id: "link0", set: "inner", name: "Pebble herd",   text: "Motes gain 2% for every Pebble.",      cost: 6000,  need: { own: [1, 5] },  link: 0 },
    { id: "link1", set: "inner", name: "Boulder wake",  text: "Pebbles gain 2% for every Boulder.",   cost: 7e4,   need: { own: [2, 5] },  link: 1 },
    { id: "link2", set: "inner", name: "Asteroid belt", text: "Boulders gain 2% for every Asteroid.", cost: 8e5,   need: { own: [3, 5] },  link: 2 },
    { id: "link3", set: "inner", name: "Moon tide",     text: "Asteroids gain 2% for every Moon.",    cost: 1e7,   need: { own: [4, 5] },  link: 3 },
    { id: "link4", set: "outer", name: "Planet shade",  text: "Moons gain 2% for every Planet.",      cost: 1.2e8, need: { own: [5, 5] },  link: 4 },
    { id: "link5", set: "outer", name: "Giant's reach", text: "Planets gain 2% for every Giant.",     cost: 1.5e9, need: { own: [6, 5] },  link: 5 },
    { id: "link6", set: "outer", name: "Dwarf light",   text: "Giants gain 2% for every Dwarf.",      cost: 1.8e10, need: { own: [7, 5] }, link: 6 },
    // comets
    { id: "tail1", set: "comet", name: "Long tail",    text: "Surges last 2x as long.",        cost: 8e3,   need: { comets: 1, mass: 3e3 } },
    { id: "tail2", set: "comet", name: "Busy sky",     text: "Comets come 30% sooner.",           cost: 3e5,   need: { comets: 3, mass: 1e5 } },
    { id: "tail3", set: "comet", name: "Heavy comets", text: "Windfalls carry 2x as much.",    cost: 1.5e7, need: { comets: 7, mass: 5e6 } },
  ];
  const UPGRADE_SETS = [
    { id: "core",  name: "The core" },
    { id: "inner", name: "Inner links" },
    { id: "outer", name: "Outer links" },
    { id: "comet", name: "Comets" },
  ];
  const SET_MULT = 1.15;           // a full set of upgrades, to all bodies
  const LINK = 0.02;

  // A shepherd buys its kind for you. Bought once with mass, kept through every reset.
  const SHEPHERD = { own: 25, costFactor: 1000, share: 0.10 };
  // What the core and the hand do before any upgrade.
  // `start` is the mass a new game begins with: a little dust, so there is something to buy at once and no bar is empty.
  const CORE = { trickle: 1, pull: 1, holdRate: 5, start: 30 };

  // ── The comet: a bonus for being there. Pays inside a band that is shown before it is caught. ──
  const COMET = {
    first: [150, 240],             // seconds of open page before the first one
    wait: [120, 360],              // then between them
    ttl: 15,                       // seconds on screen
    windfall: [60, 120],           // seconds' worth of mass
    surge: { mult: 3, secs: [30, 60] },
    surgeChance: 0.4,
  };

  // ── Reset 1 · ignition: the cloud becomes a star, the star gives light ──
  const IGNITE = {
    at: 6.5e11,                    // mass gathered in the cloud
    root: 3,                       // light = (mass / at)^(1/root)
    passive: 0.25,                 // every light ever made: +25% to all bodies
    classMult: 2,                  // each class of star in this galaxy: x2 to all bodies
    // the class of the star by the mass of its cloud, in multiples of `at`
    classes: [
      { id: "M", name: "red dwarf",    at: 1 },
      { id: "K", name: "orange star",  at: 1e2 },
      { id: "G", name: "yellow star",  at: 1e4 },
      { id: "F", name: "white star",   at: 1e7 },
      { id: "A", name: "bright white star", at: 1e10 },
      { id: "B", name: "blue giant",   at: 1e14 },
      { id: "O", name: "blue beacon",  at: 1e18 },
    ],
  };

  // The sky: constellations whose stars are bought with light. after: the star that must be lit first.
  const NODES = [
    // The Furnace: more from every body
    { id: "kindling",  con: "furnace", name: "Kindling",   text: "All bodies x2.",                                    cost: 1 },
    { id: "bellows",   con: "furnace", name: "Bellows",    text: "All bodies x2.",                                    cost: 25,    after: "kindling" },
    { id: "crucible",  con: "furnace", name: "Crucible",   text: "Owning 10, 25, 50… of a kind gives x2.5, not x2.",  cost: 6000,  after: "bellows" },
    { id: "whiteheat", con: "furnace", name: "White heat", text: "All bodies gain 1% for every star in this galaxy, up to 500 stars.", cost: 2e5,   after: "crucible" },
    // The Deep: two more kinds of body
    { id: "giants", con: "deep", name: "Giants",     text: "A seventh kind of body: the Giant.",       cost: 5 },
    { id: "dwarfs", con: "deep", name: "Dwarfs",     text: "An eighth kind of body: the Dwarf.",       cost: 1500,  after: "giants" },
    { id: "reach",  con: "deep", name: "Long reach", text: "Links give 4% for every body, not 2%.",    cost: 5e4,   after: "dwarfs" },
    // The Seed: a cloud that does not start from nothing
    { id: "seed",    con: "seed", name: "Seed",     text: "Each cloud starts with 25 Motes.",                     cost: 1 },
    { id: "sapling", con: "seed", name: "Sapling",  text: "Each cloud starts with 10 of the first four kinds.",   cost: 12,   after: "seed" },
    { id: "roots",   con: "seed", name: "Roots",    text: "The core's upgrades stay through ignition.",           cost: 800,  after: "sapling" },
    { id: "oldwood", con: "seed", name: "Old wood", text: "Every upgrade stays through ignition.",                cost: 1e5,  after: "roots" },
    // The Shepherd: more done for you
    { id: "openhand", con: "shepherd", name: "Open hand",  text: "Shepherds spend up to 25% of your mass, not 10%.", cost: 3 },
    { id: "steward",  con: "shepherd", name: "Steward",    text: "Upgrades are bought for you.",                               cost: 50,   after: "openhand" },
    { id: "watch",    con: "shepherd", name: "Dawn watch", text: "The cloud ignites by itself at the light you set.",          cost: 400,  after: "steward" },
    { id: "herd",     con: "shepherd", name: "Herd",       text: "Shepherds buy as many as the share allows, at once.",        cost: 3000, after: "watch" },
    // The Net: comets
    { id: "widenet",  con: "net", name: "Wide net",  text: "Comets come 30% sooner.",                           cost: 2 },
    { id: "driftnet", con: "net", name: "Drift net", text: "A comet you miss is caught for you at half value.", cost: 200,   after: "widenet" },
    { id: "fullnet",  con: "net", name: "Full net",  text: "Comets are caught for you at full value.",          cost: 25000, after: "driftnet" },
    // The Lens: more light
    { id: "lens",   con: "lens", name: "Lens",   text: "Ignition gives x2 light.",                    cost: 100 },
    { id: "mirror", con: "lens", name: "Mirror", text: "Ignition gives x2 light.",                    cost: 12000, after: "lens" },
    { id: "prism",  con: "lens", name: "Prism",  text: "Light x1.5 for each class of star in this galaxy.",  cost: 4e5,   after: "mirror" },
  ];
  const CONSTELLATIONS = [
    { id: "furnace",  name: "The Furnace" },
    { id: "deep",     name: "The Deep" },
    { id: "seed",     name: "The Seed" },
    { id: "shepherd", name: "The Shepherd" },
    { id: "net",      name: "The Net" },
    { id: "lens",     name: "The Lens" },
  ];
  const CON_MULT = 2;              // a finished constellation, to all bodies

  // ── Reset 2 · collapse: the sky becomes a galaxy, the galaxy gives dark matter ──
  const COLLAPSE = {
    at: 1e6,                       // light made in this galaxy: 1 dark matter, and 1 more for every tenfold past it
    mass: 2,                       // all bodies x (1 + dark matter made in this universe)^mass
    light: 0.5,                    // light x (1 + dark matter made in this universe)^light
  };
  // Bought with dark matter, kept through collapses, gone at a bang. cost of level n = cost * step^n, rounded.
  const WELLS = [
    { id: "hole",    name: "Black hole", text: "Time in the cloud runs 1.3x faster.",                         cost: 1, step: 2,   max: 6 },
    { id: "nursery", name: "Nursery",    text: "Ignition gives x2 light.",                                    cost: 1, step: 2,   max: 6 },
    { id: "gazer",   name: "Stargazer",  text: "Stars in the sky are lit for you as soon as the light is there.", cost: 2, step: 1, max: 1 },
    { id: "memory",  name: "Memory",     text: "A new galaxy starts with 1 more finished constellation.",   cost: 1, step: 1.6, max: 6 },
  ];
  // The order in which Memory brings constellations back: what works for you first.
  const MEMORY = ["shepherd", "seed", "net", "furnace", "deep", "lens"];
  // A law bends the rules of one whole galaxy. Collapse that galaxy and its reward is yours for good.
  const LAWS = [
    { id: "thin",  name: "Thin dust",  rule: "Costs grow by 1.25, not 1.15.",        reward: "Costs grow by 1.14 for good." },
    { id: "flat",  name: "Flat",       rule: "Owning many of a kind gives nothing.", reward: "Those bumps are half again as strong, for good." },
    { id: "dark",  name: "Dark",       rule: "No comets, and the hand does nothing.", reward: "Comets are three times as strong, for good." },
    { id: "alone", name: "Alone",      rule: "Only every other kind of body exists.", reward: "All bodies x10, for good." },
    { id: "dim",   name: "Dim",        rule: "Ignition gives a tenth of the light.",  reward: "Light x5, for good." },
    { id: "bare",  name: "Bare",       rule: "Stars in the sky cost five times the light.", reward: "They cost half, for good." },
  ];

  // ── Reset 3 · bang: the galaxies become a universe, and one constant of it is yours to raise ──
  // dark matter made in this universe; each universe asks `add` more than the last
  const BANG = { at: 150, add: 80 };
  const CONSTANTS = [
    { id: "G",  name: "Gravity",   sym: "G", text: "All bodies x100 for each step." },
    { id: "c",  name: "Speed",     sym: "c", text: "Time in the cloud runs 2x faster for each step." },
    { id: "h",  name: "Grain",     sym: "h", text: "Costs grow 0.01 less for each step." },
    { id: "L",  name: "Radiance",  sym: "Λ", text: "Light x10 for each step." },
    { id: "O",  name: "Density",   sym: "Ω", text: "Each collapse gives its dark matter 1 more time for each step." },
    { id: "a",  name: "Memory",    sym: "α", text: "A new universe keeps 1 more level of every well." },
  ];

  // ── Milestones, in sets. Each pays MS_MULT to all bodies, a full set MS_SET more. ──
  // stat: which counter (game.js `stat`), at: the target. lower: true means smaller is better (seconds).
  const MS_MULT = 1.04, MS_SET = 1.2;
  const M = (id, set, name, stat, at, extra) => Object.assign({ id, set, name, stat, at }, extra || {});
  const MILESTONES = [
    M("sp1", "spark", "A mote",        "bought", 1),
    M("sp2", "spark", "A handful",     "bought", 25),
    M("sp3", "spark", "Three kinds",   "kinds", 3),
    M("sp4", "spark", "A thousand",    "mass", 1e3),
    M("sp5", "spark", "Caught one",    "comets", 1),
    M("ga1", "gather", "A hill",       "mass", 1e5),
    M("ga2", "gather", "A mountain",   "mass", 1e7),
    M("ga3", "gather", "A small moon", "mass", 1e9),
    M("ga4", "gather", "A world",      "mass", 1e11),
    M("ga5", "gather", "A sun's worth", "mass", 1e13),
    M("cr1", "crowd", "25 of a kind",  "most", 25),
    M("cr2", "crowd", "50 of a kind",  "most", 50),
    M("cr3", "crowd", "100 of a kind", "most", 100),
    M("cr4", "crowd", "150 of a kind", "most", 150),
    M("cr5", "crowd", "200 of a kind", "most", 200),
    M("fa1", "family", "Four kinds",   "kinds", 4),
    M("fa2", "family", "Six kinds",    "kinds", 6),
    M("fa3", "family", "Seven kinds",  "kinds", 7),
    M("fa4", "family", "All eight",    "kinds", 8),
    M("co1", "comets", "7 comets",     "comets", 7),
    M("co2", "comets", "27 comets",    "comets", 27),
    M("co3", "comets", "77 comets",    "comets", 77),
    M("co4", "comets", "277 comets",   "comets", 277),
    M("sh1", "flock", "A shepherd",    "shepherds", 1),
    M("sh2", "flock", "Four shepherds", "shepherds", 4),
    M("sh3", "flock", "A full flock",  "shepherds", 8),
    M("st1", "stars", "First star",    "ignitions", 1,    { layer: 1 }),
    M("st2", "stars", "5 stars",       "ignitions", 5,    { layer: 1 }),
    M("st3", "stars", "25 stars",      "ignitions", 25,   { layer: 1 }),
    M("st4", "stars", "100 stars",     "ignitions", 100,  { layer: 1 }),
    M("st5", "stars", "1,000 stars",   "ignitions", 1000, { layer: 1 }),
    M("cl1", "cool", "A red dwarf",    "classM", 1, { layer: 1 }),
    M("cl2", "cool", "An orange star", "classK", 1, { layer: 1 }),
    M("cl3", "cool", "A yellow star",  "classG", 1, { layer: 1 }),
    M("ho1", "hot", "A white star",    "classF", 1, { layer: 1 }),
    M("ho2", "hot", "A bright white",  "classA", 1, { layer: 1 }),
    M("ho3", "hot", "A blue giant",    "classB", 1, { layer: 1 }),
    M("ho4", "hot", "A blue beacon",   "classO", 1, { layer: 1 }),
    M("qu1", "quick", "A star in 30 min", "fastest", 1800, { layer: 1, lower: true }),
    M("qu2", "quick", "A star in 10 min", "fastest", 600,  { layer: 1, lower: true }),
    M("qu3", "quick", "A star in 3 min",  "fastest", 180,  { layer: 1, lower: true }),
    M("qu4", "quick", "A star in 1 min",  "fastest", 60,   { layer: 1, lower: true }),
    M("sk1", "sky", "A constellation",     "cons", 1, { layer: 1 }),
    M("sk2", "sky", "Two constellations",  "cons", 2, { layer: 1 }),
    M("sk3", "sky", "Four constellations", "cons", 4, { layer: 1 }),
    M("sk4", "sky", "The whole sky",       "cons", 6, { layer: 1 }),
    M("gx1", "galaxies", "First galaxy", "collapses", 1,  { layer: 2 }),
    M("gx2", "galaxies", "3 galaxies",   "collapses", 3,  { layer: 2 }),
    M("gx3", "galaxies", "10 galaxies",  "collapses", 10, { layer: 2 }),
    M("gx4", "galaxies", "30 galaxies",  "collapses", 30, { layer: 2 }),
    M("lw1", "laws", "A law kept",   "laws", 1, { layer: 2 }),
    M("lw2", "laws", "Three laws",   "laws", 3, { layer: 2 }),
    M("lw3", "laws", "All six laws", "laws", 6, { layer: 2 }),
    M("dk1", "depth", "10 dark matter",    "dark", 10,  { layer: 2 }),
    M("dk2", "depth", "100 dark matter",   "dark", 100, { layer: 2 }),
    M("dk3", "depth", "500 dark matter",   "dark", 500, { layer: 2 }),
    M("dk4", "depth", "2,000 dark matter", "dark", 2000, { layer: 3 }),
    M("hv1", "heavy", "1e18 gathered", "mass", 1e18, { layer: 1 }),
    M("hv2", "heavy", "1e25 gathered", "mass", 1e25, { layer: 1 }),
    M("hv3", "heavy", "1e32 gathered", "mass", 1e32, { layer: 2 }),
    M("hv4", "heavy", "1e40 gathered", "mass", 1e40, { layer: 3 }),
    M("un1", "universe", "First universe", "bangs", 1, { layer: 3 }),
    M("un2", "universe", "Two constants",  "bangs", 2, { layer: 3 }),
    M("un3", "universe", "Four constants", "bangs", 4, { layer: 3 }),
    M("un4", "universe", "Every constant", "bangs", 6, { layer: 3 }),
  ];
  const MILESTONE_SETS = [
    { id: "spark",    name: "Spark" },
    { id: "gather",   name: "Gathering" },
    { id: "crowd",    name: "Crowd" },
    { id: "family",   name: "Family" },
    { id: "comets",   name: "Comets" },
    { id: "flock",    name: "Flock" },
    { id: "stars",    name: "Stars",        layer: 1 },
    { id: "cool",     name: "Cool stars",   layer: 1 },
    { id: "hot",      name: "Hot stars",    layer: 1 },
    { id: "quick",    name: "Quick",        layer: 1 },
    { id: "sky",      name: "Sky",          layer: 1 },
    { id: "heavy",    name: "Heavy",        layer: 1 },
    { id: "galaxies", name: "Galaxies",     layer: 2 },
    { id: "laws",     name: "Laws",         layer: 2 },
    { id: "depth",    name: "Depth",        layer: 2 },
    { id: "universe", name: "Universe",     layer: 3 },
  ];

  const D = {
    GROWTH, REVEAL, KINDS, BUMPS, BUMP_STEP, BUMP_MULT, UPGRADES, UPGRADE_SETS, SET_MULT, LINK, SHEPHERD, CORE, COMET,
    IGNITE, NODES, CONSTELLATIONS, CON_MULT, COLLAPSE, WELLS, MEMORY, LAWS, BANG, CONSTANTS,
    MS_MULT, MS_SET, MILESTONES, MILESTONE_SETS,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = D;
  else root.AccreteData = D;
})(typeof self !== "undefined" ? self : this);
