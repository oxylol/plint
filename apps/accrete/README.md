# Accrete

An incremental game at `https://accrete.${DOMAIN}`: a star from dust. A spark in the middle of a cloud; the
dust, rocks, moons and planets you own orbit it and feed it. Built on Filip's brief: "the most addicting
incremental game", meaning proved to hook and built on theories where there is no proof, with the visuals a big
part of it. PC first; a phone gets the same game in a simpler layout.

## What is in it

- **The cloud.** Eight kinds of body (Mote to Dwarf), each 1.15 times dearer than the last of its kind. Owning
  10, 25, 50, 100… of a kind doubles it. Upgrades in sets, shepherds that buy for you and are never taken
  back, comets that cross the scene with a windfall or a surge. Holding on the picture pulls mass in; after a
  few minutes the core pulls by itself.
- **Reset 1, ignition.** The cloud becomes a star and gives light (the cube root of its mass). Light buys stars
  in six constellations; the class of the star (M K G F A B O) is kept, each x2.
- **Reset 2, collapse.** The sky becomes a galaxy and gives dark matter (one, and one more for every tenfold of
  light). Dark matter digs wells; a galaxy can be put under one of six laws, kept for good once collapsed.
- **Reset 3, bang.** One of six constants is raised for the next universe. Raise all six and the game says it
  is complete, and goes on for whoever wants.
- **Away.** It earns at the full rate with no cap while the page is closed; on return a card says what was
  gained and where you left off. Nothing is ever taken away.
- **WHY** (the `?` button): every hook in the game, graded proven / supported / theory / designers' habit /
  left out, with its source. The research behind it is in `.scratch/limit-research-*.md` (not in git).
- **Saved in the browser only** (Filip's choice): `localStorage`, every few seconds, with an older copy beside
  it. Export and import as text under settings. The PC and the phone are two separate games.

## Files

```
server.js            hands out public/ on :5240, nothing else
public/js/big.js     numbers past 1.8e308 (mantissa x 10^exponent)
public/js/data.js    every number in the game: prices, rates, thresholds, texts
public/js/game.js    the rules; no page, no clock; the browser, the tests and the bot all run this file
public/js/format.js  how numbers and times are written (never more than four digits)
public/js/scene.js   the picture: one WebGL canvas (core, bodies, inflow, comets, sky, galaxy, universe)
public/js/audio.js   the sound, all made with Web Audio
public/js/ui.js      the panels and cards
public/js/main.js    saving, time, the hand, and what is shown and sounded for each thing that happens
public/js/why.js     the evidence list
```

## Tuning

Everything that sets the pace is in `data.js`. After changing a number, let the bot play it:

```
node test/bot.js --first        the first sitting (75 min), minute by minute, with the checks from the spec
node test/bot.js --first --idle the same for someone who never touches it
node test/bot.js --days 40      the whole game on hourly visits (takes a few minutes)
TUNE='D.IGNITE.at = 1e12' node test/bot.js --first     try a number without editing the file
```

Where the bot stands (2026-11-13): first ignition after 46 minutes of active play (about 100 if you never touch
it), first collapse on day 2, first bang on day 5, the end on day 13, visiting once an hour by day. A person
visits less and decides differently: the bot is a floor, not a forecast. Layers 2 and 3 have only ever been
played by the bot.

How the pace is built, for whoever retunes it:

- With eight kinds and prices that climb by 1.15, a cloud grows fast while new kinds are arriving and almost
  linearly after the last one. That wall is why ignition exists, and where it is placed (`IGNITE.at`).
- A multiplier on everything is the same as time running faster: twice the mass per second reaches any point
  in half the time. So a reset's strength can be read as "how much sooner".
- Light is the cube root of mass, so light per cloud grows slowly; the sky's prices (`NODES`) are spread along
  what the bot earns, about one star per visit.
- Dark matter is one per tenfold of light on purpose: roots made the upper layers run away in the bot's hands.

## Checks

```
node test/unit.js     the arithmetic (140 325 cases)
node test/rules.js    prices, time away, saving, the resets, how numbers are written
node test/bot.js      pace, and the checks that must hold at every moment
test/page.js          the page in a browser (see its head for how it is run; never against the live port)
test/shot.js          pictures of set-up states, to look at
```

## What it does not do

No save on Plint, no accounts, no other players, nothing sent to a phone, no daily rewards or streaks, no money.
With the page closed nothing runs: the time away is worked out when you come back.
