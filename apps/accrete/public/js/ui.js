// Accrete · the page around the picture: the number, the bodies, what is next, what can be bought, the wider
// views, and the cards that come up. It reads the state and calls back into main.js when something is pressed.
(function (root) {
  "use strict";
  const G = root.Accrete, D = root.AccreteData, F = root.Fmt, Big = root.Big, WHY = root.AccreteWhy;
  const NK = D.KINDS.length;
  const $ = (id) => document.getElementById(id);
  const h = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
  // write only what changed: most of the page is the same from one frame to the next
  const put = (el, text) => { if (el._t !== text) { el._t = text; el.textContent = text; } };
  const flag = (el, name, on) => { on = !!on; if (el["_c" + name] !== on) { el["_c" + name] = on; el.classList.toggle(name, on); } };
  const show = (el, on) => { on = !!on; if (el._h !== !on) { el._h = !on; el.hidden = !on; } };
  const width = (el, f) => { const w = Math.round(Math.max(0, Math.min(1, f)) * 1000) / 10 + "%"; if (el._w !== w) { el._w = w; el.style.width = w; } };
  const rgb = (c) => c.map((x) => Math.round(x * 255)).join(", ");
  const ICON = {
    sound: '<svg viewBox="0 0 24 24"><path d="M4 9.5v5h3.5L12 18V6L7.5 9.5z"/><path d="M15.5 9a4 4 0 0 1 0 6M18 6.5a7.5 7.5 0 0 1 0 11"/></svg>',
    mute: '<svg viewBox="0 0 24 24"><path d="M4 9.5v5h3.5L12 18V6L7.5 9.5z"/><path d="M16 9.5l5 5M21 9.5l-5 5"/></svg>',
    why: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="8.5"/><path d="M9.7 9.6a2.4 2.4 0 1 1 3.4 2.2c-.8.4-1.1.9-1.1 1.7M12 16.6v.1"/></svg>',
    gear: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="2.8"/><path d="M12 3.5v2.2M12 18.3v2.2M3.5 12h2.2M18.3 12h2.2M6 6l1.6 1.6M16.4 16.4L18 18M6 18l1.6-1.6M16.4 7.6L18 6"/></svg>',
  };
  const COL = { warm: "255, 190, 120", light: "255, 231, 163", dark: "185, 162, 255", echo: "127, 227, 255", good: "125, 240, 180", plain: "200, 210, 235" };
  const VIEWS = [["cloud", "Cloud"], ["sky", "Sky"], ["galaxy", "Galaxy"], ["universe", "Universe"]];
  const kindCol = (k) => rgb(root.AccreteSceneKinds[k].col);
  const massAt = (v) => F.num(v) + " mass";
  const an = (name) => (/^[aeiou]/i.test(name) ? "an " : "a ") + name;

  function UI(act) {
    const el = {};
    let tab = "upgrades", lawPick = null, constPick = null;
    const app = $("app");

    // ── built once ──
    // how many to buy
    const modes = [[1, "1"], [10, "10"], ["bump", "to next x2"], ["max", "max"]];
    el.modes = modes.map(([m, label]) => { const b = h("button", "", label); b.onclick = () => act.buyMode(m); $("buymode").appendChild(b); return [m, b]; });

    // the bodies
    el.body = D.KINDS.map((kd, k) => {
      const row = h("div", "body"); row.style.setProperty("--c", kindCol(k)); row.hidden = true; row._h = true;
      const dot = h("div", "dot"); dot.style.setProperty("--s", String(0.45 + 0.55 * (k + 1) / NK));
      const who = h("div", "who"), name = h("b", "name", kd.name), own = h("span", "own", "0"), share = h("span", "share");
      who.append(name, own);
      const bump = h("div", "bump"), bar = h("div", "bar"), bari = h("i"), btxt = h("span");
      bar.appendChild(bari); bump.append(bar, btxt, share);
      const buy = h("button", "buy"), fill = h("span", "fill"), n = h("span", "n"), price = h("span", "price fig");
      buy.append(fill, n, price); buy.onclick = () => act.buy(k);
      buy.setAttribute("aria-label", "Buy " + kd.name);
      const shep = h("button", "shep", "herded"); shep.hidden = true; shep.onclick = (e) => { e.stopPropagation(); act.toggleShepherd(k); };
      who.appendChild(shep);
      row.append(dot, who, bump, buy);
      $("bodies").appendChild(row);
      return { row, own, share, bari, btxt, bump, buy, fill, n, price, shep };
    });
    el.nextKind = h("div", "outline"); el.nextKind.hidden = true;
    { const d = h("div", "dot", "?"), w = h("div"), t = h("b", "", "A new kind of body"), c = h("div", "cond"), bar = h("div", "bar"), i = h("i"); bar.appendChild(i); w.append(t, c, bar); el.nextKind.append(d, w); el.nextKindCond = c; el.nextKindBar = i; el.nextKindBarWrap = bar; }
    $("bodies").appendChild(el.nextKind);

    // tools
    el.sound = h("button"); el.sound.title = "Sound"; el.sound.onclick = () => act.mute();
    const why = h("button"); why.innerHTML = ICON.why; why.title = "Why this game hooks: the evidence"; why.onclick = () => cardWhy();
    const gear = h("button"); gear.innerHTML = ICON.gear; gear.title = "Settings"; gear.onclick = () => cardSettings();
    $("tools").append(el.sound, why, gear);

    // what is next: four rows that are reused
    el.goal = [0, 1, 2, 3].map(() => {
      const g = h("div", "goal"), top = h("div", "top"), label = h("span", "label"), gift = h("span", "gift"), eta = h("span", "eta"), bar = h("div", "bar"), i = h("i"), of = h("div", "of");
      bar.appendChild(i); top.append(label, gift, eta); g.append(top, bar, of); g.hidden = true; g._h = true;
      $("goals").appendChild(g);
      return { g, label, gift, eta, i, of };
    });

    // upgrades | milestones
    el.tabs = [["upgrades", "Upgrades"], ["milestones", "Milestones"]].map(([id, label]) => {
      const b = h("button"), t = h("span", "", label), badge = h("span", "badge"); badge.hidden = true;
      b.append(t, badge); b.onclick = () => { tab = id; };
      $("tabs").appendChild(b);
      return { id, b, badge };
    });
    function setBox(name, bonus) {
      const box = h("section", "set"), head = h("header"), b = h("b", "", name), n = h("span"), em = h("em", "", bonus || "");
      head.append(b, n, em); box.appendChild(head);
      return { box, n, em };
    }
    function cardRow(title, text) {
      const b = h("button", "cardrow"), fill = h("span", "fill"), t = h("span", "t", title), d = h("span", "d", text), p = h("span", "p fig"), ps = h("small"), pv = h("span");
      p.append(ps, pv); b.append(fill, t, d, p);
      return { b, fill, t, d, p, ps, pv };
    }
    el.upSets = D.UPGRADE_SETS.map((set) => {
      const sb = setBox(set.name, F.mult(D.SET_MULT) + " when whole");
      const rows = D.UPGRADES.filter((u) => u.set === set.id).map((u) => { const r = cardRow(u.name, u.text); r.u = u; r.b.onclick = () => act.upgrade(u.id); sb.box.appendChild(r.b); return r; });
      $("upgrades").appendChild(sb.box);
      return Object.assign(sb, { set, rows });
    });
    el.shepSet = setBox("Shepherds", "they stay for good");
    el.shepRows = D.KINDS.map((kd, k) => { const r = cardRow("Shepherd of " + kd.plural, ""); r.b.style.setProperty("--c", kindCol(k)); r.b.onclick = () => act.shepherd(k); el.shepSet.box.appendChild(r.b); return r; });
    $("upgrades").appendChild(el.shepSet.box);

    el.msSets = D.MILESTONE_SETS.map((set) => {
      const sb = setBox(set.name, F.mult(D.MS_SET) + " when whole"), pills = h("div", "pills");
      sb.box.appendChild(pills);
      const rows = G.sets.milestones[set.id].map((m) => {
        const p = h("div", "pill"), t = h("span", "", m.name), sm = h("small"), bar = h("div", "bar"), i = h("i");
        bar.appendChild(i); p.append(sm, t, bar); pills.appendChild(p);
        return { m, p, sm, bar, i };
      });
      $("milestones").appendChild(sb.box);
      return Object.assign(sb, { set, rows });
    });
    el.msMore = h("div", "outline"); el.msMore.append(h("div", "dot", "?"), (() => { const w = h("div"); w.append(h("b", "", "More sets"), h("div", "cond", "further out")); return w; })());
    $("milestones").appendChild(el.msMore);

    // views
    el.views = VIEWS.map(([id, label]) => { const b = h("button", "", label); b.onclick = () => act.view(id); $("views").appendChild(b); return { id, b }; });
    // a phone's tab bar
    el.tabbar = [["bodies", "Bodies"], ["cloud", "Upgrades"], ["sky", "Sky"], ["galaxy", "Galaxy"], ["universe", "Universe"]].map(([id, label]) => {
      const b = h("button", "", label); b.onclick = () => act.tab(id); $("tabbar").appendChild(b); return { id, b };
    });
    $("go").onclick = () => act.go();

    // ── the sky ──
    const sky = $("pane-sky");
    {
      sky.style.setProperty("--c", COL.light);
      el.light = h("div", "big-n fig"); el.lightSub = h("div", "sub");
      el.classes = h("div", "classes");
      el.classChips = D.IGNITE.classes.map((k) => { const i = h("i", "", k.id); i.title = an(k.name); i.style.setProperty("--c", rgb(root.AccreteSceneClass[k.id])); el.classes.appendChild(i); return i; });
      el.classNext = h("div", "sub");
      el.watch = h("div", "field");
      { const lab = h("label", "", "Ignite by itself when the cloud gives at least"), inp = h("input"), sw = h("button", "switch"); inp.type = "number"; inp.min = "1"; inp.step = "1"; inp.onchange = () => act.watch(null, Number(inp.value)); sw.setAttribute("role", "switch"); sw.onclick = () => act.watch(sw.getAttribute("aria-checked") !== "true", null); el.watch.append(lab, inp, sw); el.watchInput = inp; el.watchSwitch = sw; }
      sky.append(h("h2", "", "Light"), el.light, el.lightSub, h("h2", "", "Classes of star in this galaxy"), el.classes, el.classNext, el.watch);
      el.cons = D.CONSTELLATIONS.map((con) => {
        const sb = setBox(con.name, F.mult(D.CON_MULT) + " when whole");
        const rows = G.sets.constellations[con.id].map((n) => { const r = cardRow(n.name, n.text); r.n = n; r.b.style.setProperty("--c", COL.light); r.b.onclick = () => act.node(n.id); sb.box.appendChild(r.b); return r; });
        sky.appendChild(sb.box);
        return Object.assign(sb, { con, rows });
      });
    }
    // ── the galaxy ──
    const gal = $("pane-galaxy");
    {
      gal.style.setProperty("--c", COL.dark);
      el.dark = h("div", "big-n fig"); el.darkSub = h("div", "sub");
      el.lawNow = h("div", "sub");
      el.colGoal = h("div", "goal"); el.colGoal.style.setProperty("--c", COL.dark);
      { const top = h("div", "top"), label = h("span", "label", "Collapse"), eta = h("span", "eta"), bar = h("div", "bar"), i = h("i"), of = h("div", "of"); bar.appendChild(i); top.append(label, eta); el.colGoal.append(top, bar, of); el.colBar = i; el.colOf = of; el.colEta = eta; }
      gal.append(h("h2", "", "Dark matter"), el.dark, el.darkSub, el.colGoal, el.lawNow);
      el.wellSet = setBox("Wells", "gone at a bang");
      el.wells = D.WELLS.map((w) => { const r = cardRow(w.name, w.text); r.w = w; r.b.style.setProperty("--c", COL.dark); r.b.onclick = () => act.well(w.id); el.wellSet.box.appendChild(r.b); return r; });
      gal.appendChild(el.wellSet.box);
      el.lawSet = setBox("Laws", "kept for good");
      el.laws = D.LAWS.map((l) => { const r = cardRow(l.name, ""); r.l = l; r.b.style.setProperty("--c", COL.dark); r.b.disabled = true; el.lawSet.box.appendChild(r.b); return r; });
      gal.appendChild(el.lawSet.box);
    }
    // ── the universe ──
    const uni = $("pane-universe");
    {
      uni.style.setProperty("--c", COL.echo);
      el.echo = h("div", "big-n fig"); el.echoSub = h("div", "sub");
      el.bangGoal = h("div", "goal"); el.bangGoal.style.setProperty("--c", COL.echo);
      { const top = h("div", "top"), label = h("span", "label", "Bang"), bar = h("div", "bar"), i = h("i"), of = h("div", "of"); bar.appendChild(i); top.append(label); el.bangGoal.append(top, bar, of); el.bangBar = i; el.bangOf = of; }
      uni.append(h("h2", "", "Constants"), el.echo, el.echoSub, el.bangGoal);
      el.constSet = setBox("The six constants", "kept for good");
      el.consts = D.CONSTANTS.map((k) => { const r = cardRow(k.name, k.text); r.k = k; r.b.style.setProperty("--c", COL.echo); r.b.disabled = true; el.constSet.box.appendChild(r.b); return r; });
      uni.appendChild(el.constSet.box);
      el.endNote = h("div", "sub"); uni.appendChild(el.endNote);
    }
    // names of the constellations, over the picture
    el.conNames = D.CONSTELLATIONS.map((con) => { const d = h("div", "conname"), t = h("span", "", con.name), sm = h("small"); d.append(t, sm); d.hidden = true; $("labels").appendChild(d); return { con, d, sm }; });
    el.cometTag = h("div", "tag"); el.cometTag.hidden = true; $("labels").appendChild(el.cometTag);

    // ── every frame: the numbers that move ──
    this.fast = function (s, c) {
      put($("mass"), F.num(s.mass));
      put($("rate"), F.num(c.perSec));
      const surge = $("surge");
      show(surge, s.surge.left > 0);
      if (s.surge.left > 0) put(surge, "surge " + F.mult(s.surge.mult) + " · " + F.time(s.surge.left));
      for (let k = 0; k < NK; k++) {
        const b = el.body[k];
        if (b.row._h) continue;
        const n = G.amount(s, c, k, s.set.buy), price = G.cost(s, c, k, n), can = G.afford(s.mass, price);
        flag(b.buy, "can", can);
        if (!can) width(b.fill, s.mass.div(price).toNumber());
        put(b.n, "+" + F.int(n)); put(b.price, F.num(price));
      }
    };

    // ── a few times a second: everything else ──
    this.slow = function (s, c, scene) {
      const L = G.layer(s), view = scene.view();
      app.dataset.view = view; app.dataset.layer = String(L);
      for (const [m, b] of el.modes) b.setAttribute("aria-pressed", String(s.set.buy === m));
      // the purse: light, dark matter
      const purse = $("purse");
      const coins = [];
      if (L >= 1) coins.push([COL.light, F.int(s.light) + " light"]);
      if (L >= 2) coins.push([COL.dark, F.int(s.dark) + " dark matter"]);
      const key = coins.map((x) => x[1]).join("|");
      if (purse._k !== key) { purse._k = key; purse.textContent = ""; for (const [col, text] of coins) { const sp = h("span"), i = h("i"); sp.style.setProperty("--c", col); sp.append(i, document.createTextNode(text)); purse.appendChild(sp); } }

      // bodies
      const total = c.rate;
      for (let k = 0; k < NK; k++) {
        const b = el.body[k], on = G.shown(s, c, k) && c.exists[k];
        if (on && b.row._h) { show(b.row, true); if (s.own[k] === 0) { b.row.classList.add("new"); setTimeout(() => b.row.classList.remove("new"), 700); } }
        else show(b.row, on);
        if (!on) continue;
        put(b.own, F.int(s.own[k]));
        put(b.share, s.own[k] ? F.num(c.kind[k].mul(c.speed)) + "/s" : F.num(Big.from(D.KINDS[k].rate).mul(c.global).mul(c.speed)) + "/s each");
        const flat = s.law === "flat";
        if (!flat) { const nb = G.nextBump(s.own[k]), pb = G.prevBump(s.own[k]); width(b.bari, (s.own[k] - pb) / (nb - pb)); put(b.btxt, F.mult(c.bump) + " at " + nb); }
        else { width(b.bari, 0); put(b.btxt, "no bumps"); }
        show(b.shep, !!s.shep[k]);
        if (s.shep[k]) { flag(b.shep, "off", !!s.shepOff[k]); put(b.shep, s.shepOff[k] ? "resting" : "herded"); b.shep.title = s.shepOff[k] ? "The shepherd of " + D.KINDS[k].plural + " is resting. Press to wake." : "A shepherd buys " + D.KINDS[k].plural + " for you. Press to rest."; }
      }
      const outs = G.outlines(s, c), goals = G.goals(s, c);
      const ok = outs.find((o) => o.what === "kind");
      show(el.nextKind, !!ok);
      if (ok) {
        put(el.nextKindCond, ok.at ? "at " + massAt(ok.at) + " gathered in this cloud" : ok.cond);
        const g = goals.find((x) => x.kind === "reveal");
        show(el.nextKindBarWrap, !!g); if (g) width(el.nextKindBar, g.frac);
      }

      // what is next: the nearest of each sort
      const pick = [];
      const big = goals.filter((g) => g.kind === "buy" || g.kind === "layer" || g.kind === "class" || g.kind === "reveal");
      if (big[0]) pick.push(big[0]);
      const bump = goals.find((g) => g.kind === "bump"); if (bump) pick.push(bump);
      const ms = goals.find((g) => g.kind === "milestone"); if (ms) pick.push(ms);
      if (big[1]) pick.push(big[1]);
      el.goal.forEach((r, i) => {
        const g = pick[i];
        show(r.g, !!g);
        if (!g) return;
        const col = g.kind === "bump" ? kindCol(g.k) : g.kind === "milestone" ? COL.good : g.kind === "class" ? rgb(root.AccreteSceneClass[g.id.slice(5)]) : g.kind === "reveal" ? COL.plain : COL.warm;
        if (r.g._col !== col) { r.g._col = col; r.g.style.setProperty("--c", col); }
        put(r.label, g.label);
        const gift = g.kind === "bump" ? F.mult(c.bump) : g.kind === "milestone" ? F.mult(D.MS_MULT) : g.kind === "class" ? g.reward : "";
        show(r.gift, !!gift); if (gift) put(r.gift, gift);
        put(r.eta, g.frac >= 1 ? "ready" : g.eta < 1 ? "now" : g.eta < 6 * 3600 ? "in " + F.time(g.eta) : "");
        width(r.i, g.frac);
        const cur = g.cur instanceof Big ? F.num(g.cur) : g.lower ? (isFinite(g.cur) ? F.time(g.cur) : "no star yet") : F.int(g.cur);
        const at = g.lower ? F.time(g.at) : g.cur instanceof Big ? F.num(g.at) : F.int(g.at);
        put(r.of, cur + " / " + at + (g.unit ? " " + g.unit : "") + (g.set ? " · " + g.set : ""));
      });

      // upgrades
      let canCount = 0;
      const shutShown = {};
      for (const o of outs) if (o.what === "upgrade") shutShown[o.id] = o.cond; else if (o.what === "shepherd") shutShown["s" + o.k] = o.cond;
      for (const set of el.upSets) {
        let have = 0, any = false;
        for (const r of set.rows) {
          const u = r.u, held = !!s.up[u.id], open = G.upgradeOpen(s, c, u), shut = !held && !open && shutShown[u.id];
          if (held) have++;
          const on = held || open || !!shut;
          if (on && r.b._h && open) { r.b.classList.add("new"); setTimeout(() => r.b.classList.remove("new"), 700); }
          show(r.b, on); any = any || on;
          if (!on) continue;
          flag(r.b, "have", held); flag(r.b, "shut", !!shut); r.b.disabled = held || !!shut;
          const can = open && G.afford(s.mass, Big.from(u.cost));
          flag(r.b, "can", can); if (can) canCount++;
          if (open && !can) width(r.fill, s.mass.div(u.cost).toNumber()); else if (!open) width(r.fill, 0);
          put(r.d, shut ? "Opens when you " + shut + "." : u.text);
          put(r.ps, held ? "" : shut ? "" : "mass"); put(r.pv, held || shut ? "" : F.num(u.cost));
        }
        show(set.box, any);
        put(set.n, have + " / " + set.rows.length); flag(set.box, "full", have === set.rows.length);
      }
      {
        let have = 0, any = false;
        el.shepRows.forEach((r, k) => {
          const held = !!s.shep[k], open = G.shepherdOpen(s, c, k), shut = !held && !open && shutShown["s" + k];
          if (held) have++;
          const on = held || open || !!shut;
          show(r.b, on); any = any || on;
          if (!on) return;
          flag(r.b, "have", held); flag(r.b, "shut", !!shut); r.b.disabled = held || !!shut;
          const price = G.shepherdCost(k), can = open && G.afford(s.mass, price);
          flag(r.b, "can", can); if (can) canCount++;
          if (open && !can) width(r.fill, s.mass.div(price).toNumber()); else if (!open) width(r.fill, 0);
          put(r.d, shut ? "Opens when you " + shut + "." : "Buys " + D.KINDS[k].plural + " for you with up to " + F.pct(c.share) + " of your mass. Stays through every reset.");
          put(r.ps, held || shut ? "" : "mass"); put(r.pv, held || shut ? "" : F.num(price));
        });
        show(el.shepSet.box, any); put(el.shepSet.n, have + " / " + NK);
      }
      // milestones
      const got = G.msCounts(s);
      let hidden = 0;
      for (const set of el.msSets) {
        const on = (set.set.layer || 0) <= L;
        show(set.box, on);
        if (!on) { hidden++; continue; }
        let have = 0, next = true;
        for (const r of set.rows) {
          const done = !!s.ms[r.m.id];
          if (done) have++;
          const now = !done && next && (r.m.layer || 0) <= L;
          if (now) next = false;
          flag(r.p, "done", done); flag(r.p, "now", now);
          show(r.bar, now); show(r.sm, now);
          if (now) {
            const v = G.stat(s, r.m.stat);
            width(r.i, G.msFrac(s, r.m));
            put(r.sm, r.m.lower ? (isFinite(v) ? F.time(v) : "—") + " / " + F.time(r.m.at) : (v instanceof Big ? F.num(v) : F.int(v)) + " / " + (v instanceof Big ? F.num(r.m.at) : F.int(r.m.at)));
          }
        }
        put(set.n, have + " / " + set.rows.length); flag(set.box, "full", have === set.rows.length);
      }
      show(el.msMore, hidden > 0);
      for (const t of el.tabs) {
        t.b.setAttribute("aria-selected", String(t.id === tab));
        if (t.id === "upgrades") { show(t.badge, canCount > 0 && tab !== "upgrades"); put(t.badge, String(canCount)); }
        else { show(t.badge, true); put(t.badge, got.got + "/" + D.MILESTONES.length); t.badge.style.background = "rgba(255,255,255,.12)"; t.badge.style.color = "var(--dim)"; }
      }
      show($("upgrades"), tab === "upgrades"); show($("milestones"), tab === "milestones");

      // which pane, which views
      show($("pane-cloud"), view === "cloud"); show($("pane-sky"), view === "sky"); show($("pane-galaxy"), view === "galaxy"); show($("pane-universe"), view === "universe");
      show($("views"), L >= 1);
      const nodeCan = L >= 1 && D.NODES.some((n) => G.nodeOpen(s, n) && s.light.gte(G.nodeCost(c, n)));
      const wellCan = L >= 2 && D.WELLS.some((w) => s.wells[w.id] < w.max && s.dark.gte(G.wellCost(s, w)));
      el.views.forEach((v, i) => {
        show(v.b, i <= L);
        v.b.setAttribute("aria-pressed", String(v.id === view));
        flag(v.b, "ping", v.id !== view && ((v.id === "sky" && nodeCan) || (v.id === "galaxy" && (wellCan || G.collapseOpen(s))) || (v.id === "universe" && G.bangOpen(s))));
      });
      el.tabbar.forEach((t, i) => { show(t.b, i < 2 || i - 1 <= L); t.b.setAttribute("aria-selected", String(app.dataset.tab === t.id)); });

      if (view === "sky") paneSky(s, c, scene);
      if (view === "galaxy") paneGalaxy(s, c);
      if (view === "universe") paneUniverse(s, c);
      for (const n of el.conNames) {
        show(n.d, view === "sky");
        if (view !== "sky") continue;
        const p = scene.conAt(n.con.id), st = stageBox();
        n.d.style.left = p.x - st.left + "px"; n.d.style.top = p.y - st.top + "px";
        const rows = G.sets.constellations[n.con.id]; put(n.sm, rows.filter((x) => s.nodes[x.id]).length + " / " + rows.length);
      }

      // the big button, and the lull
      const go = $("go");
      let goOn = false;
      if ((view === "cloud" || view === "sky") && G.igniteOpen(s)) {
        const cls = G.starClass(s.cloud.mass), fresh = !s.classes[cls.id];
        goOn = true; go.style.setProperty("--c", rgb(root.AccreteSceneClass[cls.id]));
        setGo("Ignite", "+" + F.int(G.lightGain(s, c)) + " light · " + (fresh ? "a new class: " : "") + an(cls.name));
      } else if (view === "galaxy" && G.collapseOpen(s)) { goOn = true; go.style.setProperty("--c", COL.dark); setGo("Collapse", "+" + F.int(G.darkGain(s, c)) + " dark matter"); }
      else if (view === "universe" && G.bangOpen(s)) { goOn = true; go.style.setProperty("--c", COL.echo); setGo("Bang", "raise one constant of the next universe"); }
      show(go, goOn);
      const lull = $("lull"), lu = view === "cloud" && !goOn && !s.comet.live ? G.lull(s, c) : null;
      const quiet = lu && lu.secs > 75;
      show(lull, !!quiet);
      if (quiet) {
        const what = lu.offer.what === "kind" ? "a " + D.KINDS[lu.offer.k].name : lu.offer.what === "shepherd" ? "a shepherd" : D.UPGRADES.find((u) => u.id === lu.offer.id).name;
        const text = lu.secs > 900 ? "Nothing useful to do for now. Come back in about " + F.time(lu.secs) + " for " + what + "; it gathers while you are away." : "Nothing to buy for " + F.time(lu.secs) + ". Then: " + what + ".";
        put(lull, text);
      }
      const hint = $("hint");
      const wantHint = view === "cloud" && L === 0 && !s.up.spin1 && s.life.pulls < 12 && c.hand;
      show(hint, wantHint); if (wantHint) put(hint, s.life.pulls ? "Hold to keep pulling" : "Press and hold here to pull mass in");
      // the comet's tag: what it carries, shown before it is caught
      const live = s.comet.live, at = live && view === "cloud" ? scene.cometAt() : null;
      show(el.cometTag, !!at);
      if (at) {
        const st = stageBox();
        el.cometTag.style.left = Math.max(120, Math.min(st.width - 120, at.x - st.left)) + "px"; el.cometTag.style.top = at.y - st.top + "px";
        el.cometTag.style.setProperty("--c", live.type === "surge" ? "140, 215, 255" : "255, 220, 130");
        put(el.cometTag, live.type === "surge" ? "surge " + F.mult(live.mult) + " for " + Math.round(live.lo) + "–" + Math.round(live.hi) + " s" : "windfall: " + F.time(live.lo) + " to " + F.time(live.hi) + " of mass");
      }
      el.sound.innerHTML = s.set.mute ? ICON.mute : ICON.sound;
      el.sound.title = s.set.mute ? "Sound is off" : "Sound is on";
    };
    function setGo(a, b) { const go = $("go"); const key = a + "|" + b; if (go._k === key) return; go._k = key; go.textContent = ""; go.append(h("b", "", a), h("span", "", b)); }
    function stageBox() { return $("stage").getBoundingClientRect(); }
    this.cometTagMove = function (scene, s) {
      const at = s.comet.live && scene.view() === "cloud" ? scene.cometAt() : null;
      if (!at || el.cometTag.hidden) return;
      const st = stageBox(); el.cometTag.style.left = Math.max(120, Math.min(st.width - 120, at.x - st.left)) + "px"; el.cometTag.style.top = at.y - st.top + "px";
    };

    function paneSky(s, c) {
      put(el.light, F.int(s.light));
      put(el.lightSub, "light to spend · " + F.int(s.stars) + (s.stars === 1 ? " star" : " stars") + " in this galaxy · " + F.int(s.lightGal) + " light made here, each +" + F.pct(D.IGNITE.passive) + " to all bodies");
      D.IGNITE.classes.forEach((k, i) => flag(el.classChips[i], "have", !!s.classes[k.id]));
      const nc = G.nextClass(s);
      put(el.classNext, nc ? "Next: " + an(nc.name) + ", from a cloud of " + massAt(Big.from(D.IGNITE.at).mul(nc.at)) + ". Each class gives " + F.mult(D.IGNITE.classMult) + " to all bodies." : "Every class of star is in this sky.");
      show(el.watch, !!s.nodes.watch);
      if (s.nodes.watch) { el.watchSwitch.setAttribute("aria-checked", String(!!s.auto.on)); if (document.activeElement !== el.watchInput) el.watchInput.value = String(Math.round(s.auto.light)); }
      for (const con of el.cons) {
        let have = 0;
        for (const r of con.rows) {
          const n = r.n, held = !!s.nodes[n.id], open = G.nodeOpen(s, n), price = G.nodeCost(c, n);
          if (held) have++;
          flag(r.b, "have", held); flag(r.b, "shut", !held && !open); r.b.disabled = held || !open;
          const can = open && s.light.gte(price);
          flag(r.b, "can", can);
          if (open && !can) width(r.fill, s.light.div(price).toNumber()); else if (!open) width(r.fill, 0);
          put(r.d, !held && !open ? "After “" + D.NODES.find((x) => x.id === n.after).name + "”. " + n.text : n.text);
          put(r.ps, held ? "" : "light"); put(r.pv, held ? "" : F.int(price));
        }
        put(con.n, have + " / " + con.rows.length); flag(con.box, "full", have === con.rows.length);
      }
    }
    function paneGalaxy(s, c) {
      put(el.dark, F.int(s.dark));
      put(el.darkSub, "dark matter to spend · " + F.int(s.darkUni) + " made in this universe: all bodies " + F.mult(s.darkUni.add(1).pow(D.COLLAPSE.mass).toNumber()) + ", light " + F.mult(s.darkUni.add(1).pow(D.COLLAPSE.light).toNumber()));
      const open = G.collapseOpen(s);
      width(el.colBar, Math.min(1, Math.max(0, s.lightGal.log10()) / Math.log10(D.COLLAPSE.at)));
      put(el.colOf, open ? "Ready: +" + F.int(G.darkGain(s, c)) + " dark matter. One more for every tenfold of light." : F.int(s.lightGal) + " / " + F.int(D.COLLAPSE.at) + " light made in this galaxy");
      put(el.colEta, open ? "ready" : "");
      const law = s.law && D.LAWS.find((l) => l.id === s.law);
      show(el.lawNow, !!law);
      if (law) put(el.lawNow, "This galaxy is under the law “" + law.name + "”: " + law.rule + " Collapse it and: " + law.reward);
      el.wells.forEach((r) => {
        const w = r.w, lv = s.wells[w.id], full = lv >= w.max, price = G.wellCost(s, w), can = !full && s.dark.gte(price);
        flag(r.b, "have", full); flag(r.b, "can", can); r.b.disabled = full;
        put(r.t, w.name + (w.max > 1 ? " · " + lv + " / " + w.max : full ? "" : ""));
        if (!full && !can) width(r.fill, s.dark.div(price).toNumber());
        put(r.ps, full ? "" : "dark matter"); put(r.pv, full ? "" : F.int(price));
      });
      el.laws.forEach((r) => {
        const kept = !!s.laws[r.l.id];
        flag(r.b, "have", kept); flag(r.b, "shut", !kept);
        put(r.d, kept ? r.l.reward : r.l.rule + " Kept: " + r.l.reward);
        put(r.t, r.l.name + (s.law === r.l.id ? " · this galaxy" : ""));
      });
      put(el.lawSet.n, G.count(s.laws) + " / " + D.LAWS.length);
    }
    function paneUniverse(s) {
      const raised = D.CONSTANTS.filter((k) => s.consts[k.id] > 0).length;
      put(el.echo, raised + " / " + D.CONSTANTS.length);
      put(el.echoSub, "constants raised · " + F.int(s.life.bangs) + (s.life.bangs === 1 ? " universe" : " universes") + " behind you");
      const at = G.bangAt(s);
      width(el.bangBar, Math.min(1, s.darkUni.div(at).toNumber()));
      put(el.bangOf, G.bangOpen(s) ? "Ready. Everything below the constants starts again; laws, milestones and shepherds stay." : F.int(s.darkUni) + " / " + F.int(at) + " dark matter made in this universe");
      el.consts.forEach((r) => { const lv = s.consts[r.k.id]; flag(r.b, "have", false); flag(r.b, "can", lv > 0); put(r.t, r.k.sym + " · " + r.k.name); put(r.ps, "level"); put(r.pv, String(lv)); });
      show(el.endNote, s.ended);
      if (s.ended) put(el.endNote, "The game is complete: every constant has been raised. It goes on for as long as you like.");
    }

    // ── things that appear for a moment ──
    this.toast = function (title, text, col, big) {
      const box = $("toasts");
      while (box.children.length >= (window.innerWidth <= 1000 ? 1 : 3)) box.firstChild.remove();
      const t = h("div", "toast" + (big ? " big" : "")), i = h("i"), w = h("div");
      if (col) t.style.setProperty("--c", col);
      w.append(h("b", "", title)); if (text) w.append(h("span", "", text));
      t.append(i, w); box.appendChild(t);
      setTimeout(() => t.remove(), 4300);
    };
    this.float = function (x, y, text, col, big) {
      const st = stageBox(), f = h("div", "float" + (big ? " big" : ""), text);
      f.style.left = x - st.left + "px"; f.style.top = y - st.top + "px";
      if (col) f.style.setProperty("--c", col);
      $("labels").appendChild(f);
      setTimeout(() => f.remove(), big ? 1700 : 1150);
    };
    this.tab = function (id) { tab = id; };
    // the count of a kind jumps when you buy it yourself (not when a shepherd does)
    this.bought = function (k) { const o = el.body[k].own; o.classList.remove("tick"); void o.offsetWidth; o.classList.add("tick"); };

    // ── cards ──
    const veil = $("veil"), card = $("card");
    let onClose = null;
    function open(cls, build, o) {
      card.className = "card" + (cls ? " " + cls : ""); card.textContent = ""; card.removeAttribute("style");
      build(card); show(veil, true); card.scrollTop = 0;
      onClose = (o && o.onClose) || null;
      veil.onclick = (e) => { if (e.target === veil && !(o && o.modal)) close(); };
    }
    function close() { show(veil, false); const f = onClose; onClose = null; if (f) f(); }
    this.close = close;
    this.cardOpen = function () { return !veil.hidden; };
    const row = (...btns) => { const r = h("div", "row"); r.append(...btns); return r; };
    const btn = (label, fn, cls) => { const b = h("button", "btn" + (cls ? " " + cls : ""), label); b.onclick = fn; return b; };
    const para = (html) => { const p = h("p"); p.innerHTML = html; return p; };
    const list = (items) => { const u = h("ul"); for (const i of items) { const li = h("li"); li.innerHTML = i; u.appendChild(li); } return u; };

    // back after time away: what was gained first, then what was left unfinished
    this.cardAway = function (sum, s, c) {
      open("", (el2) => {
        el2.style.setProperty("--c", COL.warm);
        el2.append(h("h1", "", "While you were away"), para("for <b>" + F.time(sum.secs) + "</b> the cloud went on gathering, at the full rate."));
        el2.append(h("div", "gain fig", "+" + F.num(sum.mass)), para("mass"));
        const more = [];
        if (sum.bought) more.push("the shepherds bought <b>" + F.int(sum.bought) + "</b> bodies");
        if (sum.stars) more.push("<b>" + F.int(sum.stars) + "</b> " + (sum.stars === 1 ? "star was" : "stars were") + " lit, for <b>" + F.int(sum.light) + "</b> light");
        const ms = (sum.events || []).filter((e) => e.type === "milestone");
        if (ms.length) more.push("milestones reached: <b>" + ms.map((e) => e.name).join(", ") + "</b>");
        if (more.length) el2.append(list(more));
        const g = G.goals(s, c).filter((x) => x.frac < 1 && x.frac >= 0.1)[0];
        if (g) { el2.append(h("h3", "", "Where you left off")); el2.append(para("<b>" + g.label + "</b> is " + F.pct(g.frac) + " done" + (isFinite(g.eta) ? ", about " + F.time(g.eta) + " to go." : "."))); }
        el2.append(row(btn("Carry on", close, "hot")));
      });
    };
    // the first time a reset is offered: what goes, what stays
    this.cardReset = function (kind, s, c, done) {
      open("", (el2) => {
        if (kind === "ignite") {
          const cls = G.starClass(s.cloud.mass);
          el2.style.setProperty("--c", rgb(root.AccreteSceneClass[cls.id]));
          el2.append(h("h1", "", "Ignite the cloud"), para("The cloud has gathered enough to burn. Everything in it falls into the core and becomes a star: <b>" + an(cls.name) + "</b>."));
          el2.append(h("div", "gain fig", "+" + F.int(G.lightGain(s, c))), para("light"));
          el2.append(h("h3", "", "What goes"), list(["your mass, your bodies and the upgrades bought with mass"]));
          el2.append(h("h3", "", "What stays"), list(["the star, in your sky, and its light: each light gives <b>+" + F.pct(D.IGNITE.passive) + "</b> to all bodies, and buys stars in the constellations", "its class: each class of star gives <b>" + F.mult(D.IGNITE.classMult) + "</b> to all bodies", "your shepherds, your milestones and everything they give"]));
          el2.append(para("The next cloud grows much faster. It is your choice, now or later: the longer a cloud gathers, the more light it gives."));
          el2.append(row(btn("Not yet", close), btn("Ignite", () => { close(); done(); }, "hot")));
        } else if (kind === "collapse") {
          el2.style.setProperty("--c", COL.dark);
          el2.append(h("h1", "", "Collapse the galaxy"), para("The stars of this sky fall together. What is left is <b>dark matter</b>."));
          el2.append(h("div", "gain fig", "+" + F.int(G.darkGain(s, c))), para("dark matter"));
          el2.append(h("h3", "", "What goes"), list(["the cloud, your light, the stars in the sky and the constellations"]));
          el2.append(h("h3", "", "What stays"), list(["dark matter: all bodies and all light grow with it, and it digs wells", "your shepherds, your milestones, the laws you have kept"]));
          const open2 = D.LAWS.filter((l) => !s.laws[l.id]);
          if (open2.length && s.life.collapses >= 1) {
            el2.append(h("h3", "", "A law for the next galaxy"), para("A law bends the rules of one whole galaxy. Collapse that galaxy and its reward is yours for good."));
            lawPick = null;
            const opts = [{ id: null, name: "No law", d: "The next galaxy is an ordinary one." }].concat(open2.map((l) => ({ id: l.id, name: l.name, d: l.rule + " Kept: " + l.reward })));
            const bs = opts.map((o) => { const b = h("button", "choice"), w = h("div"); w.append(h("div", "t", o.name), h("div", "d", o.d)); b.appendChild(w); b.setAttribute("aria-pressed", String(o.id === null)); b.onclick = () => { lawPick = o.id; bs.forEach((x, i) => x.setAttribute("aria-pressed", String(opts[i].id === o.id))); }; el2.appendChild(b); return b; });
          } else lawPick = null;
          el2.append(row(btn("Not yet", close), btn("Collapse", () => { close(); done(lawPick); }, "hot")));
        } else {
          el2.style.setProperty("--c", COL.echo);
          el2.append(h("h1", "", "Bang"), para("Everything you have made folds into a point and begins again as a new universe, with one of its constants raised."));
          el2.append(h("h3", "", "What goes"), list(["the cloud, the sky, the galaxy, your dark matter and its wells"]));
          el2.append(h("h3", "", "What stays"), list(["the constants, the laws you have kept, your shepherds and your milestones"]));
          el2.append(h("h3", "", "Raise one constant"));
          constPick = (D.CONSTANTS.find((k) => !s.consts[k.id]) || D.CONSTANTS[0]).id;
          const bs = D.CONSTANTS.map((k) => { const b = h("button", "choice"), w = h("div"); w.append(h("div", "t", k.name + (s.consts[k.id] ? " · level " + s.consts[k.id] : "")), h("div", "d", k.text)); b.append(h("div", "k", k.sym), w); b.setAttribute("aria-pressed", String(k.id === constPick)); b.onclick = () => { constPick = k.id; bs.forEach((x, i) => x.setAttribute("aria-pressed", String(D.CONSTANTS[i].id === k.id))); }; el2.appendChild(b); return b; });
          if (!s.ended) el2.append(para("Raise every constant once and the game is complete."));
          el2.append(row(btn("Not yet", close), btn("Bang", () => { close(); done(constPick); }, "hot")));
        }
      });
    };
    this.cardEnd = function (s) {
      open("", (el2) => {
        el2.style.setProperty("--c", COL.echo);
        el2.append(h("h1", "", "The game is complete"), para("Every constant of the universe has been raised. There is nothing further that the game holds back."));
        el2.append(h("h3", "", "What you made"));
        el2.append(list(["<b>" + F.num(s.life.mass) + "</b> mass gathered", "<b>" + F.int(s.life.ignitions) + "</b> stars lit", "<b>" + F.int(s.life.collapses) + "</b> galaxies, <b>" + F.int(s.life.bangs) + "</b> universes", "<b>" + G.msCounts(s).got + "</b> of " + D.MILESTONES.length + " milestones", "<b>" + F.time(s.life.secs) + "</b> with the page open, <b>" + F.time(s.life.away) + "</b> away"]));
        el2.append(para("It goes on for as long as you like: the constants can be raised further. Or this is a good place to stop."));
        el2.append(row(btn("Close", close, "hot")));
      });
    };
    function cardWhy() {
      open("wide", (el2) => {
        el2.append(h("h1", "", "Why this game hooks"), para("You asked for a game that is proved addicting, and built on theories where there is no proof. This is the list: every hook in the game, how well it is supported, and where that comes from."));
        el2.append(para("<b>The short of it:</b> two things are proven in games by experiments that held up. The rest is supported from other settings, is theory, or is what the longest-played games do. Nothing has been tested in an idle game. Whether this one hooks you, only you can tell."));
        let last = "";
        for (const w of WHY.list) {
          if (w.g !== last) { last = w.g; el2.append(h("h3", "", WHY.grades[w.g])); }
          const d = h("div", "why"), g = h("span", "g " + w.g, w.g === "habit" ? "habit" : w.g === "out" ? "left out" : w.g);
          const t = h("p"); t.append(g, h("b", "", w.hook));
          const a = h("a", "", w.url.replace(/^https?:\/\//, "").slice(0, 70)); a.href = w.url; a.target = "_blank"; a.rel = "noopener noreferrer";
          const src = h("p", "", w.src + " "); src.appendChild(a);
          d.append(t, h("p", "", "In the game: " + w.game), src);
          el2.appendChild(d);
        }
        el2.append(row(btn("Close", close, "hot")));
      });
    }
    this.cardWhy = cardWhy;
    function cardSettings() {
      const s = act.state();
      open("", (el2) => {
        el2.append(h("h1", "", "Settings"));
        const field = (label, ctl) => { const f = h("div", "field"), l = h("label", "", label); f.append(l, ctl); return f; };
        const seg = (opts, cur, fn) => { const sg = h("div", "seg"); sg.style.margin = "0"; sg.style.minWidth = "210px"; const bs = opts.map(([v, lab]) => { const b = h("button", "", lab); b.setAttribute("aria-pressed", String(v === cur)); b.onclick = () => { fn(v); bs.forEach((x, i) => x.setAttribute("aria-pressed", String(opts[i][0] === v))); }; sg.appendChild(b); return b; }); return sg; };
        el2.append(h("h3", "", "Numbers"), field("Big numbers: names to a trillion then powers of ten, names all the way, or powers only", seg([["names", "1.2T · 3e15"], ["long", "1.2T · 3Qa"], ["sci", "1.2e12"]], s.set.notation, (v) => act.setting("notation", v))));
        el2.append(h("h3", "", "Sound and motion"));
        const vol = h("input"); vol.type = "range"; vol.min = "0"; vol.max = "1"; vol.step = "0.05"; vol.value = String(s.set.volume); vol.style.width = "210px"; vol.oninput = () => act.setting("volume", Number(vol.value));
        el2.append(field("Sound level", vol));
        const mute = h("button", "switch"); mute.setAttribute("role", "switch"); mute.setAttribute("aria-checked", String(!s.set.mute)); mute.onclick = () => { act.mute(); mute.setAttribute("aria-checked", String(!act.state().set.mute)); };
        el2.append(field("Sound", mute));
        el2.append(field("Motion and flashes", seg([[1, "Full"], [0, "Less"]], s.set.motion, (v) => act.setting("motion", v))));
        el2.append(h("h3", "", "Your game"), para("The game is kept in this browser only, saved every few seconds. If the browser's data for this page is cleared, it is gone: keep a copy if you care about it."));
        const ta = h("textarea"); ta.placeholder = "A saved game, as text"; ta.spellcheck = false;
        const msg = h("p");
        el2.append(ta, row(btn("Copy my game as text", () => { ta.value = act.exportSave(); ta.select(); try { navigator.clipboard.writeText(ta.value); msg.textContent = "Copied. Keep it somewhere."; } catch (e) { msg.textContent = "Select the text and copy it."; } }), btn("Load the text above", () => { const r = act.importSave(ta.value); msg.textContent = r === true ? "" : r; if (r === true) close(); })), msg);
        el2.append(h("h3", "", "Start over"), para("This wipes everything, milestones too. It cannot be undone."));
        const sure = h("input"); sure.type = "text"; sure.placeholder = "type: wipe"; sure.style.width = "120px";
        const wipe = btn("Wipe my game", () => { if (sure.value.trim().toLowerCase() === "wipe") { close(); act.wipe(); } else { sure.focus(); } }, "bad");
        const r = row(sure, wipe); r.style.justifyContent = "flex-start"; el2.append(r);
        el2.append(h("h3", "", "Played"), para("<b>" + F.time(s.life.secs) + "</b> with the page open · <b>" + F.time(s.life.away) + "</b> away · <b>" + F.num(s.life.mass) + "</b> mass in all · <b>" + F.int(s.life.ignitions) + "</b> stars"));
        el2.append(row(btn("Close", close, "hot")));
      });
    }
    this.cardSettings = cardSettings;
  }
  root.AccreteUI = UI;
})(typeof self !== "undefined" ? self : this);
