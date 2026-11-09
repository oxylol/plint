"""One contact's surface events of a macro step -> patch, pass and line updates (docs/models/surface.md
section 7).

Per event (bus SURFACE_EVENT_COLUMNS: X in the material configuration, outward normal, tick, tooth,
element, h, edge radius, VB, u_rel at generation):

1. the tool at the event tick from motion's PathFrame (tip, axis, spindle speed, feed relative to the work);
2. the generation class: ``turn`` (non-rotating tool: the work turns), ``floor`` (rotating tool, normal within
   45 deg of the tool axis), ``bore`` (rotating tool fed along its axis: hole walls), ``wall`` (the rest);
3. the face patch (brick, facet), the pass (a gap longer than ``pass_gap`` opens a new one) and the profile
   line: walls one line per station (all teeth sweep the same circle there), floors one centre line per pass,
   turned faces and hole walls one line per angular bin; at most ``lines_per_pass`` lines per pass;
4. the generating pass row of the line (cnctwin.surface.lines.ROW_COLUMNS): position in the line frame, the
   frame of the outline's body (tool axis and the element's radial direction; for turning the insert's
   x_T, z_T), the cutting velocity, the normal, the path curvature with the trochoid feed correction
   (Martellotti 1941: kappa = sin(kappa_e) / r / (1 + 2 Vf.t / (r w) + (Vf / (r w))^2)), the minimum chip
   thickness, the built-up edge overcut (counter-based draws, stream ``surface.bue``), welded fragments
   (``surface.bue_fragment``) and recut dents.

Floors: an end element writes its event where it passes the trailing side of the feed; when the tool axis
is tilted so that the leading side lies lower, the leading passage generates the floor: a mirrored twin row
half a revolution earlier is added (crossed marks when both sides cut).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core.rng import hash_uniform, stream_key
from . import abi as A
from . import kinematics as KIN
from .lines import CLASSES, N_ROW, PARABOLA_SIN, R, R_BASIS, LineBuffer, cylinder_basis
from .store import PF, PI, QF, QI, Store

CLS_CODE = {name: i for i, name in enumerate(CLASSES)}
FLOOR_COS = math.cos(math.radians(45.0))
BORE_COS = 0.9
STREAM_BUE = stream_key("surface.bue")
STREAM_FRAGMENT = stream_key("surface.bue_fragment")
STREAM_FRAGMENT_H = stream_key("surface.bue_fragment_height")


@dataclass
class StepInput:
    rows: np.ndarray                 # [n, 16] surface events of the contact
    slot: int
    contact: int
    frame: object | None             # PathFrame of the contact's path
    workpiece: int
    outline: object                  # protocols.ToolOutline
    edge: object                     # core.abi.EdgeModel (nominal)
    wear: object | None              # bus ToolWear of the slot (prev)
    recut: np.ndarray                # [m, 6] recut rows of the contact
    h_max_m: float                   # largest uncut chip thickness of the step
    chip_ratio: float                # mean chip thickness ratio h / h_c of the slot's coefficients (0: unknown)
    op: int


@dataclass
class StepOutput:
    patches: list                    # patches touched (first-appearance order)
    events: int = 0
    profiled: int = 0
    unprofiled: int = 0
    rejected: int = 0
    last_event: dict | None = None   # {X, n, u_n, patch, tick, block}
    arrays: tuple | None = None      # (tick, X, N, u_n, patch) of the valid events (mark detectors)


class Gatherer:
    """Holds the per-run constants the ingestion needs (settings, model params, seed, clock, bricks)."""

    def __init__(self, settings, rough_params: dict, material, seed: int, tick_s: float, bricks, close_line):
        self.st = settings
        self.p = rough_params
        self.mat = material
        self.seed = int(seed)
        self.tick_s = float(tick_s)
        self.bricks = bricks
        self.close_line = close_line          # callback(line) when a line is full or its pass closes
        self.cos_face = math.cos(settings.face_angle_rad)

    # ------------------------------------------------------------------ entry
    def ingest(self, store: Store, inp: StepInput) -> StepOutput:
        rows = np.asarray(inp.rows, dtype=float).reshape(-1, 16)
        ok = np.all(np.isfinite(rows), axis=1)
        nrm = np.linalg.norm(rows[:, 6:9], axis=1)
        ok &= nrm > 0.0
        out = StepOutput(patches=[], rejected=int(np.count_nonzero(~ok)))
        rows = rows[ok]
        if len(rows) == 0:
            return out
        out.events = len(rows)
        X = rows[:, 3:6].copy()
        N = rows[:, 6:9] / np.linalg.norm(rows[:, 6:9], axis=1)[:, None]
        tick = rows[:, 0].astype(np.int64)
        tooth = rows[:, 1].astype(np.int64)
        elem = rows[:, 2].astype(np.int64)
        u_n = np.einsum("ij,ij->i", rows[:, 12:15], N)
        mot = KIN.sample(inp.frame, tick, self.tick_s) if inp.frame is not None else {}
        rotating = bool(inp.outline.rotating)
        cls = self._classes(N, mot, rotating, len(rows))
        patch = self._patches(store, inp, X, N, tick, mot, cls)
        out.patches = list(dict.fromkeys(int(p) for p in patch))
        self._patch_sums(store, inp, patch, X, N, tick, mot, cls)
        lift = self._lowest_point_lift(inp, X, N, tooth, elem, mot) if (mot and rotating) else np.zeros(len(X))
        hmin = self.mat.min_chip_ratio * np.maximum(rows[:, 10], 0.0)
        passes = self._passes(store, inp, patch, X, N, tick, mot, cls, u_n, lift, hmin)
        last = int(np.argmax(tick))
        out.arrays = (tick.copy(), X.copy(), N.copy(), u_n.copy(), patch.copy())
        out.last_event = {"X": X[last].tolist(), "n": N[last].tolist(), "u_n": float(u_n[last]),
                          "patch": int(patch[last]), "tick": int(tick[last]),
                          "block": int(mot["block"][last]) if mot else -1,
                          "omega": float(mot["omega"][last]) if mot else 0.0}
        if not mot:
            out.unprofiled = len(rows)
            return out
        profiled = self._lines(store, inp, rows, X, N, tick, tooth, elem, u_n, mot, cls, patch, passes, lift)
        out.profiled = profiled
        out.unprofiled = len(rows) - profiled
        return out

    # ------------------------------------------------------------------ classification
    def _classes(self, N, mot, rotating, n) -> np.ndarray:
        if not rotating:
            return np.full(n, CLS_CODE["turn"], np.int64)
        if not mot:
            return np.full(n, CLS_CODE["wall"], np.int64)
        a = mot["axis"]
        axial = np.abs(np.einsum("ij,ij->i", N, a)) > FLOOR_COS
        v = mot["vel"]
        speed = np.linalg.norm(v, axis=1)
        along = np.abs(np.einsum("ij,ij->i", v, a)) > BORE_COS * np.where(speed > 0.0, speed, np.inf)
        cls = np.full(n, CLS_CODE["wall"], np.int64)
        cls[along & ~axial] = CLS_CODE["bore"]
        cls[axial] = CLS_CODE["floor"]
        return cls

    # ------------------------------------------------------------------ patches
    def _patches(self, store, inp, X, N, tick, mot, cls) -> np.ndarray:
        """Face patch of every event: per brick, the first facet (creation order) of the running operation whose
        normal and plane match; events matching none open a facet from the first of them, and the rest are
        matched again. A patch belongs to the operation that opened it: the next part of a batch (a new blank in
        the same place) and the next setup (another stock frame) open their own patches (docs/models/surface.md
        3; before, a batch's second part joined the first part's patches and its step-over profiles compared
        the two parts' passes as overlapping passes: witness marks up to 143 um deep)."""
        wp = int(inp.workpiece)
        op = int(inp.op)
        bricks = np.asarray(self.bricks.brick_of(wp, X), dtype=np.int64)
        patch = np.full(len(X), -1, np.int64)
        _, first = np.unique(bricks, return_index=True)
        for b in bricks[np.sort(first)]:
            todo = np.nonzero(bricks == b)[0]
            while len(todo):
                facets = [q for q in store.facets.get((wp, int(b)), []) if int(store.pi[q, PI["op"]]) == op]
                hit = np.full(len(todo), -1, np.int64)
                for p in facets:
                    n_p = store.normal(p)
                    o_p = store.origin(p)
                    free = hit < 0
                    ok = free & (N[todo] @ n_p >= self.cos_face) & \
                        (np.abs((X[todo] - o_p[None, :]) @ n_p) <= self.st.plane_tol_m)
                    hit[ok] = p
                patch[todo[hit >= 0]] = hit[hit >= 0]
                todo = todo[hit < 0]
                if not len(todo):
                    break
                i = int(todo[0])
                if len(facets) >= self.st.max_facets:
                    best = np.array([max(facets, key=lambda q: float(store.normal(q) @ N[j])) for j in todo])
                    patch[todo] = best
                    break
                store.new_patch(wp, int(b), X[i], N[i], int(inp.op))
        # keep the facet normals current so later events of a curved face join consistently
        np.add.at(store.pf[:, PF["nsx"]], patch, N[:, 0])
        np.add.at(store.pf[:, PF["nsy"]], patch, N[:, 1])
        np.add.at(store.pf[:, PF["nsz"]], patch, N[:, 2])
        return patch

    def _patch_sums(self, store, inp, patch, X, N, tick, mot, cls) -> None:
        pf, pi = store.pf, store.pi
        np.add.at(pf[:, PF["xsx"]], patch, X[:, 0])
        np.add.at(pf[:, PF["xsy"]], patch, X[:, 1])
        np.add.at(pf[:, PF["xsz"]], patch, X[:, 2])
        np.add.at(pf[:, PF["count"]], patch, 1.0)
        blocks = mot["block"] if mot else np.full(len(X), -1, np.int64)
        for p in dict.fromkeys(int(q) for q in patch):
            sel = patch == p
            t0, t1 = int(np.min(tick[sel])), int(np.max(tick[sel]))
            if pi[p, PI["first_tick"]] < 0:
                pi[p, PI["first_tick"]] = t0
                pi[p, PI["block_first"]] = int(blocks[sel][np.argmin(tick[sel])])
            if t1 >= pi[p, PI["last_tick"]]:
                pi[p, PI["last_tick"]] = t1
                pi[p, PI["block_last"]] = int(blocks[sel][np.argmax(tick[sel])])
                pi[p, PI["slot"]] = int(inp.slot)
                pi[p, PI["cls"]] = int(np.bincount(cls[sel], minlength=4).argmax())
                pi[p, PI["op"]] = int(inp.op)

    # ------------------------------------------------------------------ passes
    def _gap_ticks(self, omega: np.ndarray) -> int:
        w = float(np.mean(np.abs(omega))) if len(omega) else 0.0
        rev = 2.0 * math.pi / w / self.tick_s if w > 0.0 else 0.0
        return int(max(self.st.pass_gap_s / self.tick_s, 4.0 * rev))

    def _lowest_point_lift(self, inp, X, N, tooth, elem, mot) -> np.ndarray:
        """Height of each generating element above the tool's lowest point along the event normal (rotating
        tools): r_e (r_hat . n) + z_e (a . n) - min over the outline of (z (a . n) - r sqrt(1 - (a . n)^2)),
        so a pass's generated height is that of its lowest point whatever element wrote the event."""
        a = mot["axis"]
        rel = X - mot["tip"]
        ax = np.einsum("ij,ij->i", rel, a)
        radial = rel - ax[:, None] * a
        r_act = np.linalg.norm(radial, axis=1)
        r_hat = radial / np.where(r_act > 0.0, r_act, 1.0)[:, None]
        r_e, z_e = self._element_rz(inp, tooth, elem)
        cosb = np.clip(np.einsum("ij,ij->i", a, N), -1.0, 1.0)
        pts = np.asarray(inp.outline.pts, dtype=float)
        keys, inverse = np.unique(np.round(cosb, 6), return_inverse=True)
        sb = np.sqrt(np.maximum(0.0, 1.0 - keys * keys))
        lowest = np.array([float(np.min(pts[:, 1] * c - pts[:, 0] * s_)) for c, s_ in zip(keys, sb)])
        return r_e * np.einsum("ij,ij->i", r_hat, N) + z_e * cosb - lowest[inverse.reshape(-1)]

    def _passes(self, store, inp, patch, X, N, tick, mot, cls, u_n, lift, hmin=None) -> np.ndarray:
        """The pass of every event (per patch and tool: a gap longer than the hold opens a new one) and the pass
        sums: lowest generated height (corrected to the tool's lowest point), mean height, deflection, minimum chip
        thickness of the generating elements (for the step-over profile), tool tip, feed direction, tool axis,
        ticks and blocks."""
        out = np.full(len(X), -1, np.int64)
        omega = mot["omega"] if mot else np.zeros(len(X))
        blocks = mot["block"] if mot else np.full(len(X), -1, np.int64)
        uniq, first, inv = np.unique(patch, return_index=True, return_inverse=True)
        inv = inv.reshape(-1)
        k = len(uniq)
        t_first = np.full(k, np.iinfo(np.int64).max, np.int64)
        np.minimum.at(t_first, inv, tick)
        t_last = np.full(k, np.iinfo(np.int64).min, np.int64)
        np.maximum.at(t_last, inv, tick)
        w_abs = np.zeros(k)
        np.add.at(w_abs, inv, np.abs(omega))
        counts = np.bincount(inv, minlength=k).astype(float)
        w_mean = w_abs / counts
        # heights in each patch's frame (origin, current normal)
        nsum = store.pf[uniq][:, PF["nsx"]:PF["nsz"] + 1]
        nrm = nsum / np.maximum(np.linalg.norm(nsum, axis=1), 1e-300)[:, None]
        org = store.pf[uniq][:, PF["ox"]:PF["oz"] + 1]
        w = np.einsum("ij,ij->i", X - org[inv], nrm[inv]) - lift
        last_of = np.zeros(k, np.int64)
        order = np.argsort(tick, kind="stable")
        last_of[inv[order]] = order                        # index of the latest event of each patch
        first_of = np.full(k, -1, np.int64)
        for i in order[::-1]:
            first_of[inv[i]] = i
        q_of = np.zeros(k, np.int64)
        for j in np.argsort(first, kind="stable"):
            p = int(uniq[j])
            rev = 2.0 * math.pi / w_mean[j] / self.tick_s if w_mean[j] > 0.0 else 0.0
            gap = int(max(self.st.pass_gap_s / self.tick_s, 4.0 * rev))
            q = store.active.get((p, int(inp.slot)), -1)
            if q >= 0 and (store.qi[q, QI["closed"]] != 0 or int(t_first[j]) - int(store.qi[q, QI["tick1"]]) > gap):
                self.close_pass(store, q)
                q = -1
            if q < 0:
                sel = inv == j
                code = int(np.bincount(cls[sel], minlength=4).argmax())
                q = store.new_pass(p, int(inp.slot), code, int(t_first[j]), int(blocks[first_of[j]]), int(inp.op))
            q_of[j] = q
        qf, qi = store.qf, store.qi
        qe = q_of[inv]
        out[:] = qe
        np.minimum.at(qf[:, QF["w_min"]], qe, w)
        np.add.at(qf[:, QF["w_sum"]], qe, w)
        np.add.at(qf[:, QF["un_sum"]], qe, u_n)
        np.add.at(qf[:, QF["count"]], qe, 1.0)
        if hmin is not None:
            np.add.at(qf[:, QF["hmin_sum"]], qe, hmin)
        if mot:
            for c in range(3):
                np.add.at(qf[:, QF["csx"] + c], qe, mot["tip"][:, c])
            v = mot["vel"]
            speed = np.linalg.norm(v, axis=1)
            unit = v / np.where(speed > 0.0, speed, 1.0)[:, None]
            for c in range(3):
                np.add.at(qf[:, QF["dsx"] + c], qe, unit[:, c])
            qf[q_of, QF["ax"]:QF["az"] + 1] = mot["axis"][last_of]
        qi[q_of, QI["tick1"]] = np.maximum(qi[q_of, QI["tick1"]], t_last)
        qi[q_of, QI["block_last"]] = blocks[last_of]
        return out

    def close_line_key(self, store: Store, key) -> None:
        """Evaluate and drop one open line (its pass stays open)."""
        line = store.lines.pop(key, None)
        if line is not None:
            self.close_line(line)

    def close_pass(self, store: Store, q: int) -> None:
        """Close pass q and evaluate its open lines."""
        store.qi[q, QI["closed"]] = 1
        p = int(store.qi[q, QI["patch"]])
        for key in sorted(k for k in store.lines if k[0] == p and k[1] == q):
            line = store.lines.pop(key)
            self.close_line(line)
        slot = int(store.qi[q, QI["slot"]])
        if store.active.get((p, slot)) == q:
            del store.active[(p, slot)]

    # ------------------------------------------------------------------ lines
    def _sub_keys(self, inp, X, N, tooth, elem, mot, cls) -> np.ndarray:
        """Line key within a patch and pass: walls the station, floors 0, turned faces and hole walls the angular
        bin about the work or tool axis (bin edges half a bin off the event angles, so events sampled at the bin
        spacing never alternate between neighbours)."""
        sub = np.zeros(len(X), np.int64)
        bins = int(self.st.turn_bins)
        wall = cls == CLS_CODE["wall"]
        if wall.any():
            station = np.asarray(inp.edge.elem_station, dtype=np.int64)
            nt, ne = station.shape
            j, k = tooth[wall], elem[wall]
            ok = (j >= 0) & (j < nt) & (k >= 0) & (k < ne)
            sub[wall] = np.where(ok, station[np.clip(j, 0, nt - 1), np.clip(k, 0, ne - 1)], k)
        turn = cls == CLS_CODE["turn"]
        if turn.any():
            theta = np.mod(np.arctan2(X[turn, 1], X[turn, 0]), 2.0 * math.pi)
            sub[turn] = np.mod(np.floor(theta / (2.0 * math.pi) * bins + 0.5).astype(np.int64), bins)
        bore = cls == CLS_CODE["bore"]
        if bore.any():
            a = mot["axis"][bore]
            rel = X[bore] - mot["tip"][bore]
            radial = rel - np.einsum("ij,ij->i", rel, a)[:, None] * a
            ref = np.where(np.abs(a[:, 2:3]) < 0.9, np.array([[0.0, 0.0, 1.0]]), np.array([[1.0, 0.0, 0.0]]))
            e1 = np.cross(a, ref)
            e1 /= np.linalg.norm(e1, axis=1)[:, None]
            e2 = np.cross(a, e1)
            theta = np.mod(np.arctan2(np.einsum("ij,ij->i", radial, e2), np.einsum("ij,ij->i", radial, e1)),
                           2.0 * math.pi)
            sub[bore] = np.mod(np.floor(theta / (2.0 * math.pi) * bins + 0.5).astype(np.int64), bins)
        return sub

    def _new_line(self, store, inp, patch, q, sub, i, X, N, mot, cls) -> LineBuffer | None:
        c = int(cls[i])
        name = CLASSES[c]
        if name in ("wall", "floor"):
            v = mot["vel"][i]
            f = v - (v @ N[i]) * N[i]
            norm = float(np.linalg.norm(f))
            if norm <= 1e-12:
                return None
            d = f / norm
            m = np.cross(d, N[i])
            return LineBuffer(patch=patch, pass_id=q, sub=sub, slot=int(inp.slot), cls=name, cylindrical=False,
                              origin=X[i].copy(), d=d, n=N[i].copy(), axis=m, op=int(inp.op))
        if name == "turn":
            axis_point = np.zeros(3)
            axis_dir = np.array([0.0, 0.0, 1.0])
            vr, vz = float(mot["v_rho"][i]), float(mot["v_z"][i])
        else:
            axis_point = mot["tip"][i].copy()
            axis_dir = mot["axis"][i].copy()
            vel = mot["vel"][i]
            vr, vz = 0.0, float(vel @ axis_dir)
        rho, z, rho_hat, _ = cylinder_basis(X[i:i + 1], axis_point, axis_dir)
        n2 = np.array([float(N[i] @ rho_hat[0]), float(N[i] @ axis_dir)])
        if np.linalg.norm(n2) <= 1e-12:
            return None
        n2 /= np.linalg.norm(n2)
        f2 = np.array([vr, vz])
        f2 -= (f2 @ n2) * n2
        if np.linalg.norm(f2) <= 1e-15:
            return None
        d2 = f2 / np.linalg.norm(f2)
        return LineBuffer(patch=patch, pass_id=q, sub=sub, slot=int(inp.slot), cls=name, cylindrical=True,
                          origin=axis_point, d=d2, n=n2, axis=axis_dir, rho0=float(rho[0]), z0=float(z[0]),
                          op=int(inp.op))

    def _lines(self, store, inp, rows, X, N, tick, tooth, elem, u_n, mot, cls, patch, passes, lift) -> int:
        """Assign the events to profile lines (opening lines within the per-pass budget), build every accepted
        event's pass row in one vectorised computation with its line's frame, and append them per line."""
        sub = self._sub_keys(inp, X, N, tooth, elem, mot, cls)
        keys = np.stack([patch, passes, sub], axis=1)
        uniq, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
        inverse = inverse.reshape(-1)
        accepted = []                                   # (key, line, event indices)
        profiled = 0
        for u in np.argsort(first, kind="stable"):
            key = (int(uniq[u, 0]), int(uniq[u, 1]), int(uniq[u, 2]))
            idx = np.nonzero(inverse == u)[0]
            line = store.lines.get(key)
            if line is None:
                q = key[1]
                if store.qi[q, QI["lines"]] >= self.st.lines_per_pass:
                    continue
                i0 = int(idx[0])
                line = self._new_line(store, inp, key[0], q, key[2], i0, X, N, mot, cls)
                if line is None:
                    continue
                w = abs(float(mot["omega"][i0]))
                line.hold_ticks = 4.0 * 2.0 * math.pi / w / self.tick_s if w > 0.0 else 0.0
                store.lines[key] = line
                store.qi[q, QI["lines"]] += 1
            profiled += len(idx)
            if line.cls == "floor":
                idx = self._lowest_of_each_pass(idx, tick, tooth, lift)
            accepted.append((key, line, idx))
        if not accepted:
            return profiled
        sel = np.concatenate([idx for _, _, idx in accepted])
        owner = np.concatenate([np.full(len(idx), k, np.int64) for k, (_, _, idx) in enumerate(accepted)])
        frames = self._frames([line for _, line, _ in accepted], owner)
        base = self._rows_batch(inp, frames, rows[sel], X[sel], N[sel], tick[sel], tooth[sel], elem[sel], u_n[sel],
                                {k: v[sel] for k, v in mot.items()})
        recut = self._recut_targets(inp, tick, tooth, keys, inverse, uniq)
        for k, (key, line, idx) in enumerate(accepted):
            mine = owner == k
            new_rows = base["rows"][mine]
            extra = [new_rows]
            if line.cls == "floor" and inp.outline.rotating:
                extra.append(self._lead_twins(line, new_rows, base["r_act"][mine], base["r_hat"][mine],
                                              mot["omega"][sel][mine]))
            frag = self._fragments(inp, new_rows, tick[sel][mine], tooth[sel][mine], elem[sel][mine])
            if len(frag):
                extra.append(frag)
            new_rows = np.concatenate(extra) if len(extra) > 1 else new_rows
            dents = recut.get(key)
            if dents:
                new_rows = np.concatenate([new_rows, self._dent_rows(store, inp, line, dents, new_rows, idx)])
            if mot["block"].size:
                blk = mot["block"][idx]
                if line.block_first < 0:
                    line.block_first = int(blk[0])
                line.block_last = int(blk[-1])
            order_rows = np.argsort(new_rows[:, R["tick"]], kind="stable")
            self._append(store, line, key, new_rows[order_rows])
        return profiled

    @staticmethod
    def _frames(lines: list, owner: np.ndarray) -> dict:
        """Per-event line frames: origin / axis point O, d and n (cylindrical: (rho, z, 0)), the third vector
        (cartesian m, cylindrical the axis), the cylindrical flag and (rho0, z0)."""
        O = np.array([ln.origin for ln in lines])
        D = np.array([np.append(ln.d, 0.0) if ln.cylindrical else ln.d for ln in lines])
        Nl = np.array([np.append(ln.n, 0.0) if ln.cylindrical else ln.n for ln in lines])
        M = np.array([ln.axis for ln in lines])
        cyl = np.array([ln.cylindrical for ln in lines])
        rz = np.array([[ln.rho0, ln.z0] for ln in lines])
        return {"O": O[owner], "D": D[owner], "N": Nl[owner], "M": M[owner], "cyl": cyl[owner], "rz": rz[owner]}

    @staticmethod
    def _lowest_of_each_pass(idx, tick, tooth, lift) -> np.ndarray:
        """Floors: the elements of one tooth pass are one rigid edge in one pose, so the event nearest the tool's
        lowest point carries the generating outline for all of them (its template window covers the rest)."""
        groups = {}
        for i in idx:
            g = (int(tick[i]), int(tooth[i]))
            if g not in groups or lift[i] < lift[groups[g]]:
                groups[g] = int(i)
        return np.array(sorted(groups.values()), dtype=np.int64)

    def _append(self, store, line, key, rows) -> None:
        cap = int(self.st.floor_line_rows if line.cls == "floor" else self.st.line_rows)
        rest = line.append(rows, cap)
        while len(rest):
            full = store.lines.pop(key)
            self.close_line(full)
            fresh = LineBuffer(patch=full.patch, pass_id=full.pass_id, sub=full.sub, slot=full.slot, cls=full.cls,
                               cylindrical=full.cylindrical, origin=full.origin.copy(), d=full.d.copy(),
                               n=full.n.copy(), axis=full.axis.copy(), rho0=full.rho0, z0=full.z0, op=full.op,
                               block_first=full.block_last, block_last=full.block_last, hold_ticks=full.hold_ticks)
            store.lines[key] = fresh
            rest = fresh.append(rest, cap)
            line = fresh

    # ------------------------------------------------------------------ rows
    @staticmethod
    def _line_vectors(V, fr, rho_hat, theta_hat) -> np.ndarray:
        """[n, 3] line components (d, n, m) of S vectors V at the events (per-event frames)."""
        cart = np.stack([np.einsum("ij,ij->i", V, fr["D"]), np.einsum("ij,ij->i", V, fr["N"]),
                         np.einsum("ij,ij->i", V, fr["M"])], axis=1)
        if not fr["cyl"].any():
            return cart
        v_rho = np.einsum("ij,ij->i", V, rho_hat)
        v_z = np.einsum("ij,ij->i", V, fr["M"])
        v_th = np.einsum("ij,ij->i", V, theta_hat)
        cyl = np.stack([v_rho * fr["D"][:, 0] + v_z * fr["D"][:, 1], v_rho * fr["N"][:, 0] + v_z * fr["N"][:, 1],
                        v_th], axis=1)
        return np.where(fr["cyl"][:, None], cyl, cart)

    def _rows_batch(self, inp, fr, ev, X, N, tick, tooth, elem, u_n, mot) -> dict:
        n = len(X)
        rows = np.zeros((n, N_ROW))
        rows[:, R["tick"]] = tick
        rows[:, R["tooth"]] = tooth
        rows[:, R["elem"]] = elem
        rows[:, R["u_n"]] = u_n
        rows[:, R["VB"]] = ev[:, 11]
        rows[:, R["edge_r"]] = ev[:, 10]
        rows[:, R["kind"]] = A.KIND_CUT
        rows[:, R["hmin"]] = self.mat.min_chip_ratio * np.maximum(ev[:, 10], 0.0)
        r_e, z_e = self._element_rz(inp, tooth, elem)
        rows[:, R["r_e"]] = r_e
        rows[:, R["z_e"]] = z_e
        # coordinates in the line frames (cylindrical lines about their axis, cartesian lines from their origin)
        rel = X - fr["O"]
        z_ax = np.einsum("ij,ij->i", rel, fr["M"])
        radial = rel - z_ax[:, None] * fr["M"]
        rho = np.linalg.norm(radial, axis=1)
        rho_hat = radial / np.where(rho > 0.0, rho, 1.0)[:, None]
        theta_hat = np.cross(fr["M"], rho_hat)
        drho, dz = rho - fr["rz"][:, 0], z_ax - fr["rz"][:, 1]
        cyl = fr["cyl"]
        rows[:, R["s"]] = np.where(cyl, drho * fr["D"][:, 0] + dz * fr["D"][:, 1], np.einsum("ij,ij->i", rel, fr["D"]))
        rows[:, R["w"]] = np.where(cyl, drho * fr["N"][:, 0] + dz * fr["N"][:, 1], np.einsum("ij,ij->i", rel, fr["N"]))
        rows[:, R["m"]] = np.where(cyl, 0.0, z_ax)
        r_act = np.zeros(n)
        r_hat = np.zeros((n, 3))
        if inp.outline.rotating:
            a = mot["axis"]
            rel_t = X - mot["tip"]
            ax = np.einsum("ij,ij->i", rel_t, a)
            radial_t = rel_t - ax[:, None] * a
            r_act = np.linalg.norm(radial_t, axis=1)
            r_hat = radial_t / np.where(r_act > 0.0, r_act, 1.0)[:, None]
            v_hat = np.cross(a, r_hat)
            omega = mot["omega"]
            t_hat = np.sign(omega)[:, None] * v_hat
            rw = r_act * np.abs(omega)
            safe = np.where(rw > 0.0, rw, np.inf)
            ratio = np.einsum("ij,ij->i", mot["vel"], t_hat) / safe
            q2 = np.einsum("ij,ij->i", mot["vel"], mot["vel"]) / (safe * safe)
            corr = 1.0 / np.maximum(1.0 + 2.0 * ratio + q2, 1e-6)
            curv = -np.einsum("ij,ij->i", N, r_hat) / np.where(r_act > 0.0, r_act, np.inf) * corr
            body_r, body_a = r_hat, a
        else:
            x_c, z_c = KIN.turning_tool_axes(inp.frame)
            ez = np.array([0.0, 0.0, 1.0])
            rel_w = X.copy()
            rel_w[:, 2] = 0.0
            rw_ = np.linalg.norm(rel_w, axis=1)
            rh = rel_w / np.where(rw_ > 0.0, rw_, 1.0)[:, None]
            th = np.cross(ez[None, :], rh)
            body_r = x_c[0] * rh + x_c[1] * ez[None, :] + x_c[2] * th
            body_a = z_c[0] * rh + z_c[1] * ez[None, :] + z_c[2] * th
            v_hat = th
            curv = np.zeros(n)
        rows[:, R["curv"]] = curv
        rows[:, R_BASIS:R_BASIS + 3] = self._line_vectors(body_r, fr, rho_hat, theta_hat)
        rows[:, R_BASIS + 3:R_BASIS + 6] = self._line_vectors(body_a, fr, rho_hat, theta_hat)
        rows[:, R_BASIS + 6:R_BASIS + 9] = self._line_vectors(v_hat, fr, rho_hat, theta_hat)
        rows[:, R_BASIS + 9:R_BASIS + 12] = self._line_vectors(N, fr, rho_hat, theta_hat)
        in_plane = np.abs(rows[:, R_BASIS + A.B_VM]) < PARABOLA_SIN
        rows[:, R["mode"]] = np.where(in_plane & bool(inp.outline.rotating), A.MODE_PARABOLA, A.MODE_OUTLINE)
        self._bue(inp, rows, tick, tooth, elem)
        return {"rows": rows, "r_act": r_act, "r_hat": r_hat}

    def _element_rz(self, inp, tooth, elem):
        edge = inp.edge
        r = np.asarray(edge.elem_r_m, dtype=float)
        z = np.asarray(edge.elem_z_m, dtype=float)
        j = np.clip(tooth, 0, r.shape[0] - 1)
        k = np.clip(elem, 0, r.shape[1] - 1)
        r_e = r[j, k].copy()
        z_e = z[j, k].copy()
        wear = inp.wear
        if wear is not None:
            rad = np.asarray(getattr(wear, "radial_wear_m", np.zeros((0, 0))), dtype=float)
            axi = np.asarray(getattr(wear, "axial_wear_m", np.zeros((0, 0))), dtype=float)
            if rad.shape == r.shape:
                r_e -= rad[j, k]
            if axi.shape == r.shape:
                z_e += axi[j, k]
        return r_e, z_e

    def _bue(self, inp, rows, tick, tooth, elem) -> None:
        wear = inp.wear
        bue = np.asarray(getattr(wear, "bue_height_m", np.zeros((0, 0))), dtype=float) if wear is not None else None
        if bue is None or bue.shape != np.shape(inp.edge.elem_r_m):
            return
        j = np.clip(tooth, 0, bue.shape[0] - 1)
        k = np.clip(elem, 0, bue.shape[1] - 1)
        h = bue[j, k]
        for i in np.nonzero(h > 0.0)[0]:
            u = hash_uniform(self.seed, STREAM_BUE, int(tick[i]), int(tooth[i]), int(elem[i]))
            rows[i, R["bue_d"]] = float(self.p["bue_overcut_ratio"]) * h[i] * u
            rows[i, R["bue_b"]] = float(self.p["bue_width_ratio"]) * h[i]

    def _fragments(self, inp, rows, tick, tooth, elem) -> np.ndarray:
        wear = inp.wear
        bue = np.asarray(getattr(wear, "bue_height_m", np.zeros((0, 0))), dtype=float) if wear is not None else None
        if bue is None or bue.shape != np.shape(inp.edge.elem_r_m):
            return np.zeros((0, N_ROW))
        j = np.clip(tooth, 0, bue.shape[0] - 1)
        k = np.clip(elem, 0, bue.shape[1] - 1)
        h = bue[j, k]
        out = []
        for i in np.nonzero(h > 0.0)[0]:
            u = hash_uniform(self.seed, STREAM_FRAGMENT, int(tick[i]), int(tooth[i]), int(elem[i]))
            if u >= float(self.p["bue_fragment_probability"]):
                continue
            uh = hash_uniform(self.seed, STREAM_FRAGMENT_H, int(tick[i]), int(tooth[i]), int(elem[i]))
            row = rows[i].copy()
            row[R["kind"]] = A.KIND_ADD
            row[R["mode"]] = A.MODE_DIRECT
            row[R["add_h"]] = float(self.p["bue_fragment_height_ratio"]) * h[i] * uh
            row[R["add_b"]] = float(self.p["bue_width_ratio"]) * h[i]
            row[R["tick"]] = rows[i, R["tick"]] + 0.5
            out.append(row)
        return np.array(out) if out else np.zeros((0, N_ROW))

    def _lead_twins(self, line, rows, r_act, r_hat, omega) -> np.ndarray:
        """Mirrored rows at the leading side for elements whose leading passage lies lower."""
        shift = -2.0 * r_act[:, None] * r_hat                      # trailing -> leading position
        dw = shift @ line.n if not line.cylindrical else np.zeros(len(rows))
        lower = dw < 0.0
        if not lower.any():
            return np.zeros((0, N_ROW))
        twins = rows[lower].copy()
        twins[:, R["s"]] += (shift[lower] @ line.d)
        twins[:, R["w"]] += dw[lower]
        twins[:, R["m"]] += (shift[lower] @ line.axis)
        for c in (A.B_RD, A.B_RN, A.B_RM, A.B_VD, A.B_VN, A.B_VM):
            twins[:, R_BASIS + c] = -twins[:, R_BASIS + c]
        w = np.abs(omega[lower])
        half_rev = np.where(w > 0.0, math.pi / np.where(w > 0.0, w, 1.0) / self.tick_s, 0.0)
        twins[:, R["tick"]] -= np.round(half_rev)
        return twins

    # ------------------------------------------------------------------ recut dents
    def _recut_targets(self, inp, tick, tooth, keys, inverse, uniq) -> dict:
        """line key -> [(event index, recut row)] of the recut rows: the chip is pressed into the surface the
        same tooth generates nearest in time."""
        rec = np.asarray(inp.recut, dtype=float).reshape(-1, 6)
        out: dict = {}
        for r in rec:
            cand = np.nonzero(tooth == int(r[1]))[0]
            if len(cand) == 0:
                continue
            i = int(cand[np.argmin(np.abs(tick[cand] - int(r[0])))])
            u = int(inverse[i])
            key = (int(uniq[u, 0]), int(uniq[u, 1]), int(uniq[u, 2]))
            out.setdefault(key, []).append((i, r, 0))
        return out

    def _dent_rows(self, store, inp, line, dents, rows, idx) -> np.ndarray:
        """KIND_ADD rows of the recut dents (depth t_chip (1 - 1 / hardness ratio), parabolic dent)."""
        ratio = float(self.p["recut_chip_hardness_ratio"])
        width = float(self.p["recut_dent_width_ratio"])
        chip = inp.h_max_m / inp.chip_ratio if inp.chip_ratio > 0.0 else inp.h_max_m
        depth = max(chip * (1.0 - 1.0 / ratio), 0.0)
        out = []
        n0 = len(idx)
        for i, r, _ in dents:
            if depth <= 0.0:
                continue
            same = np.nonzero(rows[:n0, R["tooth"]] == float(r[1]))[0]
            if len(same) == 0:
                same = np.arange(n0)
            q = int(same[np.argmin(np.abs(rows[same, R["tick"]] - float(r[0])))])
            row = rows[q].copy()
            row[R["kind"]] = A.KIND_ADD
            row[R["mode"]] = A.MODE_DIRECT
            row[R["add_h"]] = -depth
            row[R["add_b"]] = width * depth
            row[R["tick"]] = max(float(r[0]), row[R["tick"]]) + 0.75
            out.append(row)
            p = line.patch
            store.pf[p, PF["recut"]] += 1.0
            store.pf[p, PF["recut_max"]] = max(store.pf[p, PF["recut_max"]], depth)
        return np.array(out) if out else np.zeros((0, N_ROW))


__all__ = ["Gatherer", "StepInput", "StepOutput", "CLS_CODE"]
