"""Cutter radius compensation (G41/G42/G40, FANUC type C), tool nose radius compensation and optional
chamfering / corner rounding (,C ,R) as a block-stream filter of the controller core.

The core hands every work-frame :class:`~cnctwin.controller.hooks.PathItem` to ``push`` while a
compensation is active; the filter holds items until it can see far enough ahead and releases them
with compensated end points (``flush`` releases everything at a buffer stop). Its whole state is the
JSON dict ``to_data()``: the queue of held items (serialised), the compensated tool position, the
pending corner points that belong to the next block, the mode and flags. Nothing else is kept, so a
checkpoint can fall anywhere inside a compensated contour.

Rules (fanuc-30ibplus-om-mc-2019 6.4-6.6; parameters in db/models/controller-cycles.yaml):

* look-ahead: N blocks (parameter 19625, default 3); the corner at the end of block A is computed with
  the next block B that moves in the offset plane when B lies within the N - 1 following blocks (so at
  most N - 2 blocks without movement in between); otherwise, and before a bypassed move (G28, G53,
  canned-cycle positioning), a buffer stop, a block with zero travel or G41/G42 given again alone, A
  ends with the vector normal to it;
* start-up (first block moving in the plane after G41/G42, G00/G01 only: PS0034 for an arc): type A (end
  normal of B), type B (normal of the start-up block, then the corner vectors), inner corners always to
  the intersection; type C: a start-up block without movement shifts by the offset normal to B;
* offset mode corners by :func:`geometry.corner` (inner intersection, outer obtuse intersection, outer
  acute two extension vectors, or circular connection), D changes take effect at the end of the block
  that names them, a side change goes to the intersection of both offset paths when it lies on them;
* IJ type vector (I J K in a G00/G01 block, in the start-up block or alone): the vector normal to (I, J)
  replaces the intersection; G40 I J K cancels toward that direction; G39 inserts an arc of radius r;
* cancel (G40 or D0): type A (A's end normal, then to the programmed end), type B (intersection and the
  normal of the cancel block), inner corners via the intersection; type C cancel without movement
  moves back by the vector in the G40 block; arcs: PS0034;
* interference check (5008, 19607#5 CAV = 0): for the next block, direction check and circular angle
  check between its start and end corner vectors; interfering inner vectors are erased when a corner
  has several, otherwise PS0041 before the preceding block is executed; non-adjacent corners in the
  window are compared by direction (PS0041);
* tool nose radius compensation: the same with the nose radius, the commanded point being the
  imaginary tool nose: compensated nose-centre point + r * tip direction (tip 0-9);
* ,C / ,R between this block and the next G01/G02/G03 (PS0051 missing move, PS0052 not G01/G02/G03,
  PS0050 in a thread block, PS0055 too large) are applied to the programmed path before offsetting.
"""

from __future__ import annotations

import math

from .. import serial
from ..alarms import ControllerAlarm
from ..hooks import PathItem
from ..ir import Kind
from ..pipeline import PLANES, UNIT
from ..cycles import common
from . import geometry as G

SIDE = {"G41": 1.0, "G42": -1.0}


def _norm3(v):
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return [v[0] / n, v[1] / n, v[2] / n] if n > 0 else [0.0, 0.0, 0.0]


def _cross3(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _dot3(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


class Frame2D:
    """The compensation plane in the work frame: e1, e2 the images of the program plane axes, n = e1 x e2."""

    def __init__(self, e1, e2):
        self.e1 = _norm3(e1)
        self.e2 = _norm3(e2)
        self.n = _norm3(_cross3(self.e1, self.e2))

    @classmethod
    def of(cls, ex, plane: str) -> "Frame2D":
        a, b, _ = PLANES[plane]
        lin = ex.transform().linear
        ea = [sum(lin[i][k] * UNIT[a][k] for k in range(3)) for i in range(3)]
        eb = [sum(lin[i][k] * UNIT[b][k] for k in range(3)) for i in range(3)]
        return cls(ea, eb)

    def to2(self, p3):
        return (_dot3(p3, self.e1), _dot3(p3, self.e2))

    def w(self, p3) -> float:
        return _dot3(p3, self.n)

    def to3(self, uv, w):
        return [uv[0] * self.e1[i] + uv[1] * self.e2[i] + w * self.n[i] for i in range(3)]

    def data(self) -> list:
        return [list(self.e1), list(self.e2)]


class CompFilter:
    """One active compensation (``kind`` cutter | nose | corner) of one path."""

    def __init__(self, ex, kind: str, data: dict | None = None):
        self.kind = kind
        if data is not None:
            self.d = data
            return
        self.d = {"kind": kind, "queue": [], "mode": "startup", "pos": None, "prog": None, "last": None,
                  "last_d": 0.0, "pending": [], "pending_arc": None, "erase_head": 0,
                  "startup_seq": int(ex.s.block_seq), "first_startup": True, "after_cancel": None,
                  "cancelled": False, "frame": None}

    # ================================================================== core interface
    def to_data(self) -> dict:
        return self.d

    def push(self, ex, item: PathItem) -> list[PathItem]:
        marker = self._marker(item)
        number_key = "offset_number" if self.kind == "nose" else "d"
        entry = {"item": serial.dump(item), "frame": Frame2D.of(ex, item.plane or "XY").data(),
                 "radius": float(item.comp_radius_m), "d_number": int((item.modal or {}).get(number_key, 1)),
                 "tip": self._tip(ex, item)}
        if marker is not None:
            entry = {"marker": marker, "seq": int(item.block_seq)}
        self.d["queue"].append(entry)
        return self._drain(ex, final=False)

    def flush(self, ex, reason: str) -> list[PathItem]:
        return self._drain(ex, final=True, reason=reason)

    # ================================================================== queue helpers
    def _marker(self, item: PathItem) -> dict | None:
        if item.kind != "command" or item.command is None:
            return None
        cmd = item.command
        if item.corner and item.corner.get("code") == "G39":
            return {"g39": dict(item.corner.get("words") or {})}
        modal = cmd.modal or {}
        if cmd.kind == Kind.MODE and "comp_marker" in modal:
            return dict(modal["comp_marker"])
        if cmd.kind == Kind.MODE and "motion_words" in modal and "comp_ijk" in modal["motion_words"]:
            return {"ijk": dict(modal["motion_words"]["comp_ijk"])}
        return None

    def _tip(self, ex, item: PathItem) -> int:
        if self.kind != "nose":
            return 0
        number = int((item.modal or {}).get("offset_number", ex.s.offset_number) or 0)
        row = ex.s.tool_offsets.get(str(number))
        return int(row[3]) if row else 0

    def _item(self, entry) -> PathItem:
        return serial.load(PathItem, entry["item"])

    def _frame(self, entry) -> Frame2D:
        return Frame2D(*entry["frame"])

    def _d(self, entry) -> float:
        """Signed offset of an entry (G41 left +, G42 right -, sign of the D value); 0 for D0."""
        if self.kind == "corner":
            return 0.0
        item = entry["item"]
        side = SIDE.get(item.get("comp"), 0.0)
        if entry.get("d_number", 1) == 0:
            return 0.0
        return side * float(entry["radius"])

    def _xyz(self, ex, values):
        out = [0.0, 0.0, 0.0]
        for i, letter in enumerate(ex.letters):
            if letter in "XYZ":
                out["XYZ".index(letter)] = float(values[i])
        return out

    def _set_xyz(self, ex, values, p3):
        out = list(values)
        for i, letter in enumerate(ex.letters):
            if letter in "XYZ":
                out[i] = p3["XYZ".index(letter)]
        return out

    def _is_motion(self, entry) -> bool:
        return "item" in entry and entry["item"]["kind"] in ("rapid", "linear", "arc")

    def _in_plane(self, ex, entry) -> bool:
        item = entry["item"]
        f = self._frame(entry)
        s2 = f.to2(self._xyz(ex, item["start"]))
        e2 = f.to2(self._xyz(ex, item["end"]))
        if item["kind"] == "arc":
            return True
        return G.dist(s2, e2) > 1e-12

    def _no_travel(self, ex, entry) -> bool:
        item = entry["item"]
        return item["kind"] != "arc" and all(abs(a - b) <= 0.0 for a, b in zip(item["start"], item["end"]))

    def _bypass(self, entry) -> bool:
        item = entry["item"]
        return bool(item.get("bypass")) or item.get("space") == "direct"

    def _seg(self, ex, entry, f: Frame2D) -> G.Seg:
        item = entry["item"]
        s2 = f.to2(self._xyz(ex, item["start"]))
        e2 = f.to2(self._xyz(ex, item["end"]))
        if item["kind"] != "arc":
            return G.Seg("line", s2, e2)
        c2 = f.to2(list(item["centre"]))
        sweep = float(item["sweep_rad"]) * (1.0 if _dot3(item["normal"], f.n) >= 0.0 else -1.0)
        return G.Seg("arc", s2, e2, c2, sweep)

    # ================================================================== look-ahead
    def _window_complete(self, ex, start: int) -> bool:
        """The N - 1 blocks after the block of queue entry ``start`` have been read (FANUC reads N blocks
        in compensation mode, parameter 19625), or a block that ends the look-ahead (bypassed move, move
        without travel, G40 / G41 alone, cycle marker) has arrived and is complete."""
        queue = self.d["queue"]
        seq_a = int(queue[start]["item"]["block_seq"])
        limit = int(common.params(ex)["crc_lookahead_blocks"])
        seen: list[int] = []
        for k in range(start + 1, len(queue)):
            entry = queue[k]
            seq = int(entry["seq"]) if "marker" in entry else int(entry["item"]["block_seq"])
            if seq != seq_a and seq not in seen:
                seen.append(seq)
                if len(seen) > limit - 1:
                    return True
            if "marker" in entry and (entry["marker"].get("cancel") or entry["marker"].get("respecified")
                                      or entry["marker"].get("cycle")) and self._block_complete(k):
                return True
            if self._is_motion(entry) and (self._bypass(entry) or self._no_travel(ex, entry)):
                return True
        return False

    def _block_complete(self, k: int) -> bool:
        """Every item of the block of queue entry k has arrived (a later block has started)."""
        queue = self.d["queue"]
        entry = queue[k]
        seq = int(entry["seq"]) if "marker" in entry else int(entry["item"]["block_seq"])
        for later in queue[k + 1:]:
            other = int(later["seq"]) if "marker" in later else int(later["item"]["block_seq"])
            if other != seq:
                return True
        return False

    def _lookahead(self, ex, start: int, final: bool) -> tuple[str, int | None, dict]:
        """From queue index ``start`` (a motion of block A): ("found", index of B, info) | ("none", None, info)
        | ("wait", None, {}). info: reason, g39 words, ijk direction, cancel kind."""
        queue = self.d["queue"]
        seq_a = int(queue[start]["item"]["block_seq"])
        limit = int(common.params(ex)["crc_lookahead_blocks"])
        seen: list[int] = []
        info: dict = {}
        for k in range(start + 1, len(queue)):
            entry = queue[k]
            seq = int(entry["seq"]) if "marker" in entry else int(entry["item"]["block_seq"])
            if seq != seq_a and seq not in seen:
                seen.append(seq)
                if len(seen) > limit - 1:
                    return "none", None, dict(info, reason="window")
            if "marker" in entry:
                marker = entry["marker"]
                if "g39" in marker:
                    info["g39"] = marker["g39"]
                elif "ijk" in marker:
                    return "none", None, dict(info, reason="ijk", ijk=marker["ijk"])
                elif marker.get("cancel") or marker.get("respecified") or marker.get("cycle"):
                    if not final and not self._block_complete(k):
                        return "wait", None, {}
                    has_move = any(self._is_motion(e) and int(e["item"]["block_seq"]) == seq and self._in_plane(ex, e)
                                   and not self._bypass(e) for e in queue[k + 1:])
                    if marker.get("cancel") and has_move:
                        continue
                    if marker.get("respecified") and has_move:
                        continue
                    reason = "cancel_alone" if marker.get("cancel") else "respec" if marker.get("respecified") else "cycle"
                    return "none", None, dict(info, reason=reason)
                continue
            if not self._is_motion(entry):
                continue
            if self._bypass(entry):
                return "none", None, dict(info, reason="bypass")
            if self._no_travel(ex, entry):
                return "none", None, dict(info, reason="zero")
            if not self._in_plane(ex, entry):
                continue
            return "found", k, info
        if final:
            return "none", None, dict(info, reason="end")
        return "wait", None, {}

    # ================================================================== drain
    def _drain(self, ex, final: bool, reason: str = "") -> list[PathItem]:
        out: list[PathItem] = []
        queue = self.d["queue"]
        while queue:
            entry = queue[0]
            if "marker" in entry:
                if not self._handle_marker(ex, entry, out, final):
                    break
                continue
            if not self._is_motion(entry):
                queue.pop(0)
                out.append(self._item(entry))
                continue
            item = entry["item"]
            mode = self.d["mode"]
            if self.kind == "corner":
                if not item.get("corner"):
                    queue.pop(0)
                    out.append(self._rebase(ex, entry))
                    continue
                status, nxt, info = self._lookahead(ex, 0, final)
                if status == "wait":
                    break
                queue.pop(0)
                out.extend(self._release_corner_only(ex, entry, None if nxt is None else nxt - 1, info))
                continue
            cancel = item.get("comp") == "G40" or self._d(entry) == 0.0
            if self._bypass(entry) or mode in ("off", "cancel_pending") or (cancel and mode != "offset"):
                queue.pop(0)
                out.extend(self._release_plain(ex, entry))
                if self._bypass(entry) and self.d["mode"] == "offset":
                    self.d["mode"] = "startup"
                    self.d["first_startup"] = False
                continue
            if cancel:
                queue.pop(0)
                out.extend(self._release_cancel(ex, entry))
                continue
            if mode == "offset" and not self._in_plane(ex, entry):
                queue.pop(0)
                out.extend(self._release_off_plane(ex, entry))
                continue
            if mode == "startup" and (not self._in_plane(ex, entry) or self._no_travel(ex, entry)):
                queue.pop(0)
                out.extend(self._release_plain(ex, entry))
                continue
            status, nxt, info = self._lookahead(ex, 0, final)
            if status == "wait" or (not final and not self._window_complete(ex, 0)):
                break
            queue.pop(0)
            nxt = None if nxt is None else nxt - 1
            out.extend(self._release_block(ex, entry, nxt, info, startup=mode == "startup"))
        if final and self.kind != "corner" and self.d["mode"] == "offset" and self.d["pending"]:
            out.extend(self._flush_pending(ex))
        self._maybe_end(ex, out)
        return out

    def _flush_pending(self, ex) -> list[PathItem]:
        """A buffer stop after a block whose corner had several vectors: the tool stops at the first one
        (the single-block stop point); the remaining vectors are recomputed with the next block."""
        self.d["pending"] = []
        self.d["pending_arc"] = None
        return []

    def _maybe_end(self, ex, out) -> None:
        """Compensation fully ended: release what is left and leave the core's compensation mode."""
        queue = self.d["queue"]
        if self.kind == "corner":
            if not any("item" in e and e["item"].get("corner") for e in queue):
                for entry in queue:
                    if "item" in entry:
                        out.append(self._rebase(ex, entry))
                self.d["queue"] = []
                ex.end_comp()
            return
        if self.d["mode"] == "off" and not queue:
            ex.end_comp()

    # ================================================================== markers
    def _handle_marker(self, ex, entry, out, final: bool) -> bool:
        queue = self.d["queue"]
        marker = entry["marker"]
        if "g39" in marker:
            moved = any(self._is_motion(e) and int(e["item"]["block_seq"]) == entry["seq"] for e in queue[1:])
            if moved:
                raise ControllerAlarm("crc_g39_motion", "G39 with a move command")
            queue.pop(0)
            return True
        if marker.get("cancel"):
            if not final and not self._block_complete(0):
                return False
            has_move = any(self._is_motion(e) and int(e["item"]["block_seq"]) == entry["seq"] for e in queue[1:])
            queue.pop(0)
            if not has_move and self.d["mode"] == "offset":
                if self._types(common.params(ex))[1]:
                    out.extend(self._move_to_program(ex))
                    self.d["mode"] = "off"
                else:
                    self.d["mode"] = "cancel_pending"
            return True
        if marker.get("cycle") or marker.get("respecified"):
            queue.pop(0)
            if marker.get("cycle") and self.d["mode"] == "offset":
                out.extend(self._move_to_program(ex))
                self.d["mode"] = "startup"
                self.d["first_startup"] = False
            return True
        queue.pop(0)
        return True

    def _move_to_program(self, ex) -> list[PathItem]:
        """Back from the compensated point to the programmed point of the last block (zero vector)."""
        prog, last = self.d["prog"], self.d["last"]
        if prog is None or last is None or self.d["pos"] is None:
            return []
        template = serial.load(PathItem, last["item"])
        f = Frame2D(*last["frame"])
        target = self._set_xyz(ex, template.end, f.to3(prog[0:2], prog[2]))
        item = self._clone(template, "rapid" if template.kind == "rapid" else "linear", self.d["pos"], target)
        self.d["pos"] = list(target)
        self.d["pending"] = []
        return [item]

    # ================================================================== releases
    def _clone(self, template: PathItem, kind: str, start, end, centre=None, normal=None, sweep=0.0,
               r0=0.0, r1=0.0) -> PathItem:
        item = serial.load(PathItem, serial.dump(template))
        item.kind = kind
        item.start = list(start)
        item.end = list(end)
        item.corner = None
        if kind == "arc":
            item.centre, item.normal, item.sweep_rad, item.r_start, item.r_end = list(centre), list(normal), sweep, r0, r1
        else:
            item.centre, item.normal, item.sweep_rad, item.r_start, item.r_end = None, None, 0.0, 0.0, 0.0
        return item

    def _rebase(self, ex, entry) -> PathItem:
        item = self._item(entry)
        if self.d["pos"] is not None and item.kind in ("rapid", "linear", "arc"):
            item.start = list(self.d["pos"])
        if item.kind in ("rapid", "linear", "arc"):
            self.d["pos"] = list(item.end)
        return item

    def _release_plain(self, ex, entry) -> list[PathItem]:
        """An uncompensated move from the current compensated position to its programmed end."""
        comp = entry["item"].get("comp")
        if self.d["mode"] == "cancel_pending" or (self.d["mode"] == "startup" and comp == "G40"):
            self.d["mode"] = "off" if comp == "G40" else "startup"
        item = self._rebase(ex, entry)
        self._remember(ex, entry, item.end)
        self.d["pending"] = []
        self.d["pending_arc"] = None
        return [item]

    def _release_off_plane(self, ex, entry) -> list[PathItem]:
        """A move without travel in the offset plane: the in-plane position stays the compensated one."""
        item = self._item(entry)
        f = self._frame(entry)
        pos = self.d["pos"] or item.start
        uv = f.to2(self._xyz(ex, pos))
        end = self._set_xyz(ex, item.end, f.to3(uv, f.w(self._xyz(ex, item.end))))
        item.start = list(pos)
        item.end = end
        self.d["pos"] = list(end)
        return [item]

    def _remember(self, ex, entry, end) -> None:
        f = self._frame(entry)
        prog = f.to2(self._xyz(ex, entry["item"]["end"]))
        self.d["prog"] = [prog[0], prog[1], f.w(self._xyz(ex, entry["item"]["end"]))]
        self.d["last"] = entry
        self.d["pos"] = list(end)

    def _release_cancel(self, ex, entry) -> list[PathItem]:
        """G40 / D0 block with movement in offset mode (its start corner was set by the previous block)."""
        item = self._item(entry)
        if item.kind == "arc":
            raise ControllerAlarm("crc_arc_startup", "G40 in a circular block")
        out = self._pending_moves(ex, entry, item)
        start = self.d["pos"]
        moved = self._clone(item, item.kind, start, item.end)
        out.append(moved)
        self.d["mode"] = "off" if item.comp == "G40" else "startup"
        self.d["first_startup"] = False
        self._remember(ex, entry, moved.end)
        self.d["pending"] = []
        return out

    def _pending_moves(self, ex, entry, item: PathItem) -> list[PathItem]:
        """Corner points left over from the previous block's end: they belong to this block."""
        out = []
        pending = self.d["pending"]
        arc = self.d["pending_arc"]
        f = self._frame(entry)
        w = f.w(self._xyz(ex, self.d["pos"] or item.start))
        kind = "rapid" if item.kind == "rapid" else "linear"
        limit = float(common.params(ex)["crc_vector_limit_m"])
        for k, point in enumerate(pending):
            p3 = self._set_xyz(ex, item.start, f.to3(point, w))
            here = f.to2(self._xyz(ex, self.d["pos"]))
            if limit > 0.0 and abs(point[0] - here[0]) <= limit and abs(point[1] - here[1]) <= limit:
                continue
            if arc is not None and k == 0:
                centre = f.to3(arc[0:2], w)
                sweep_uv = G.sweep_between(arc[0:2], here, point, -1.0 if arc[2] else 1.0)
                normal = list(f.n)
                out.append(self._clone(item, "arc", self.d["pos"], p3, centre, normal, sweep_uv,
                                       G.dist(here, arc[0:2]), G.dist(point, arc[0:2])))
            else:
                out.append(self._clone(item, kind, self.d["pos"], p3))
            self.d["pos"] = list(p3)
        self.d["pending"] = []
        self.d["pending_arc"] = None
        return out

    def _release_block(self, ex, entry, nxt: int | None, info: dict, startup: bool) -> list[PathItem]:
        p = common.params(ex)
        item = self._item(entry)
        f = self._frame(entry)
        d = self._d(entry)
        if startup and item.kind == "arc":
            raise ControllerAlarm("crc_arc_startup", "start-up of compensation in a circular block")
        seg_a = self._seg(ex, entry, f)
        if item.corner:
            seg_a, nxt = self._insert_corner(ex, entry, seg_a, nxt, info, f)
        ijk = (item.modal or {}).get("comp_ijk")
        out: list[PathItem] = []
        tip = G.tip_vector(entry.get("tip", 0), entry["radius"]) if self.kind == "nose" else (0.0, 0.0)
        if startup:
            corner_pts, arc = self._startup_corner(ex, entry, seg_a, nxt, info, d, f, ijk, p)
            if self.d["first_startup"] and self._types(p)[1] and int(item.block_seq) != self.d["startup_seq"]:
                shift = G.offset_point(seg_a, d, at_end=False)
                shift_uv = G.add(shift, tip)
                w0 = f.w(self._xyz(ex, self.d["pos"] or item.start))
                p3 = self._set_xyz(ex, item.start, f.to3(shift_uv, w0))
                out.append(self._clone(item, "rapid" if item.kind == "rapid" else "linear",
                                       self.d["pos"] or item.start, p3))
                self.d["pos"] = list(p3)
                corner_pts, arc = self._offset_corner(ex, entry, seg_a, nxt, info, d, f, ijk, p)
        else:
            out.extend(self._pending_moves(ex, entry, item))
            corner_pts, arc = self._offset_corner(ex, entry, seg_a, nxt, info, d, f, ijk, p)
        erase = int(self.d.get("erase_head", 0))
        self.d["erase_head"] = 0
        if erase:
            corner_pts = corner_pts[erase:] or corner_pts[-1:]
            arc = None
        corner_pts = self._check_next(ex, entry, seg_a, nxt, corner_pts, d, f, p, startup)
        corner_pts = [G.add(q, tip) for q in corner_pts]
        unique = [corner_pts[0]]
        for q in corner_pts[1:]:
            if G.dist(q, unique[-1]) > 1e-12:
                unique.append(q)
        corner_pts = unique
        end_uv = corner_pts[0]
        w_end = f.w(self._xyz(ex, item.end))
        end3 = self._set_xyz(ex, item.end, f.to3(end_uv, w_end))
        start3 = list(self.d["pos"] or item.start)
        if item.kind == "arc" and not startup:
            start2 = f.to2(self._xyz(ex, start3))
            centre2 = G.add(seg_a.c, tip)
            sweep_uv = G.sweep_between(centre2, start2, end_uv, seg_a.sweep,
                                       full=abs(abs(seg_a.sweep) - 2 * math.pi) < 1e-9 and G.dist(start2, end_uv) < 1e-12)
            if G.arc_interferes(G.Seg("arc", seg_a.p0, seg_a.p1, seg_a.c, seg_a.sweep), G.sub(start2, tip), G.sub(end_uv, tip)):
                raise ControllerAlarm("crc_interference", "compensated arc reverses")
            w0 = f.w(self._xyz(ex, start3))
            centre3 = f.to3(centre2, w0)
            normal = list(f.n)
            r0 = G.dist(start2, centre2)
            r1 = G.dist(end_uv, centre2)
            out.append(self._clone(item, "arc", start3, end3, centre3, normal, sweep_uv, r0, r1))
        else:
            out.append(self._clone(item, item.kind, start3, end3))
        self.d["pending"] = [list(q) for q in corner_pts[1:]]
        self.d["pending_arc"] = [arc[0][0] + tip[0], arc[0][1] + tip[1], bool(arc[1])] if arc is not None else None
        self.d["mode"] = "offset"
        self.d["first_startup"] = False
        self.d["last_d"] = d
        self._remember(ex, entry, end3)
        return out

    def _types(self, p) -> tuple[str, bool]:
        """(type used for blocks with movement "A"/"B", type C for blocks without movement) of 5003."""
        t = str(p["crc_startup_type"]).upper()
        return ("B" if t in ("B", "CB") else "A"), t in ("C", "CB")

    def _startup_corner(self, ex, entry, seg_a, nxt, info, d, f, ijk, p):
        if ijk is not None:
            return self._ijk_corner(seg_a, ijk, d, f, entry), None
        if nxt is None:
            if info.get("reason") == "ijk":
                return self._ijk_corner(seg_a, info["ijk"], d, f, entry), None
            return G.perpendicular(seg_a, d, at_end=True).points, None
        nentry = self.d["queue"][nxt]
        seg_b = self._seg(ex, nentry, f)
        if nentry["item"].get("comp") == "G40" or self._d(nentry) == 0.0:
            return G.perpendicular(seg_a, d, at_end=True).points, None
        d_b = self._d(nentry)
        if d_b * d < 0.0:
            return G.side_change(seg_a, seg_b, d, d_b).points, None
        phi = G.turn(seg_a, seg_b) * (1.0 if d > 0.0 else -1.0)
        if phi > 0.0 or abs(phi) < G.TANGENT_RAD:
            return G.corner(seg_a, seg_b, d, p["crc_outer_corner"]).points, None
        moving, _ = self._types(p)
        if moving == "A":
            return [G.offset_point(seg_b, d, at_end=False)], None
        corner = G.corner(seg_a, seg_b, d, p["crc_outer_corner"])
        return [G.offset_point(seg_a, d, at_end=True)] + corner.points, corner.arc

    def _offset_corner(self, ex, entry, seg_a, nxt, info, d, f, ijk, p):
        if ijk is not None:
            return self._ijk_corner(seg_a, ijk, d, f, entry), None
        if nxt is None:
            if info.get("reason") == "ijk":
                return self._ijk_corner(seg_a, info["ijk"], d, f, entry), None
            return G.perpendicular(seg_a, d, at_end=True).points, None
        nentry = self.d["queue"][nxt]
        seg_b = self._seg(ex, nentry, f)
        nitem = nentry["item"]
        if info.get("g39") is not None:
            return self._g39_corner(seg_a, seg_b, d, info["g39"], f, entry)
        if nitem.get("comp") == "G40" or self._d(nentry) == 0.0:
            return self._cancel_corner(ex, seg_a, seg_b, nentry, d, f, p)
        d_b = self._d(nentry)
        if d_b * d < 0.0:
            return G.side_change(seg_a, seg_b, d, d_b).points, None
        corner = G.corner(seg_a, seg_b, d, p["crc_outer_corner"])
        return corner.points, corner.arc

    def _cancel_corner(self, ex, seg_a, seg_b, nentry, d, f, p):
        if seg_b.kind == "arc":
            raise ControllerAlarm("crc_arc_startup", "G40 in a circular block")
        ijk = (nentry["item"].get("modal") or {}).get("comp_ijk")
        if ijk is not None:
            direction = self._ijk_direction(ijk, f, nentry)
            virtual = G.Seg("line", seg_a.p1, G.add(seg_a.p1, direction))
            oa, ov = G.offset(seg_a, d), G.offset(virtual, d)
            hit = G.nearest(G.intersections(oa, ov), seg_a.p1) if oa is not None and ov is not None else None
            return ([hit] if hit is not None else G.perpendicular(seg_a, d, True).points), None
        phi = G.turn(seg_a, seg_b) * (1.0 if d > 0.0 else -1.0)
        if phi > 0.0:
            return G.corner(seg_a, seg_b, d, p["crc_outer_corner"]).points, None
        if self._types(p)[0] == "B":
            corner = G.corner(seg_a, seg_b, d, p["crc_outer_corner"])
            return corner.points + [G.offset_point(seg_b, d, at_end=False)], corner.arc
        return G.perpendicular(seg_a, d, at_end=True).points, None

    def _g39_corner(self, seg_a, seg_b, d, words, f, entry):
        va = G.offset_point(seg_a, d, at_end=True)
        if words:
            direction = self._ijk_direction(words, f, entry)
            vb = G.ijk_vector(seg_a.p1, direction, d).points[0]
        else:
            vb = G.offset_point(seg_b, d, at_end=False)
        phi = G.turn(seg_a, seg_b)
        return [va, vb], (seg_a.p1, phi < 0.0)

    def _ijk_direction(self, ijk: dict, f: Frame2D, entry) -> tuple[float, float]:
        vec = [float(ijk.get("I", 0.0)), float(ijk.get("J", 0.0)), float(ijk.get("K", 0.0))]
        return f.to2(vec)

    def _ijk_corner(self, seg_a, ijk, d, f, entry):
        return G.ijk_vector(seg_a.p1, self._ijk_direction(ijk, f, entry), d).points

    # ================================================================== interference check
    def _check_next(self, ex, entry, seg_a, nxt, points, d, f, p, startup: bool):
        """Direction / circular angle check of the next block B (between this corner and B's end corner)
        and of later corners in the window against this one; vector erasure, else PS0041."""
        if not p["crc_interference_check"] or nxt is None or self.kind == "corner":
            return points
        queue = self.d["queue"]
        nentry = queue[nxt]
        if nentry["item"].get("comp") == "G40" or self._d(nentry) == 0.0:
            return points
        seg_b = self._seg(ex, nentry, f)
        _, nn, _ = self._lookahead(ex, nxt, final=True)
        later = []
        index, seg_prev, entry_prev = nn, seg_b, nentry
        while index is not None and len(later) < max(1, int(p["crc_lookahead_blocks"]) - 2):
            centry = queue[index]
            if centry["item"].get("comp") == "G40" or self._d(centry) == 0.0 or self._d(centry) * d < 0.0:
                break
            seg_c = self._seg(ex, centry, f)
            try:
                corner = G.corner(seg_prev, seg_c, self._d(entry_prev), p["crc_outer_corner"])
            except ControllerAlarm:
                break
            later.append((seg_prev, seg_c, corner.points))
            _, index, _ = self._lookahead(ex, index, final=True)
            seg_prev, entry_prev = seg_c, centry
        if not later:
            return points
        mine = list(points)
        theirs = list(later[0][2])
        first, _ = self._erase(seg_b, mine, theirs)
        if first is None:
            raise ControllerAlarm("crc_interference", "the compensated path of the next block runs backwards")
        for seg_x, _, pts in later[1:]:
            for q in pts:
                if G.direction_interferes(seg_a.p1, seg_x.p1, mine[-1], q):
                    raise ControllerAlarm("crc_interference", "overcut detected in the look-ahead blocks")
        return first

    def _erase(self, seg_b: G.Seg, mine: list, theirs: list):
        """Vector erasure for the block between corner ``mine`` and corner ``theirs`` (FANUC 6.6.6.2)."""
        def bad(u, w):
            return G.direction_interferes(seg_b.p0, seg_b.p1, u, w) or G.arc_interferes(seg_b, u, w)

        pairs = [(i, j) for i in range(len(mine)) for j in range(len(theirs)) if bad(mine[i], theirs[j])]
        if not pairs:
            return mine, theirs
        i = min(i for i, _ in pairs)
        j = max(j for _, j in pairs)
        keep_mine, keep_theirs = mine[:i], theirs[j + 1:]
        if not keep_mine or not keep_theirs or bad(keep_mine[-1], keep_theirs[0]):
            return None, None
        self.d["erase_head"] = j + 1
        return keep_mine, keep_theirs

    # ================================================================== chamfer / corner rounding
    def _insert_corner(self, ex, entry, seg_a, nxt, info, f):
        """,C / ,R of block A with the next block B: A is trimmed, the chamfer or fillet becomes a new
        element of A's block queued before B, and B's start is moved to its end."""
        item = entry["item"]
        request = dict(item["corner"])
        queue = self.d["queue"]
        if ",A" in request:
            raise ControllerAlarm("unsupported_function", "direct drawing dimension programming (,A)")
        if (item.get("modal") or {}).get("thread"):
            raise ControllerAlarm("chamfer_in_thread", ",C/,R in a threading block")
        if nxt is None:
            raise ControllerAlarm("chamfer_missing_move", ",C/,R without a following G01/G02/G03 block")
        nentry = queue[nxt]
        if nentry["item"]["kind"] not in ("linear", "arc") or (nentry["item"].get("modal") or {}).get("thread"):
            raise ControllerAlarm("chamfer_not_g01", ",C/,R followed by a block that is not G01/G02/G03")
        if ex.dialect.name == "haas" and nentry["item"]["kind"] != "linear":
            raise ControllerAlarm("chamfer_not_g01", ",C/,R must be followed by a G01 block (Haas)")
        if nentry["item"].get("plane") != item.get("plane"):
            raise ControllerAlarm("chamfer_missing_move", ",C/,R followed by a block in another plane")
        seg_b = self._seg(ex, nentry, f)
        size = request.get(",C", request.get(",R"))
        if ",C" in request and ",R" not in request:
            a2, insert, b2 = G.chamfer(seg_a, seg_b, float(request[",C"]))
        else:
            a2, insert, b2 = G.round_corner(seg_a, seg_b, float(size))
        w = f.w(self._xyz(ex, item["end"]))
        template = self._item(entry)
        mid3 = self._set_xyz(ex, template.end, f.to3(insert.p0, w))
        new_end3 = self._set_xyz(ex, template.end, f.to3(insert.p1, w))
        if insert.kind == "arc":
            centre3 = f.to3(insert.c, w)
            extra = self._clone(template, "arc", mid3, new_end3, centre3, list(f.n), insert.sweep,
                                insert.radius, insert.radius)
        else:
            extra = self._clone(template, "linear", mid3, new_end3)
        extra.corner = None
        entry["item"]["end"] = list(mid3)
        entry["item"]["corner"] = None
        if template.kind == "arc":
            entry["item"]["sweep_rad"] = a2.sweep * (1.0 if _dot3(item["normal"], f.n) >= 0.0 else -1.0)
        new_entry = dict(entry, item=serial.dump(extra))
        queue.insert(0, new_entry)
        nentry["item"]["start"] = list(new_end3)
        if nentry["item"]["kind"] == "arc":
            nentry["item"]["sweep_rad"] = b2.sweep * (1.0 if _dot3(nentry["item"]["normal"], f.n) >= 0.0 else -1.0)
        return a2, 0

    def _release_corner_only(self, ex, entry, nxt, info) -> list[PathItem]:
        """The ``corner`` kind (no compensation): apply ,C / ,R and release the trimmed block."""
        item = entry["item"]
        if item.get("corner"):
            f = self._frame(entry)
            seg_a = self._seg(ex, entry, f)
            self._insert_corner(ex, entry, seg_a, nxt, info, f)
        out = self._rebase(ex, entry)
        return [out]


class CompFactory:
    def __init__(self, kind: str):
        self.kind = kind

    def create(self, ex) -> CompFilter:
        return CompFilter(ex, self.kind)

    def from_data(self, ex, data: dict) -> CompFilter:
        return CompFilter(ex, self.kind, data)


__all__ = ["CompFilter", "CompFactory", "Frame2D"]
