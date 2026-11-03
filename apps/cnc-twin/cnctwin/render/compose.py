"""One picture of a run at one tick: the in-process workpiece, fixture, machine, tool (moved by frames.bin),
chips, coolant streams, chip piles, glow/smoke/steam cues, labels and a colour layer
(docs/models/render.md "Pictures of a run").

Everything is drawn in the stock frame S. Objects given in M (chips, jets, components with ``frame: M``) are
placed with ``pose_M_S`` from the keyframe fields (or ``scene.stock.pose_M_S``); a run that gives neither has
those objects left out and says so in ``FrameInfo.notes``. The workpiece changes at keyframes (its geometry is
the newest keyframe at or before the frame); the tool, labels and cues follow the frame's own tick.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import camera as CAM
from . import colour as C
from . import cues as Q
from . import layers as L
from . import params as P
from . import shapes as SH
from .rundir import RunData, db_item
from .scene import ColorBar, Label, Material, RenderMesh, Scene, section_plane

CAMERA_MODES = ("follow", "part", "setup", "overview")
FINDING_LABEL_ROWS = 4          # newest findings listed under the title, one line each
FINDING_LABEL_FIRST_ROW = 4.2   # first line, in label heights from the top (below the title and the time line)
ERROR_LABEL_SRGB = (0.70, 0.08, 0.05)     # findings of severity error or fatal
NOTICE_LABEL_SRGB = (0.55, 0.30, 0.0)     # info and warning findings


@dataclass
class FrameOptions:
    layer: str | None = None              # one of layers.LAYERS, or None for appearance colours
    camera: str = "follow"                # follow | part | setup | overview
    projection: str = "perspective"       # perspective | orthographic
    section: tuple | None = None          # (point_S_m, normal): keep the side the normal points to
    chips: bool = True
    pile: bool = True                     # settled chip piles (drawn over the part)
    coolant: bool = True
    cues: bool = True
    machine: bool = True
    enclosure: bool = True
    labels: bool = True
    colour_range: tuple | None = None     # fixed colour bar limits (display units); None: from the frame
    path: int = 0
    title: str | None = None
    finding_hold_s: float | None = None   # simulated seconds a finding stays labelled (None: from render.yaml)
    aspect: float = 16.0 / 9.0
    height_px: int = 720


@dataclass
class FrameInfo:
    tick: int
    time_s: float
    keyframe: int | None
    tool_number: int | None
    notes: list = field(default_factory=list)
    counts: dict = field(default_factory=dict)
    shadow_focus: tuple | None = None     # (lo, hi) the shadow map covers


def _inv(pose: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = pose[:3, :3].T
    out[:3, 3] = -pose[:3, :3].T @ pose[:3, 3]
    return out


def _density(material: dict | None) -> float | None:
    if not material:
        return None
    entry = material.get("density_kg_m3")
    if isinstance(entry, dict):
        if entry.get("value") is not None:
            return float(entry["value"])
        table = entry.get("table")
        if table:
            return float(table[0][1])
        return None
    return float(entry) if entry is not None else None


class FrameComposer:
    """Builds :class:`Scene` objects of one run (static data read once)."""

    def __init__(self, run: RunData, seed: int | None = None) -> None:
        self.run = run
        self.view = P.view()
        self.seed = int(seed if seed is not None else run.run.get("seed", run.project.get("seed", 0)) or 0)
        family = run.material_family()
        palette = self.view["material_srgb"]
        self.material_srgb = tuple(palette.get(family, palette["default"]))
        material = run.project.get("material") or {}
        self.density_kg_m3 = _density(material) or _density(db_item("materials", material.get("id")))
        self.thresholds = Q.thresholds(run.coolant())
        self.ticks = run.frame_ticks
        self.tips = {}
        self.findings = self._findings()
        stock = run.scene.get("stock") or {}
        self.pose_W_S = np.asarray(stock.get("pose_W_S", np.eye(4)), dtype=np.float64).reshape(4, 4)
        gravity = np.asarray(run.scene.get("gravity_M", [0.0, 0.0, -1.0]), dtype=np.float64)
        self.up_M = -gravity / float(np.linalg.norm(gravity))
        self._tool_trees: dict = {}
        self._static_tree = (None, None)

    # ------------------------------------------------------------------ tool motion
    def _path_arrays(self, path: int):
        if path not in self.tips:
            self.tips[path] = (self.run.tip_S(path), self.run.axis_T_S(path), self.run.spindle_angle(path))
        return self.tips[path]

    def tool_pose(self, tick: int, path: int = 0) -> np.ndarray:
        tips, axes, angles = self._path_arrays(path)
        if not len(tips):
            return np.eye(4)
        i = self.run.frame_index(tick)
        return SH.tool_pose(tips[i], axes[i], float(angles[i]))

    def follow_target(self, tick: int, path: int = 0) -> np.ndarray:
        tips, _, _ = self._path_arrays(path)
        if not len(tips):
            return np.zeros(3)
        window = int(round(float(self.view["follow_smoothing_s"]) / self.run.tick_s))
        return CAM.smoothed_target(self.ticks, tips, tick, window)

    # ------------------------------------------------------------------ findings
    def _findings(self) -> list[dict]:
        from .rundir import _json

        doc = _json(self.run.path / "findings.json")
        items = doc.get("findings", []) if isinstance(doc, dict) else (doc or [])
        return [f for f in items if isinstance(f, dict) and "time_s" in f]

    # ------------------------------------------------------------------ the scene
    def compose(self, tick: int, options: FrameOptions | None = None) -> tuple[Scene, CAM.Camera, FrameInfo]:
        o = options or FrameOptions()
        run = self.run
        info = FrameInfo(int(tick), tick * run.tick_s, None, None)
        scene = Scene(shadows=True)
        entry = run.keyframe_at(tick)
        k = int(entry["k"]) if entry is not None else None
        info.keyframe = k
        fields = run.fields(k) if k is not None else {}
        chips = run.chips(k) if (k is not None and o.chips) else {}
        pose_M_S, pose_known = run.pose_M_S(tick)
        pose_S_M = _inv(pose_M_S)
        pose_S_T = self.tool_pose(tick, o.path)
        # ---- the workpiece
        if k is not None:
            ipw = run.ipw(k)
            part_v, part_f, attrs = ipw.vertices, ipw.faces, ipw.attrs
        else:
            part_v, part_f, attrs = np.zeros((0, 3)), np.zeros((0, 3), np.int64), {}
        if len(part_f) == 0:
            stock = run.stock_mesh()
            if stock is not None:
                part_v, part_f = stock
            attrs = {}
        layer = L.evaluate(run, o.layer, k, tick, part_v, attrs, fields, chips) if o.layer else None
        vmin, vmax = (o.colour_range if o.colour_range is not None else
                      (self._frame_range(layer) if layer is not None else (0.0, 1.0)))
        part_temp = L.surface_temperature(fields, part_v) if (len(part_v) and o.cues) else None
        part_srgb = L.colours(layer.part if layer is not None else None, layer.spec if layer else None, vmin, vmax,
                              self.material_srgb, len(part_v)) if layer is not None else None
        emission = Q.emission(part_temp) if part_temp is not None and o.cues else None
        scene.add(RenderMesh("part", part_v, part_f, Material(srgb=self.material_srgb, cap=True),
                             vertex_srgb=part_srgb, vertex_emission=emission))
        info.counts["part_faces"] = int(len(part_f))
        # ---- components (machine, fixture, enclosure)
        for comp in run.components():
            if comp["kind"] in ("enclosure", "door", "window") and not o.enclosure:
                continue
            if comp["kind"] == "machine" and not o.machine:
                continue
            mesh = run.mesh(comp["mesh"])
            if mesh is None:
                continue
            v, f = mesh
            if comp["frame"] == "S":
                place = comp["pose"]
            elif comp["frame"] == "T":
                place = pose_S_T @ comp["pose"]
            else:
                if not pose_known:
                    info.notes.append(f"{comp['id']}: frame M without pose_M_S, not drawn")
                    continue
                place = pose_S_M @ comp["pose"]
            scene.add(RenderMesh(comp["id"], SH.transform(v, place), f, self._component_material(comp)))
        # ---- the tool
        number = run.tool_number_at(tick, o.path)
        info.tool_number = number
        tool_entry = run.tool_entry(number)
        tool_temp = L.tool_temperature(run, tick) if (o.cues or o.layer == "temperature") else None
        if tool_entry is not None:
            self._add_tool(scene, tool_entry, pose_S_T, layer, vmin, vmax, tool_temp if o.cues else None)
        cam = self.camera(tick, o, scene)
        info.shadow_focus = self.focus(cam, o, scene)
        # ---- chips, piles and coolant
        to_S = pose_S_M if pose_known else None
        if o.chips and chips:
            if to_S is None:
                info.notes.append("chips are in M and the run gives no pose_M_S: not drawn")
            else:
                info.counts["chips"] = self._add_chips(scene, chips, to_S, layer, vmin, vmax, o, cam)
        if o.chips and o.pile:
            grid = L.pile_grid(fields, chips)
            if grid is not None:
                info.counts["pile_cells"] = self._add_pile(scene, grid, chips, layer, vmin, vmax, k, pose_S_T,
                                                           part_v, part_f)
        jets_on = False
        if o.coolant and fields:
            jets_on = self._add_jets(scene, fields, to_S, pose_S_T, k, part_v, part_f, info)
        # ---- cues
        if o.cues:
            self._add_cues(scene, tick, part_v, part_temp, pose_S_T, tool_temp, jets_on, pose_S_M)
        # ---- section, labels, colour bar
        if o.section is not None:
            scene.planes = section_plane(o.section[0], o.section[1])[None, :]
        if layer is not None:
            scene.colorbars.append(ColorBar(layer.spec.title, layer.spec.unit, vmin, vmax, layer.spec.cmap()))
        if o.labels:
            self._labels(scene, tick, info, tool_entry, pose_S_T, o)
        return scene, cam, info

    # ------------------------------------------------------------------ parts of the scene
    def _component_material(self, comp: dict) -> Material:
        v = self.view
        kind = comp["kind"]
        if kind == "enclosure":
            return Material(srgb=tuple(v["enclosure_srgb"]), opacity=float(comp["opacity"] or v["enclosure_opacity"]),
                            cull_back=False, two_sided=True, clip=False, cast_shadow=False, receive_shadow=False,
                            specular=0.0)
        if kind in ("door", "window"):
            return Material(srgb=tuple(v["enclosure_srgb"]), opacity=float(comp["opacity"] or v["door_opacity"]),
                            cull_back=False, two_sided=True, clip=False, cast_shadow=False, receive_shadow=False,
                            specular=0.0)
        colour = {"fixture": v["fixture_srgb"], "spindle": v["holder_srgb"], "holder": v["holder_srgb"]}.get(
            kind, v["machine_srgb"])
        opacity = 1.0 if comp["opacity"] is None else float(comp["opacity"])
        cut = kind == "fixture"                  # section planes cut the part, the fixture and the chips only
        return Material(srgb=tuple(colour), opacity=opacity, cap=cut, clip=cut)

    def _add_tool(self, scene: Scene, entry: dict, pose_S_T: np.ndarray, layer, vmin, vmax, tool_temp) -> None:
        envelope = self.run.mesh(entry.get("envelope"))
        if envelope is None:
            return
        v_T, f = envelope
        colour = None
        if layer is not None and layer.tool is not None and np.isfinite(layer.tool):
            colour = tuple(C.apply_colormap(layer.spec.display([layer.tool]), vmin, vmax, layer.spec.cmap())[0])
        emission = None
        if tool_temp is not None and tool_temp > self.thresholds.glow_K:
            radius = 0.5 * float(entry.get("diameter_m") or (np.ptp(v_T[:, 0]) if len(v_T) else 0.0))
            weight = np.clip(1.0 - v_T[:, 2] / max(radius, 1e-9), 0.0, 1.0)
            emission = Q.emission([tool_temp])[0][None, :] * weight[:, None]
        scene.add(RenderMesh("tool", SH.transform(v_T, pose_S_T), f,
                             Material(srgb=colour or tuple(self.view["tool_srgb"]), clip=False), vertex_emission=emission))
        holder = self.run.mesh(entry.get("holder_envelope"))
        if holder is not None:
            scene.add(RenderMesh("holder", SH.transform(holder[0], pose_S_T), holder[1],
                                 Material(srgb=tuple(self.view["holder_srgb"]), clip=False)))

    def _chip_sizes(self, chips: dict, n: int) -> np.ndarray:
        aspect = np.asarray(self.view["chip_aspect"], dtype=np.float64)
        if "length_m" in chips and np.asarray(chips["length_m"]).size == n:
            length = np.asarray(chips["length_m"], dtype=np.float64).reshape(-1)
        elif "mass_kg" in chips and self.density_kg_m3:
            volume = np.asarray(chips["mass_kg"], dtype=np.float64).reshape(-1) / self.density_kg_m3
            t = np.cbrt(volume / (aspect[1] * aspect[2]))
            length = t * aspect[2]
        else:
            length = np.full(n, float(self.view["chip_default_length_m"]))
        length = np.where(np.isfinite(length) & (length > 0.0), length, float(self.view["chip_default_length_m"]))
        return 0.5 * length[:, None] * (aspect / aspect[2])[None, :]

    def _add_chips(self, scene: Scene, chips: dict, to_S: np.ndarray, layer, vmin, vmax, o: FrameOptions,
                   cam: CAM.Camera) -> int:
        pos_M = np.asarray(chips.get("pos_M_m", np.zeros((0, 3))), dtype=np.float64).reshape(-1, 3)
        n = len(pos_M)
        if n == 0:
            return 0
        pos = SH.transform(pos_M, to_S)
        vel = None
        if "vel_M_m_s" in chips and np.asarray(chips["vel_M_m_s"]).size == 3 * n:
            vel = np.asarray(chips["vel_M_m_s"], dtype=np.float64).reshape(-1, 3) @ to_S[:3, :3].T
        size = self._chip_sizes(chips, n)
        # never smaller than chip_min_size_px on screen
        pixel = np.array([cam.pixel_size_at(x, o.height_px) for x in pos])
        floor = 0.5 * float(self.view["chip_min_size_px"]) * pixel
        size = np.maximum(size, floor[:, None])
        seeds = Q.chip_orientation_seeds(self.seed, np.arange(n))
        verts, faces = SH.chips(pos, size, seeds, vel)
        if "colour_rgb" in chips and np.asarray(chips["colour_rgb"]).size == 3 * n:
            srgb = np.asarray(chips["colour_rgb"], dtype=np.float64).reshape(-1, 3) / 255.0
        else:
            srgb = np.broadcast_to(np.asarray(self.material_srgb), (n, 3))
        temps = np.asarray(chips.get("temp_K", np.full(n, np.nan)), dtype=np.float64).reshape(-1)
        if layer is not None and layer.chips is not None and len(layer.chips) == n:
            srgb = L.colours(layer.chips, layer.spec, vmin, vmax, self.material_srgb, n)
        glow = Q.emission(temps) if o.cues else np.zeros((n, 3))
        per_face = 8
        scene.add(RenderMesh("chips", verts, faces, Material(shading="flat", cull_back=True),
                             face_srgb=np.repeat(srgb, per_face, axis=0),
                             face_emission=np.repeat(glow, per_face, axis=0)))
        if o.cues:
            sparks = Q.sparks(pos, vel, temps)
            if sparks is not None:
                scene.sprites.append(sparks)
        return n

    def _trees(self, k, part_v, part_f, pose_S_T: np.ndarray):
        """BVH of the part and the static S-frame components (per keyframe), and the tool's (in T)."""
        from ..geom.bvh import build_tree
        from ..geom.mesh import Mesh

        key, tree = self._static_tree
        if key != k:
            verts, faces, offset = [], [], 0
            if len(part_f):
                verts.append(part_v)
                faces.append(part_f)
                offset = len(part_v)
            for comp in self.run.components():
                if comp["frame"] != "S" or comp["kind"] in ("enclosure", "door", "window"):
                    continue
                mesh = self.run.mesh(comp["mesh"])
                if mesh is None:
                    continue
                verts.append(SH.transform(mesh[0], comp["pose"]))
                faces.append(mesh[1] + offset)
                offset += len(mesh[0])
            tree = build_tree(Mesh(np.concatenate(verts), np.concatenate(faces))) if verts else None
            self._static_tree = (k, tree)
        return tree

    def _first_hit(self, origin, direction, tree, pose_S_T, tool_entry) -> float:
        from ..geom.bvh import ray_first_hit

        best = math.inf
        if tree is not None:
            hit = ray_first_hit(tree, origin, direction, 0.0, math.inf)
            if hit is not None:
                best = hit[0]
        tool_tree = self._tool_tree(tool_entry)
        if tool_tree is not None:
            inv = _inv(pose_S_T)
            o_T = inv[:3, :3] @ origin + inv[:3, 3]
            d_T = inv[:3, :3] @ direction
            hit = ray_first_hit(tool_tree, o_T, d_T, 0.0, math.inf)
            if hit is not None:
                best = min(best, hit[0])
        return best

    def _tool_tree(self, entry):
        if entry is None or not entry.get("envelope"):
            return None
        rel = entry["envelope"]
        if rel not in self._tool_trees:
            from ..geom.bvh import build_tree
            from ..geom.mesh import Mesh

            mesh = self.run.mesh(rel)
            self._tool_trees[rel] = build_tree(Mesh(mesh[0], mesh[1])) if mesh is not None else None
        return self._tool_trees[rel]

    def _add_jets(self, scene: Scene, fields: dict, to_S, pose_S_T, k, part_v, part_f, info: FrameInfo) -> bool:
        origins = np.asarray(fields.get("jet_origin_M_m", np.zeros((0, 3))), dtype=np.float64).reshape(-1, 3)
        if not len(origins):
            return False
        if to_S is None:
            info.notes.append("coolant jets are in M and the run gives no pose_M_S: not drawn")
            return True
        dirs = np.asarray(fields["jet_direction_M"], dtype=np.float64).reshape(-1, 3)
        n = len(origins)
        speed = np.asarray(fields.get("jet_speed_m_s", np.zeros(n)), dtype=np.float64).reshape(-1)
        flow = np.asarray(fields.get("jet_flow_m3_s", np.zeros(n)), dtype=np.float64).reshape(-1)
        half = np.asarray(fields.get("jet_half_angle_rad", np.zeros(n)), dtype=np.float64).reshape(-1)
        v = self.view
        tree = self._trees(k, part_v, part_f, pose_S_T)
        tool_entry = self.run.tool_entry(info.tool_number)
        drawn = 0
        for i in range(n):
            if flow[i] <= 0.0 or speed[i] <= 0.0:
                continue
            o_S = to_S[:3, :3] @ origins[i] + to_S[:3, 3]
            d_S = to_S[:3, :3] @ dirs[i]
            d_S = d_S / float(np.linalg.norm(d_S))
            length = self._first_hit(o_S, d_S, tree, pose_S_T, tool_entry)
            if not math.isfinite(length):
                length = float(v["jet_max_length_m"])
            r0 = math.sqrt(flow[i] / (math.pi * speed[i]))           # continuity: Q = v pi r0^2
            r1 = r0 + length * math.tan(max(half[i], 0.0))
            verts, faces, frac = SH.jet(o_S, d_S, length, r0, r1, int(round(float(v["jet_sides"]))))
            alpha = 1.0 - (1.0 - float(v["jet_end_opacity_fraction"])) * frac
            scene.add(RenderMesh(f"jet_{i}", verts, faces,
                                 Material(srgb=tuple(v["jet_srgb"]), opacity=float(v["jet_opacity"]), cull_back=False,
                                          two_sided=True, clip=False, cast_shadow=False, receive_shadow=False),
                                 vertex_alpha=alpha))
            drawn += 1
        info.counts["jets"] = drawn
        return drawn > 0

    def _add_pile(self, scene: Scene, grid: L.PileGrid, chips: dict, layer, vmin, vmax, k, pose_S_T, part_v,
                  part_f) -> int:
        tree = self._trees(k, part_v, part_f, pose_S_T)
        nu, nv = grid.height_m.shape
        eu, ev, ew = grid.axes[:, 0], grid.axes[:, 1], grid.axes[:, 2]
        floor = float(self.view["pile_min_height_m"])
        occupied = np.argwhere(np.isfinite(grid.height_m) & (grid.height_m > floor))
        if not len(occupied) or tree is None:
            return 0
        from ..geom.bvh import rays_first_hit

        top = None
        if len(part_v):
            top = float(np.max((part_v - grid.origin) @ ew))
        lift = (top if top is not None else 0.0) + 1.0
        centres = grid.origin + ((occupied[:, 0:1] + 0.5) * grid.cell_m) * eu + ((occupied[:, 1:2] + 0.5) * grid.cell_m) * ev
        starts = centres + lift * ew
        t, _ = rays_first_hit(tree, starts, -ew)
        base = np.full((nu, nv), np.nan)
        hit = np.isfinite(t)
        base[occupied[hit, 0], occupied[hit, 1]] = lift - t[hit]
        verts, faces, heights = SH.heightfield(grid.origin, eu, ev, ew, grid.cell_m, base, grid.height_m, floor)
        if not len(faces):
            return 0
        if "colour_rgb" in chips and np.asarray(chips["colour_rgb"]).size:
            colour = tuple(np.asarray(chips["colour_rgb"], dtype=np.float64).reshape(-1, 3).mean(axis=0) / 255.0)
        else:
            colour = self.material_srgb
        srgb = None
        if layer is not None and layer.spec.name == "chip_pile":
            srgb = L.colours(heights, layer.spec, vmin, vmax, colour, len(verts))
        scene.add(RenderMesh("chip_pile", verts, faces, Material(srgb=colour, two_sided=True, cull_back=False),
                             vertex_srgb=srgb))
        return int(len(faces) // 2)

    def _add_cues(self, scene: Scene, tick: int, part_v, part_temp, pose_S_T, tool_temp, coolant_on: bool,
                  pose_S_M) -> None:
        th = self.thresholds
        time_s = tick * self.run.tick_s
        up_S = pose_S_M[:3, :3] @ self.up_M
        spots = []
        max_spots = int(round(float(P.number(P.CUES, "max_hot_spots"))))
        lowest = min([x for x in (th.smoke_K, th.boil_K) if x is not None], default=None)
        if part_temp is not None and lowest is not None and coolant_on:
            hot = np.flatnonzero(np.isfinite(part_temp) & (part_temp > lowest))
            if len(hot):
                order = hot[np.lexsort((hot, -part_temp[hot]))][:max_spots]
                for i in order:
                    spots.append(Q.HotSpot(Q.source_key(self.seed, part_v[i]), part_v[i], float(part_temp[i])))
        if tool_temp is not None and coolant_on and lowest is not None and tool_temp > lowest:
            spots.append(Q.HotSpot(1, pose_S_T[:3, 3].copy(), float(tool_temp)))
        for kind, threshold in (("smoke", th.smoke_K), ("steam", th.boil_K)):
            sprites = Q.puffs(spots, time_s, threshold, kind, self.seed, up_S)
            if sprites is not None:
                scene.sprites.append(sprites)

    def _labels(self, scene: Scene, tick: int, info: FrameInfo, tool_entry, pose_S_T, o: FrameOptions) -> None:
        run = self.run
        time_s = tick * run.tick_s
        project = run.project
        title = o.title if o.title is not None else str(project.get("name", ""))
        block = run.block_at(tick)
        scene.title = title
        where = f"t = {time_s:.3f} s"
        if block is not None:
            where += f"   {block.get('op', '')} line {block.get('line', '')}: {block.get('block', '')}"
        scene.subtitle = where
        if tool_entry is not None:
            text = f"T{tool_entry.get('number')}"
            comment = self._tool_comment(tool_entry.get("number"))
            if comment:
                text += f" {comment}"
            scene.labels.append(Label(text, anchor_m=pose_S_T[:3, 3] + pose_S_T[:3, 2] * float(
                tool_entry.get("length_m") or 0.0), xy_px=(12, -12)))
        hold = (float(self.view["finding_label_hold_s"]) if o.finding_hold_s is None else float(o.finding_hold_s))
        size = float(self.view["label_font_size"])
        shown = [f for f in self.findings if float(f["time_s"]) <= time_s <= float(f["time_s"]) + hold]
        for row, finding in enumerate(shown[-FINDING_LABEL_ROWS:]):
            text = f"{finding.get('severity', '')}: {finding.get('message', '')}"
            colour = ERROR_LABEL_SRGB if finding.get("severity") in ("error", "fatal") else NOTICE_LABEL_SRGB
            scene.labels.append(Label(text, xy_px=(size, (FINDING_LABEL_FIRST_ROW + 1.6 * row) * size),
                                      srgb=colour, scale_with_height=True))

    def _tool_comment(self, number) -> str:
        for item in (self.run.project.get("tools") or {}).get("list", []) or []:
            if number is not None and int(item.get("number", -1)) == int(number):
                return str(item.get("comment") or "")
        return ""

    def _frame_range(self, layer: L.LayerValues) -> tuple[float, float]:
        values = []
        if layer.spec.name == "chip_pile":
            return L.run_range(self.run, "chip_pile")
        if layer.part is not None:
            values.append(np.asarray(layer.part, dtype=np.float64))
        if layer.tool is not None:
            values.append(np.array([layer.tool]))
        if layer.chips is not None:
            values.append(np.asarray(layer.chips, dtype=np.float64))
        if layer.spec.name == "coolant_reach":
            return 0.0, 100.0
        if not values:
            return (0.0, 1.0) if not layer.spec.symmetric else (-1.0, 1.0)
        return C.robust_range(layer.spec.display(np.concatenate(values)), symmetric=layer.spec.symmetric)

    # ------------------------------------------------------------------ cameras
    def focus(self, cam: CAM.Camera, o: FrameOptions, scene: Scene):
        """Region the shadow map covers: around the target of the follow camera (the view height on every side),
        the fitted box of the other cameras."""
        if o.camera == "follow":
            half = np.full(3, cam.view_height_m)
            return cam.target - half, cam.target + half
        half = np.full(3, 0.5 * cam.view_height_m)
        return cam.target - half, cam.target + half

    def camera(self, tick: int, o: FrameOptions, scene: Scene) -> CAM.Camera:
        """follow: the tool tip (smoothed); part: the workpiece; setup: workpiece, fixture and tool; overview:
        everything opaque (machine included)."""
        v = self.view
        if o.camera not in CAMERA_MODES:
            raise ValueError(f"camera {o.camera!r} is not one of {CAMERA_MODES}")
        if o.camera == "follow":
            cam = CAM.follow(self.follow_target(tick, o.path))
            if o.projection == CAM.ORTHOGRAPHIC:
                cam = CAM.Camera(cam.eye, cam.target, cam.up, CAM.ORTHOGRAPHIC, cam.fov_y_rad, cam.view_height_m,
                                 cam.near_m, cam.far_m)
            return cam
        fixture_ids = {str(item.get("id")) for item in self.run.scene.get("fixture", []) or []}
        if o.camera == "part":
            names = {"part"}
        elif o.camera == "setup":
            names = {"part", "tool", "holder"} | fixture_ids
        else:
            names = None
        boxes = [m.bounds() for m in scene.meshes if not m.transparent and (names is None or m.name in names)]
        boxes = [b for b in boxes if b is not None]
        if not boxes:
            return CAM.follow(self.follow_target(tick, o.path))
        lo = np.min([b[0] for b in boxes], axis=0)
        hi = np.max([b[1] for b in boxes], axis=0)
        direction = CAM.direction(float(v["overview_azimuth_rad"]), float(v["overview_elevation_rad"]))
        return CAM.fit(lo, hi, direction, projection=o.projection, aspect=o.aspect)
