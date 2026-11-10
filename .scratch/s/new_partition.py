def _flank_partition(work, tool, group: EdgeGroup, state: SlowState, h: np.ndarray, h_min: float, k_ref: float,
                     primary: "PrimaryState | None", mu: float, edge_t_scale, land_factor: float, work_surface_K,
                     ref_surface_K: float, ref_edge_t_scale: float = 1.0, *, indentation=False,
                     rubbing: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """(share of the flank contact's heat entering the tool, flank contact length, softening of the contact's flow
    stress at each h node; softening of the group's reference contact).

    The heat the hot loop books as flank (wear land) and ploughing (edge) heat - thermal splits both with this share
    and deposits the tool's part over the contact length (docs/models/thermal.md 1) - is, per unit edge length: the
    friction on the wear land (tau_land VB V) and on the edge arc below the stagnation point (the friction part of
    the edge force, contact.contact_power_parts), and the plastic work of the material pressed under the edge (its
    pressure part). The share follows from equal mean temperatures of tool and work over the contact VB + arc
    (heat.flank_heat_to_tool: Blok 1937), the plastic work being a heat source inside the work, uniform to the depth
    of the ploughed zone (ploughing_zone_depth_ratio x arc). The work's flow stress in the contact is taken at the
    contact's own temperature and strain rate (:func:`softened_flank_contact`, a fixed point): the softening s
    (k_contact / ``k_ref``) per node scales the node's edge, ploughing and indentation forces and the land's
    stresses, and the share is the one of the softened heat. ``edge_t_scale``: the calibration's factor on the
    tangential edge force per node (1 without calibration); ``land_factor``: the land's mean friction stress per unit
    k (calibration included); ``indentation`` (bool or per node): the node's forces are the indentation contact (the
    rake face in contact, depth h). The reference contact (the group's scalars: Ke, the land stresses, K_sp) is the
    edge contact at depth h_min with the work surface at ``ref_surface_K`` (the shear plane's temperature of the
    primary-zone state k_ref is taken at) and the edge factor ``ref_edge_t_scale``."""
    h = np.atleast_1d(np.asarray(h, dtype=float))
    n = h.shape[0]
    surface = np.broadcast_to(np.atleast_1d(np.asarray(work_surface_K, dtype=float)), (n,))
    scale = np.broadcast_to(np.atleast_1d(np.asarray(edge_t_scale, dtype=float)), (n,))
    indent = np.broadcast_to(np.atleast_1d(np.asarray(indentation, dtype=bool)), (n,))
    frac = np.zeros(n)
    length = np.zeros(n)
    soft = np.ones(n)
    flank_temp = state.tool_flank_temp_K if state.tool_flank_temp_K is not None else state.tool_temp_K
    vb = max(float(state.VB_m), 0.0)
    v = float(state.speed_m_s)
    cache: dict = {}

    def contact(depth, indented, t_scale, t_ws):
        key = (depth, indented, t_scale, t_ws)
        if key not in cache:
            k_c, share, l_f, _ = softened_flank_contact(
                work, tool, group.edge_radius_m, group.rake_rad, group.incl_rad, depth, h_min, k_ref, primary, mu, vb,
                land_factor, v, state.contact_width_m, t_ws, flank_temp, state.contact_time_s, edge_t_scale=t_scale,
                indentation=indented)
            cache[key] = (share, l_f, k_c / k_ref if k_ref > 0.0 else 1.0)
        return cache[key]

    for k in range(n):
        depth = h_min if (rubbing or h[k] >= h_min) and not indent[k] else float(h[k])
        frac[k], length[k], soft[k] = contact(depth, bool(indent[k]), float(scale[k]), float(surface[k]))
    group_indent = bool(np.all(indent)) and n > 0
    soft_ref = contact(h_min, group_indent, float(ref_edge_t_scale), float(ref_surface_K))[2] if h_min > 0.0 else 1.0
    return frac, length, soft, float(soft_ref)


def flank_contact_heat(work, tool, edge_radius_m: float, rake_rad: float, incl_rad: float, depth_m: float,
