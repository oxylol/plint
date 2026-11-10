

CONTACT_BISECTIONS = 60         # halvings of the contact-temperature bracket (below 1e-12 K for any bracket)


@dataclass(frozen=True)
class PrimaryState:
    """The primary-zone state the edge's reference shear flow stress k_ref is taken at (equivalent strain, already
    limited to ``edge_strain_cap``; equivalent strain rate; temperature)."""

    strain: float
    strain_rate_1_s: float
    temp_K: float


def contact_strain_rate(speed_m_s: float, arc_m: float) -> float:
    """Equivalent strain rate of the material pressed under the edge: the velocity V between the tool face and the
    material below the deforming zone taken up across the depth of the ploughed zone, delta =
    ``ploughing_zone_depth_ratio`` x arc (Prandtl's field under the contact, Hill 1950; the ploughing slip-line field of
    Waldorf, DeVor & Kapoor 1998), shear rate V / delta, equivalent rate V / (sqrt(3) delta)."""
    delta = P.number(P.CONTACT, "ploughing_zone_depth_ratio") * float(arc_m)
    if not (delta > 0.0 and speed_m_s > 0.0):
        return 0.0
    return float(speed_m_s) / (math.sqrt(3.0) * delta)


def contact_shear_flow_stress(work: WorkMaterial, k_ref_Pa: float, primary: "PrimaryState | None",
                              strain_rate_1_s: float, temp_K: float) -> float:
    """Shear flow stress of the work at a contact's own state: k_ref (the primary-zone state ``primary``) times the
    Johnson-Cook ratio sigma(eps, epsdot_c, T_c) / sigma(eps, epsdot_ref, T_ref) at the same (capped) strain, with
    the kernels' clamps (T* to [0, max_jc_Tstar], the rate ratio floor); k_ref without a flow-stress law or state."""
    jc = work.jc
    if jc is None or primary is None or not k_ref_Pa > 0.0:
        return float(k_ref_Pa)
    ox = P.get(P.OXLEY)
    t_max, r_min = float(ox["max_jc_Tstar"]), float(ox["min_jc_rate_ratio"])
    ref = K.jc_flow_stress(primary.strain, primary.strain_rate_1_s, primary.temp_K, jc, t_max, r_min)
    if not ref > 0.0:
        return float(k_ref_Pa)
    rate = strain_rate_1_s if strain_rate_1_s > 0.0 else primary.strain_rate_1_s
    return float(k_ref_Pa * K.jc_flow_stress(primary.strain, rate, temp_K, jc, t_max, r_min) / ref)


def solidus_K(work: WorkMaterial) -> float | None:
    """The work's solidus (else liquidus, else the melting temperature of its flow-stress law): above it the work
    carries no shear stress."""
    for value in (getattr(work, "solidus_K", None), getattr(work, "liquidus_K", None)):
        if value is not None and float(value) > 0.0:
            return float(value)
    if work.jc is not None and float(work.jc[abi.MAT_TMELT_K]) > 0.0:
        return float(work.jc[abi.MAT_TMELT_K])
    return None


def softened_flank_contact(work, tool, edge_radius_m: float, rake_rad: float, incl_rad: float, depth_m: float,
                           h_min_m: float, k_ref_Pa: float, primary: "PrimaryState | None", mu: float, VB_m: float,
                           land_factor: float, speed_m_s: float, contact_width_m: float | None, work_surface_K: float,
                           tool_flank_K: float | None, contact_time_s: float | None = None, *,
                           edge_t_scale: float = 1.0, indentation: bool = False) -> tuple[float, float, float, dict]:
    """(shear flow stress of the work in the flank contact, share of the contact's heat entering the tool, contact
    length, details) with the flow stress at the contact's own temperature and strain rate.

    The edge pressure (2 + pi) k, the friction min(mu p, k) on the arc and the wear land's stresses (``land_factor``
    x k: Smithey's mean friction stress per unit k) are all linear in k, so the contact's friction and plastic-work
    fluxes are s (q_f, q_d) with s = k_c / k_ref and (q_f, q_d) those at k_ref (:func:`flank_contact_heat`). Equal mean
    temperatures of the two faces (Blok 1937, the plastic work a source in the work, g its surface factor) give the
    contact temperature

        T(s) = (M_w T_t + M_t T_ws + M_w M_t s (q_f + g q_d)) / (M_w + M_t)     (T_ws + M_w s (q_f + g q_d) adiabatic)

    (the equal-temperature solution itself, also where the heat share :func:`flank_contact_heat` books is clamped to
    [0, 1]), increasing in s; k_c = :func:`contact_shear_flow_stress` at T and at the contact's strain rate
    (:func:`contact_strain_rate`) falls with T, so T = T(s(T)) has one root (bisection, fixed steps). Johnson-Cook's
    thermal term is bounded by the work's solidus (:func:`solidus_K`), above which the work carries no stress: when
    the flow stress at the solidus would still heat the contact beyond it, the contact sits at the solidus with the
    stress that keeps it there (T(s) = T_solidus). Returns flank_contact_heat's share, length and details at k_c,
    the details extended by ``k_contact_Pa``, ``k_ref_Pa``, ``softening`` (s), ``T_flow_K`` (the temperature the
    flow stress is taken at), ``strain_rate_1_s`` and ``solidus_bound``."""
    v = float(speed_m_s)
    vb = max(float(VB_m), 0.0)
    share, l_f, info = flank_contact_heat(work, tool, edge_radius_m, rake_rad, incl_rad, depth_m, h_min_m, k_ref_Pa,
                                          mu, vb, land_factor * k_ref_Pa, v, contact_width_m, work_surface_K,
                                          tool_flank_K, contact_time_s, edge_t_scale=edge_t_scale,
                                          indentation=indentation)
    details = {"k_contact_Pa": float(k_ref_Pa), "k_ref_Pa": float(k_ref_Pa), "softening": 1.0,
               "T_flow_K": float("nan"), "strain_rate_1_s": 0.0, "solidus_bound": False}
    q_f, q_d = info["q_friction_W_m2"], info["q_deformation_W_m2"]
    if work.jc is None or primary is None or not (k_ref_Pa > 0.0 and l_f > 0.0 and v > 0.0 and q_f + q_d > 0.0):
        info.update(details)
        return float(k_ref_Pa), share, l_f, info
    t_ws = float(work_surface_K)
    adiabatic = tool is None or tool_flank_K is None
    if adiabatic or not info["M_w"] > 0.0:
        width = contact_width_m or P.number(P.CONTACT, "default_contact_aspect") * l_f
        m_w = H.work_side_resistance(l_f, width, v, work, t_ws)
        g = 1.0
        if q_d > 0.0 and info["arc_m"] > 0.0:
            peclet = v * 0.5 * l_f / (2.0 * work.diffusivity(t_ws))
            g = H.layer_source_factor(peclet, P.number(P.CONTACT, "ploughing_zone_depth_ratio") * info["arc_m"] / l_f)
        if adiabatic:
            base, slope = t_ws, m_w * (q_f + g * q_d)
        else:
            m_t = H.tool_side_resistance(l_f, float(tool_flank_K), tool, width, contact_time_s)
            base = (m_w * float(tool_flank_K) + m_t * t_ws) / (m_w + m_t)
            slope = m_w * m_t * (q_f + g * q_d) / (m_w + m_t)
    else:
        m_w, m_t, g = info["M_w"], info["M_t"], info["g"]
        base = (m_w * float(tool_flank_K) + m_t * t_ws) / (m_w + m_t)
        slope = m_w * m_t * (q_f + g * q_d) / (m_w + m_t)
    rate = contact_strain_rate(v, info["arc_m"] if info["arc_m"] > 0.0 else l_f)

    def soft(temp_K: float) -> float:
        return contact_shear_flow_stress(work, k_ref_Pa, primary, rate, temp_K) / k_ref_Pa

    cap = solidus_K(work)
    bound = False
    if cap is not None and base + slope * soft(cap) > cap:
        s = max((cap - base) / slope, 0.0) if slope > 0.0 else 0.0
        temp = cap
        bound = True
    else:
        lo = base
        hi = base + slope * soft(lo)
        if cap is not None:
            hi = min(hi, cap)
        for _ in range(CONTACT_BISECTIONS):
            mid = 0.5 * (lo + hi)
            if base + slope * soft(mid) > mid:
                lo = mid
            else:
                hi = mid
        temp = 0.5 * (lo + hi)
        s = soft(temp)
    k_c = s * k_ref_Pa
    share, l_f, info = flank_contact_heat(work, tool, edge_radius_m, rake_rad, incl_rad, depth_m, h_min_m, k_c, mu, vb,
                                          land_factor * k_c, v, contact_width_m, work_surface_K, tool_flank_K,
                                          contact_time_s, edge_t_scale=edge_t_scale, indentation=indentation)
    info.update(k_contact_Pa=float(k_c), k_ref_Pa=float(k_ref_Pa), softening=float(s), T_flow_K=float(temp),
                strain_rate_1_s=float(rate), solidus_bound=bound)
    return float(k_c), share, l_f, info
