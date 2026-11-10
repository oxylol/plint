"""Import to switch the contact softening off (the code before the change) for before/after comparisons."""
from cnctwin.mechanics import coefficients as CO

def _unsoftened(work, tool, edge_radius_m, rake_rad, incl_rad, depth_m, h_min_m, k_ref_Pa, primary, mu, VB_m,
                land_factor, speed_m_s, contact_width_m, work_surface_K, tool_flank_K, contact_time_s=None, *,
                edge_t_scale=1.0, indentation=False):
    share, l_f, info = CO.flank_contact_heat(work, tool, edge_radius_m, rake_rad, incl_rad, depth_m, h_min_m, k_ref_Pa,
                                             mu, VB_m, land_factor * k_ref_Pa, speed_m_s, contact_width_m,
                                             work_surface_K, tool_flank_K, contact_time_s, edge_t_scale=edge_t_scale,
                                             indentation=indentation)
    info.update(k_contact_Pa=k_ref_Pa, softening=1.0, T_flow_K=float("nan"), solidus_bound=False)
    return float(k_ref_Pa), share, l_f, info

CO.softened_flank_contact = _unsoftened
