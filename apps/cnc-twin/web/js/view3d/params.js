/* Settings of the 3D view in SI: written from db/models/view3d.yaml (units, sources and ranges are
 * there) by tests/unit/view3d/sync_params.py. Do not edit: change the YAML file and run the script. */

export const PARAMS = Object.freeze({
  "attribute_sample_bricks": 24,
  "brick_cache_bytes": 100663296,
  "brick_cache_bytes_touch": 33554432,
  "cap_srgb": [
    0.82,
    0.32,
    0.28
  ],
  "chip_aspect": [
    1.0,
    5.0,
    12.0
  ],
  "chip_curl_rad": 2.6179938779914944,
  "chip_default_length_m": 0.0012,
  "chip_default_srgb": [
    0.78,
    0.79,
    0.8
  ],
  "chunk_bricks": 4,
  "crease_angle_rad": 0.6108652381980153,
  "deflected_tool_opacity": 0.55,
  "deflected_tool_srgb": [
    0.92,
    0.41,
    0.2
  ],
  "deflection_exaggeration_default": 1000,
  "deflection_window_s": 1.0,
  "door_opacity": 0.22,
  "enclosure_opacity": 0.1,
  "enclosure_srgb": [
    0.78,
    0.82,
    0.86
  ],
  "far_scene_factor": 4,
  "field_sample_keyframes": 12,
  "fill_light_intensity": 1.2,
  "fixture_cap_srgb": [
    0.5,
    0.42,
    0.62
  ],
  "fixture_srgb": [
    0.34,
    0.38,
    0.44
  ],
  "follow_smoothing_s": 0.12,
  "follow_view_height_m": 0.07,
  "fov_rad": 0.5235987755982988,
  "frames_chunk_records": 4096,
  "frames_chunks_kept": 8,
  "framing_margin": 1.15,
  "holder_srgb": [
    0.36,
    0.38,
    0.42
  ],
  "jet_end_opacity_fraction": 0.5,
  "jet_max_length_m": 0.6,
  "jet_opacity": 0.45,
  "jet_samples": 48,
  "jet_sides": 12,
  "key_light_azimuth_rad": 0.6108652381980153,
  "key_light_elevation_rad": 0.8726646259971648,
  "key_light_intensity": 2.0,
  "keyframes_cached": 8,
  "layer_unlit_fraction": 0.35,
  "lod_cell_factors": [
    4,
    16
  ],
  "machine_srgb": [
    0.56,
    0.6,
    0.62
  ],
  "material_srgb": {
    "aluminium": [
      0.8,
      0.81,
      0.83
    ],
    "brass": [
      0.82,
      0.7,
      0.4
    ],
    "cast_iron": [
      0.45,
      0.45,
      0.46
    ],
    "composite": [
      0.24,
      0.24,
      0.26
    ],
    "copper": [
      0.85,
      0.55,
      0.4
    ],
    "default": [
      0.75,
      0.75,
      0.76
    ],
    "foam": [
      0.93,
      0.93,
      0.9
    ],
    "magnesium": [
      0.76,
      0.77,
      0.78
    ],
    "nickel": [
      0.64,
      0.63,
      0.6
    ],
    "polymer": [
      0.88,
      0.88,
      0.84
    ],
    "stainless": [
      0.68,
      0.69,
      0.7
    ],
    "steel": [
      0.6,
      0.61,
      0.63
    ],
    "titanium": [
      0.58,
      0.57,
      0.55
    ],
    "tool_steel": [
      0.56,
      0.57,
      0.6
    ],
    "wood": [
      0.72,
      0.56,
      0.38
    ]
  },
  "max_antialias_pixel_ratio": 1.5,
  "max_bricks_per_file_read": 6,
  "max_deflection_exaggeration": 100000,
  "max_lod_cluster_px": 1.5,
  "max_lod_cluster_px_touch": 2.5,
  "max_overview_frames": 20000,
  "max_parallel_requests": 4,
  "max_pixel_ratio": 2,
  "max_pixel_ratio_touch": 1.5,
  "max_shadow_miss_m": 0.02,
  "max_tap_s": 0.5,
  "measure_marker_px": 8,
  "measure_srgb": [
    0.38,
    0.31,
    0.84
  ],
  "min_chip_size_px": 1.5,
  "min_circle_sine": 0.02,
  "min_deflection_exaggeration": 1,
  "min_pile_height_m": 0.0001,
  "min_surface_field_cos": 0.5,
  "min_zone_radius_m": 0.001,
  "near_fraction": 0.002,
  "nozzle_length_m": 0.025,
  "nozzle_radius_m": 0.004,
  "orbit_damping": 0.15,
  "pick_tolerance_px": 6,
  "pose_chain_tolerance_m": 0.0001,
  "pose_chain_tolerance_rad": 0.001,
  "prefetch_keyframes": 2,
  "rebuild_budget_s": 0.008,
  "refresh_period_s": 3.0,
  "shadow_opacity": 0.35,
  "spray_drop_size_m": 0.0008,
  "spray_drops_per_jet": 120,
  "spray_life_s": 0.3,
  "spray_speed_fraction": 0.3,
  "standard_gravity_m_s2": 9.80665,
  "tool_sides": 32,
  "tool_srgb": [
    0.5,
    0.52,
    0.56
  ],
  "tool_zone_height_ratio": 1.0,
  "view_azimuth_rad": -0.9599310885968813,
  "view_elevation_rad": 0.5585053606381855,
  "zone_radius_factor": 1.5
});
