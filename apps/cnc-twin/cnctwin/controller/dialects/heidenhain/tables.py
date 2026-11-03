"""HEIDENHAIN TNC 640 tables: condition keys -> NC error numbers, built-in texts, M functions, word lists.

Error numbers and texts are those of db/controllers/heidenhain-tnc640.yaml (HEIDENHAIN "NC Error
Messages", NC software xxxxxx-18, heidenhain-nc-errors-2023); the copies below are used only when the
database file is absent and are then marked ``source: builtin``. FN 14: ERROR = n maps n = 1000-1199 to
the predefined errors 280-03E8 + (n - 1000) (hexadecimal), other numbers to 280-0064 "FN 14: error
code %1" (db messages "FN 14").
"""

from __future__ import annotations

ALARM_CODES = {
    "syntax": "160-0003", "too_many_digits": "160-0003", "no_data_after_address": "160-0003",
    "illegal_minus": "160-0003", "illegal_decimal_point": "160-0003", "improper_address": "160-0003",
    "expression_format": "160-0003", "macro_format": "160-0003", "improper_g_code": "160-0003",
    "unsupported_word": "160-00CF", "undefined_m_code": "160-00C3", "multiple_codes": "160-0122",
    "duplicate_address": "1A0-0008", "illegal_axis": "160-001F", "illegal_plane": "1A0-0051",
    "program_not_found": "201-0804", "end_of_record": "280-0400", "sub_without_m99": "280-0400",
    "feed_zero": "160-0069", "feed_tooth": "160-00E0", "arc_no_end": "1A0-000B", "arc_direction": "1A0-000C",
    "arc_no_radius": "1A0-000D", "pole_missing": "1A0-000F", "pole_illegal": "160-0021",
    "circle_illegal": "1A0-0082", "radius_tolerance": "1A0-0084", "arc_end_point": "1A0-0084",
    "illegal_radius": "1A0-0083", "radius_tolerance_r": "1A0-0083", "circle_general": "1A0-00BE",
    "helix_illegal": "1A0-0081", "crc_interference": "1A0-0013", "crc_no_intersection": "1A0-0094",
    "crc_arc_startup": "1A0-0030", "crc_g39_motion": "1A0-0030", "comp_start": "1A0-0030", "comp_end": "1A0-0031",
    "comp_switch": "1A0-002D", "comp_tool_change": "1A0-0064", "appr_without_comp": "1A0-0071",
    "approach_failed": "1A0-0072", "departure_failed": "1A0-0073", "rounding_failed": "1A0-0033",
    "chamfer_failed": "1A0-0034", "chamfer_missing_move": "1A0-0035", "chamfer_not_g01": "1A0-0076",
    "chamfer_in_thread": "1A0-0076", "chamfer_too_large": "280-03FF", "zero_length": "1A0-012A",
    "cycle_undefined": "160-000B", "cycle_incomplete": "280-03F9", "cycle_missing": "280-0403",
    "slot_width": "280-0404", "pocket_small": "280-0405", "q202_missing": "280-0406",
    "q218_q219": "280-0408", "q220_large": "280-040B", "depth_sign": "280-0435", "thread_depth": "280-042E",
    "infeed_zero": "280-0442", "plunge_type": "280-0450", "cycle_tool_radius": "280-044F",
    "tap_depth_small": "1A0-0060", "tap_depth_sign": "1A0-0061", "tap_in_comp": "1A0-005F",
    "tap_no_spindle": "1A0-00B3", "pattern_missing": "160-00EE", "cycle_in_comp": "1A0-009E",
    "contour_intersects": "1A0-00AD", "depth_zero": "280-0488",
    "zero_divide": "160-0032", "overflow_float": "160-0032", "argument_out_of_range": "160-0032",
    "overflow_integer": "160-0032", "q_invalid": "160-0028", "q_undefined": "160-0028",
    "write_protected": "160-0028", "variable_out_of_range": "160-0028", "q_global_undefined": "160-00FE",
    "string_undeclared": "160-00AF", "fn14_no_text": "160-00C1", "fn18_number": "160-0089",
    "fn18_axis": "160-008A", "label_zero": "160-0018", "label_not_found": "160-001B",
    "sequence_number_not_found": "160-001B", "jump_not_found": "160-001B", "label_twice": "160-001C",
    "recursive_label": "160-00A8", "nesting": "160-00AA", "recursive_program": "160-00AB",
    "m2_in_subprogram": "160-0160", "tool_undefined": "160-00C5", "tool_zero": "280-05E4",
    "tool_definition": "280-05E8", "tool_axis_missing": "160-0056", "no_tool_axis": "1A0-000A",
    "tool_not_in_magazine": "160-00C5", "spindle_turning": "280-03E8", "spindle_not_rotating": "160-00A0",
    "spindle_rpm": "160-00DC", "spindle_zero": "160-00FC", "spindle_nmax": "160-00E3",
    "tilt_not_allowed": "1A0-00F5", "tilt_angle": "1A0-00F7", "tilt_no_angle": "1A0-00F8",
    "tilt_no_solution": "1A0-00FC", "tilt_vectors": "1A0-00FD", "tilt_points_close": "1A0-00FE",
    "tilt_points_line": "1A0-00FF", "tilt_no_rotary": "1A0-0100", "tilt_vectors_short": "1A0-0101",
    "tcp_not_allowed": "160-00D0", "tcp_comp": "1A0-004F", "tcp_zero_vector": "1A0-00B2",
    "rotation_not_permitted": "280-03EE", "shift_not_permitted": "280-03F1", "illegal_scale": "1A0-0086",
    "transform_illegal": "1A0-008A", "blank_contradictory": "280-044A", "wait_mismatch": "TW0008",
    "wait_illegal_p": "160-0003", "fk_undetermined": "TW0040", "fk_contradiction": "TW0041",
    "software_limit": "1A1-0010",
    # keys of the milling toolpaths shared with the Siemens dialect
    "cycle_reference_plane": "280-0435", "cycle_slot_violation": "280-0404", "cycle_infeed_width": "280-0442",
    "cycle_parameter": "280-03F9", "cycle_finish_allowance": "280-0405",
}

ALARM_TEXTS = {
    "160-0003": "Block format incorrect", "160-00CF": "NC command not available", "160-00C3": "M function not permitted",
    "160-0122": "Contradictory M functions", "1A0-0008": "Axis double programmed", "160-001F": "Wrong axis programmed",
    "1A0-0051": "Illegal axis programmed", "201-0804": "Program not found", "280-0400": "Program start undefined",
    "160-0069": "Feed rate is missing", "160-00E0": "FZ was programmed in conjunction with the number of teeth 0",
    "1A0-000B": "No arc end point programmed", "1A0-000C": "Arc block: No direction of rotation programmed",
    "1A0-000D": "No arc radius programmed in CR block", "1A0-000F": "Pole is missing",
    "160-0021": "Pole incorrectly defined", "1A0-0082": "Circle incorrectly programmed",
    "1A0-0083": "Circle incorrectly programmed", "1A0-0084": "Circle incorrectly programmed",
    "1A0-00BE": "General error in the calculation of a circle", "1A0-0081": "Invalid helix definition",
    "1A0-0013": "Tool radius too large", "1A0-0094": "Cannot calculate tool radius compensation",
    "1A0-0030": "Path comp wrongly started", "1A0-0031": "Path comp wrongly ended",
    "1A0-002D": "Compensation switching not permitted",
    "1A0-0064": "Tool change not allowed with tool compensation active",
    "1A0-0071": "APPR/DEP CT or APPR/DEP LN not allowed with inactive tool radius compensation",
    "1A0-0072": "Approach movement cannot be calculated", "1A0-0073": "Departure movement cannot be calculated",
    "1A0-0033": "Rounding cannot be calculated", "1A0-0034": "Chamfer cannot be calculated",
    "1A0-0035": "Two successive transitional elements not permitted",
    "1A0-0076": "Rounding or chamfer cannot be calculated", "280-03FF": "Rounding radius too large",
    "1A0-012A": "Contour element with length=0 programmed", "160-000B": "CYCL DEF not defined",
    "280-03F9": "CYCL DEF incomplete", "280-0403": "No fixed cycle defined", "280-0404": "Insufficient slot width",
    "280-0405": "Pocket too small", "280-0406": "Q202 not defined", "280-0408": "Q218 must be greater than Q219",
    "280-040B": "Q220 too large", "280-0435": "Check the depth sign", "280-042E": "Thread depth too large",
    "280-0442": "Enter an infeed not equal to 0", "280-0450": "Plunging type is not possible",
    "280-044F": "Tool radius too large", "1A0-0060": "Plunging depth in tapping cycle too small",
    "1A0-0061": "Plunging depth in tapping cycle has wrong sign",
    "1A0-005F": "Thread cutting not allowed with tool compensation active",
    "1A0-00B3": "No spindle available for tapping", "160-00EE": "No point pattern defined",
    "1A0-009E": "SL cycle call with active tool radius compensation", "1A0-00AD": "Contour intersects itself",
    "280-0488": "Machining depth defined as 0", "160-0032": "Arithmetical error", "160-0028": "Invalid Q parameter",
    "160-00FE": "Global Q parameter not defined", "160-00AF": "Undeclared string variable",
    "280-0064": "FN 14: error code %1", "160-00C1": "FN14 error without text",
    "160-0089": "Incorrect number for FN17/FN18", "160-008A": "Incorrect axis index for FN17/FN18",
    "160-0018": "Jump to label 0 not permitted", "160-001B": "Label number not found",
    "160-001C": "Label number already assigned", "160-00A8": "Recursive label call",
    "160-00AA": "Excessive program nesting", "160-00AB": "Recursive program call",
    "160-0160": "M2/M30 in subprogram", "160-00C5": "Tool not defined", "280-05E4": "Tool number 0 not permitted",
    "280-05E8": "Tool definition is missing", "160-0056": "Tool axis is missing", "1A0-000A": "No tool axis defined",
    "280-03E8": "Spindle must be turning", "160-00A0": "Spindle is not rotating", "160-00DC": "Wrong rpm",
    "160-00FC": "0 rpm not permitted", "160-00E3": "Spindle speed S is greater than NMAX from the tool table",
    "1A0-00F5": "Tilting the working plane is not allowed", "1A0-00F7": "Entered angle not permitted",
    "1A0-00F8": "Angle cannot be calculated", "1A0-00FC": "No solution in limited range",
    "1A0-00FD": "Vectors are not perpendicular", "1A0-00FE": "Points on a plane too close to each other",
    "1A0-00FF": "Points on a plane lie on a line", "1A0-0100": "No rotary axes available",
    "1A0-0101": "Vectors are too short", "160-00D0": "M128 / M129 not allowed here",
    "1A0-004F": "Tool compensation not switched off for 5-axis movement",
    "1A0-00B2": "Programmed direction vector is zero vector", "280-03EE": "Rotation not permitted",
    "280-03F1": "Datum shift not permitted", "1A0-0086": "Scaling factor invalid",
    "1A0-008A": "Illegal coordinate transformation", "280-044A": "Contradictory workpc. blank dim.",
    "1A1-0010": "Limit switch %1%2",
    "TW0008": "PATH WAIT CAN NEVER BE RELEASED",
    "TW0040": "FK CONTOUR ELEMENT NOT DETERMINED",
    "TW0041": "FK DATA CONTRADICTORY",
}

# M functions of the TNC 640 (db words.M) the twin executes; others raise 160-00C3 / TW0012
M_START = (3, 4, 8, 13, 14, 126, 127)
M_EXECUTED = frozenset({0, 1, 2, 3, 4, 5, 8, 9, 13, 14, 30, 89, 91, 92, 99, 126, 127, 128, 129, 140})
M_LISTED = frozenset({0, 1, 2, 3, 4, 5, 8, 9, 13, 14, 30, 89, 91, 92, 94, 97, 98, 99, 101, 102, 103, 107, 108,
                      109, 110, 111, 116, 117, 118, 120, 126, 127, 128, 129, 130, 136, 137, 138, 140, 141, 143, 144,
                      145, 148, 149, 197})

WORD_LIST = {
    "program": "BEGIN PGM / END PGM name MM|INCH, BLK FORM 0.1 / 0.2, * structure blocks, ; comments, ~ continuation",
    "tools": "TOOL CALL n|\"name\" X|Y|Z S F DL DR DR2, TOOL DEF n L R (or preselection), FAUTO",
    "paths": "L, CC, C (DR+ DR-), CR (R, DR), CT, RND R, CHF, LP / CP / CTP (PR PA IPR IPA, helix with IZ), "
             "R0 RL RR R+ R-, F FMAX FAUTO FU FZ, I-prefixed incremental coordinates, M91 M92",
    "approach": "APPR LT / LN / CT / LCT and the polar APPR PLT / PLN / PCT / PLCT (LEN CCA R), DEP LT / LN / CT / LCT",
    "fk": "FPOL, FL FLT FC FCT with X Y IX IY PR PA AN IAN LEN DR R CCX CCY ICCX ICCY CCPR CCPA (determined "
          "elements), CLSD+ / CLSD-, FSELECT",
    "cycles": "CYCL DEF 200 201 202 203 204 205 206 207 208 209 240 251 252 253 (Q parameters), 7 (datum shift), 8 (mirroring), "
              "10 (rotation), 11 (scaling), 26 (axis-specific scaling), 32 (tolerance), 247 (presetting); CYCL CALL, "
              "CYCL CALL POS, CYCL CALL PAT, M99, M89; PATTERN DEF POS ROW PAT FRAME CIRC PITCHCIRC (ROTX ROTY shear)",
    "parameters": "Q0-Q1999 (Q100-Q199 read only), QL0-QL499, QR0-QR499, QS0-QS1999; FN 0 - FN 18 (FN 9-12 jumps, FN 14 "
                  "ERROR, FN 16 F-PRINT, FN 17 SYSWRITE ID50, FN 18 SYSREAD), formulas with + - * / ^ ( ) SIN COS TAN "
                  "ASIN ACOS ATAN SQ SQRT LN LOG EXP ABS INT FRAC SGN NEG PI %; QS = \"text\" / QS / STRING FORMULA "
                  "|| / TOCHAR / SUBSTR, TONUMB INSTR STRLEN STRCOMP",
    "program_flow": "LBL n / \"name\" / LBL 0, CALL LBL n REP m, CALL PGM name, SEL PGM / CALL SELECTED PGM, "
                    "SEL CYCLE and CYCL DEF 12 PGM CALL (program as cycle), STOP, M0 M1 M2 M30",
    "tilting": "PLANE SPATIAL / PROJECTED / EULER / VECTOR / POINTS / RELATIV / AXIAL / RESET with MOVE (DIST F) / "
               "TURN (F) / STAY and SEQ+ / SEQ-; M128 F, M129, FUNCTION TCPM, FUNCTION RESET TCPM",
    "spindle_coolant": "S, M3 M4 M5 M8 M9 M13 M14; M140 MB distance / MB MAX [F] (retract along the tool axis); "
                       "M126 / M127 (shortest path of modulo rotary axes on / off); other listed M functions TW0012",
    "iso": "%name G71 / G70 header, N blocks with *, G00 G01 G02 G03 G05 G06 G10-G16 (polar), G17-G19, G24 G25 (chamfer, "
           "round), G28 (mirror), G29 (pole), G30 G31 (blank), G38 (stop), G40-G42, G54 (datum shift), G72 (scaling), "
           "G73 (rotation), G79 (cycle call), G90 G91, G98 L (label), L n,m (label call), G99 (tool definition), "
           "G200-G209 G240 G251-G253 (cycles), I J K (pole), T, D00-D18, M; G26 / G27 (tangential approach and "
           "departure) raise TW0012",
}

# the TNC 640 cycle numbers (heidenhain-tnc640-cycles-2023 pp. 790-792, db words.cycles): the ones the
# dialect does not model raise TW0012, numbers the control does not have 160-000B
TNC640_CYCLES = frozenset({7, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 32, 39, 200,
                           201, 202, 203, 204, 205, 206, 207, 208, 209, 220, 221, 224, 225, 232, 233, 238, 239, 240,
                           241, 247, 251, 252, 253, 254, 256, 257, 258, 262, 263, 264, 265, 267, 270, 271, 272, 273,
                           274, 275, 276, 277, 285, 286, 287, 291, 292, 1271, 1272, 1273, 1274, 1278, 1281, 1282})

__all__ = ["ALARM_CODES", "ALARM_TEXTS", "M_START", "M_EXECUTED", "M_LISTED", "WORD_LIST", "TNC640_CYCLES"]
