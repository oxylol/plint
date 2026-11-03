"""SINUMERIK 840D sl tables: G functions with their groups, power-on states, alarm numbers and texts.

G group numbers are Siemens' (Programming Manual Fundamentals, list of G functions; db/controllers/
siemens-840d-sl.yaml words.G). ``kind`` is the handler's semantic key. Alarm texts are those of the
controller database (siemens-802dsl-diag-2011, the numbers checked against the 840D sl list); the
built-in copies below are used only when the database file is absent (tests, a machine without it)
and are marked ``source: builtin``.
"""

from __future__ import annotations

# code -> (group, kind, arg); group numbers as Siemens lists them
G_FUNCTIONS: dict[str, tuple[int, str, str]] = {
    "G0": (1, "motion", "rapid"), "G1": (1, "motion", "linear"), "G2": (1, "motion", "cw"),
    "G3": (1, "motion", "ccw"), "CIP": (1, "motion", "cip"), "CT": (1, "motion", "ct"),
    "G33": (1, "motion", "thread"), "G331": (1, "motion", "tap"), "G332": (1, "motion", "tap_return"),
    "G4": (2, "dwell", ""), "G63": (2, "tapping_once", ""), "G74": (2, "reference", ""), "G75": (2, "fixed_point", ""),
    "G25": (3, "lower_limit", ""), "G26": (3, "upper_limit", ""), "G110": (3, "pole", "current"),
    "G111": (3, "pole", "zero"), "G112": (3, "pole", "pole"), "G58": (3, "axial_offset", "abs"),
    "G59": (3, "axial_offset", "add"),
    "G17": (6, "plane", "XY"), "G18": (6, "plane", "ZX"), "G19": (6, "plane", "YZ"),
    "G40": (7, "comp", "G40"), "G41": (7, "comp", "G41"), "G42": (7, "comp", "G42"),
    "G500": (8, "wcs", "G500"), "G54": (8, "wcs", "G54"), "G55": (8, "wcs", "G55"), "G56": (8, "wcs", "G56"),
    "G57": (8, "wcs", "G57"),
    "G53": (9, "suppress", "G53"), "G153": (9, "suppress", "G153"), "SUPA": (9, "suppress", "SUPA"),
    "G60": (10, "exact", "exact_stop"), "G64": (10, "exact", "blend"), "G641": (10, "exact", "blend_adis"),
    "G642": (10, "exact", "blend_tol"), "G643": (10, "exact", "blend_tol"), "G644": (10, "exact", "blend_dyn"),
    "G645": (10, "exact", "blend_tol"),
    "G9": (11, "exact_once", ""),
    "G601": (12, "stop_window", "fine"), "G602": (12, "stop_window", "coarse"), "G603": (12, "stop_window", "ipo"),
    "G70": (13, "units", "inch"), "G71": (13, "units", "mm"), "G700": (13, "units", "inch"),
    "G710": (13, "units", "mm"),
    "G90": (14, "distance", "G90"), "G91": (14, "distance", "G91"),
    "G93": (15, "feed", "inverse_time"), "G94": (15, "feed", "per_min"), "G95": (15, "feed", "per_rev"),
    "G96": (15, "feed", "css"), "G97": (15, "feed", "css_off"),
    "CFC": (16, "feed_reference", "CFC"), "CFTCP": (16, "feed_reference", "CFTCP"),
    "CFIN": (16, "feed_reference", "CFIN"),
    "NORM": (17, "approach", "NORM"),
    "G450": (18, "corner", "arc"), "G451": (18, "corner", "intersection"),
    "BRISK": (21, "accel", "brisk"), "SOFT": (21, "accel", "soft"), "DRIVE": (21, "accel", "drive"),
    "FFWOF": (24, "feedforward", "off"), "FFWON": (24, "feedforward", "on"),
    "WALIMON": (28, "walim", "on"), "WALIMOF": (28, "walim", "off"),
    "DIAMOF": (29, "diameter", "off"), "DIAMON": (29, "diameter", "on"), "DIAM90": (29, "diameter", "g90"),
    "COMPOF": (30, "compressor", "off"), "COMPON": (30, "compressor", "on"), "COMPCURV": (30, "compressor", "on"),
    "COMPCAD": (30, "compressor", "on"), "COMPSURF": (30, "compressor", "on"),
}
for _n in range(505, 600):
    G_FUNCTIONS[f"G{_n}"] = (8, "wcs", f"G{_n}")

# functions of the 840D sl that this build does not model: they raise TW0012, never pass silently
UNMODELLED = frozenset({
    "ASPLINE", "BSPLINE", "CSPLINE", "POLY", "G34", "G35", "OEMIPO1", "OEMIPO2", "INVCW", "INVCCW", "G147",
    "G148", "G247", "G248", "G347", "G348", "G340", "G341", "KONT", "KONTT", "KONTC", "G931", "G942", "G952",
    "G962", "G972", "G973", "TRANSMIT", "TRACYL", "TRAANG", "TRACON", "CUT3DC", "CUT3DF", "CUT3DFS",
    "CUT3DFF", "CUT2D", "CUT2DF", "ORIWKS", "ORIMKS", "ORIAXES", "ORIVECT", "CASE", "MEAS", "MEAW",
    "POS", "POSA", "SPOSA", "WAITP", "WAITS", "WAITE", "WAITMC", "INIT", "START", "SETINT", "ENABLE",
    "DISABLE", "CLEARM", "SETM", "PCALL", "ISOCALL", "MEASURE", "TOROT", "TOFRAME", "TOROTOF", "PAROT",
    "PAROTOF", "ROTS", "AROTS", "CROTS", "CHKDNO", "OFFN", "CUTCONON", "CUTCONOF", "SPATH", "UPATH",
})

POWER_ON_MILL = {"motion": "G0", "plane": "G17", "7": "G40", "wcs": "G500", "exact": "G60", "stop_window": "G601",
                 "units": "G71", "distance": "G90", "feed": "G94", "feed_reference": "CFC", "approach": "NORM",
                 "corner": "G450", "accel": "BRISK", "feedforward": "FFWOF", "walim": "WALIMON",
                 "diameter": "DIAMOF", "compressor": "COMPOF", "feed_mode": "per_min"}
POWER_ON_LATHE = dict(POWER_ON_MILL, plane="G18", diameter="DIAMON", feed="G95", feed_mode="per_rev")

# keywords that are statements or block functions (not G functions)
KEYWORDS = frozenset({"TRANS", "ATRANS", "ROT", "AROT", "SCALE", "ASCALE", "MIRROR", "AMIRROR", "STOPRE",
                      "TRAORI", "TRAFOOF", "RET", "MCALL", "CALL", "M17"})
FRAME_WORDS = ("TRANS", "ATRANS", "ROT", "AROT", "SCALE", "ASCALE", "MIRROR", "AMIRROR")

# extended addresses written NAME=value
ADDRESSES = frozenset({"I", "J", "K", "I1", "J1", "K1", "CR", "AR", "AP", "RP", "TURN", "F", "S", "T", "D", "H",
                       "CHF", "CHR", "RND", "RNDM", "ADIS", "ADISPOS", "LIMS", "SPOS", "RPL", "SF", "FP", "M",
                       "P", "L", "E", "A3", "B3", "C3"})

# condition key -> 840D sl alarm number
ALARM_CODES = {
    "syntax": "12080", "too_many_digits": "12080", "no_data_after_address": "12080", "illegal_minus": "12080",
    "illegal_decimal_point": "12080", "improper_address": "12080", "expression_format": "12080",
    "macro_format": "12080", "nc_macro_same_block": "12080", "block_interpretation": "12110",
    "undefined_name": "12550", "duplicate_address": "12000", "same_g_group": "12060", "improper_g_code": "12470",
    "unexpected_parameter": "12090", "structure_nesting": "12640", "structure_depth": "12641",
    "def_position": "14500", "proc_missing": "14510", "name_defined_twice": "12170", "value_range": "12160",
    "type_mismatch": "12150", "string_too_long": "12210", "label_twice": "12450",
    "end_of_record": "14000", "sub_without_m99": "14000", "program_not_found": "14011", "nesting": "14012",
    "pass_count": "14013", "call_parameters": "14020", "missing_p": "14020", "invalid_p": "14020",
    "arc_end_point": "14040", "radius_tolerance": "14040", "radius_tolerance_r": "14095", "illegal_radius": "14040",
    "arc_no_radius": "14040", "arc_tangential": "14045", "arc_turns": "14048", "zero_divide": "14051",
    "overflow_float": "14051", "overflow_integer": "14051", "argument_out_of_range": "14051",
    "sequence_number_not_found": "14080", "jump_not_found": "14080", "sequence_out_of_range": "14080",
    "arc_radius_small": "14095", "illegal_type_conversion": "14096", "no_number": "14098",
    "polar_radius": "14200", "polar_angle": "14210", "polar_pole": "14270", "polar_format": "14280",
    "illegal_plane": "16410", "axis_twice": "16420", "chamfer_negative": "16500",
    "write_protected": "17070", "variable_out_of_range": "17020", "illegal_value": "17050",
    "no_tool": "17160", "illegal_d": "17181", "illegal_offset_number": "17180", "tool_not_in_magazine": "17190",
    "feed_zero": "10860", "feed_negative": "14800", "multiple_codes": "14770", "too_many_m": "14770",
    "m_group_twice": "14760", "arc_overdetermined": "14900", "cip_collinear": "14920",
    "comp_start_not_linear": "10753", "comp_end_not_linear": "10754", "plane_change_in_comp": "10757",
    "crc_interference": "10751", "crc_no_intersection": "10751", "crc_arc_startup": "10753",
    "crc_g39_motion": "10753", "chamfer_too_large": "10883", "chamfer_missing_move": "10882",
    "chamfer_not_g01": "10882", "chamfer_in_thread": "10882", "comp_without_tool": "10750",
    "css_no_s": "10900", "css_negative_lims": "14820", "css_range": "14840",
    "thread_pitch_zero": "16720", "thread_no_axis": "16740", "spindle_speed_illegal": "16763",
    "frame_illegal": "18311", "frame_rotation": "18310", "illegal_scale": "12160",
    "transform_active": "14400", "tilt_no_solution": "61184", "tilt_mode": "61183", "tilt_record": "61182",
    "tilt_rotary_positions": "61189", "tilt_retract": "61190", "tilt_direct": "61153",
    "wait_mismatch": "TW0008", "wait_illegal_p": "14020", "software_limit": "10720", "working_area": "10730",
    "param_not_variable": "12320", "arc_angle": "14910",
    "cycle_no_tool_offset": "61000", "cycle_pitch": "61001", "cycle_machining_type": "61002",
    "cycle_no_feed": "61003", "cycle_axes": "61004", "cycle_tool_radius": "61105",
    "cycle_finish_allowance": "61010", "cycle_scaling": "61011", "cycle_parameter": "61019",
    "cycle_value_high": "61021", "cycle_value_low": "61022", "cycle_value_zero": "61023",
    "cycle_reference_plane": "61101", "cycle_spindle_direction": "61102", "cycle_holes_zero": "61103",
    "cycle_slot_violation": "61104", "cycle_first_depth": "61107", "cycle_cdir": "61109",
    "cycle_bottom_allowance": "61110", "cycle_infeed_width": "61111", "cycle_radius_negative": "61112",
    "cycle_crad": "61113", "cycle_helix": "61108", "cycle_length_width": "61118", "cycle_thread_type": "61120",
    "cycle_finished_diameter": "61601", "cycle_tool_width": "61602", "cycle_recess_type": "61603",
    "cycle_tool_violates": "61604", "cycle_contour": "61605", "cycle_contour_prep": "61606",
    "cycle_start_point": "61607", "cycle_tool_point": "61608", "cycle_shape": "61609",
    "cycle_infeed_depth": "61610", "cycle_no_intersection": "61611", "cycle_thread_finish": "61612",
    "cycle_no_tool": "61008", "cycle_tool_radius_small": "61007",
}

# Built-in copies of the database texts for the numbers above (used only without the database)
ALARM_TEXTS = {
    "10720": "[Channel %1: ] Block %3 axis %2 software limit switch %4",
    "10730": "[Channel %1: ] Block %3 axis %2 working area limitation %4",
    "10750": "[Channel %1: ] Block %2 tool radius compensation activated without tool number",
    "10751": "[Channel %1: ] Block %2 danger of collision due to tool radius compensation",
    "10753": "[Channel %1: ] Block %2 selection of the tool radius compensation only possible in linear block",
    "10754": "[Channel %1: ] Block %2 deselection of the tool radius compensation only possible in linear block",
    "10757": "[Channel %1: ] Block %2 changing the compensation plane while tool radius compensation is active "
             "not possible",
    "10860": "[Channel %1: ] Block %2 feedrate not programmed",
    "10882": "[Channel %1: ] Block %2 activation of chamfers or radii (non-modal) without traversing movement "
             "in the block",
    "10883": "[Channel %1: ] Block %2 chamfer or fillet has to be reduced",
    "10900": "[Channel %1: ] Block %2 no S value programmed for constant cutting speed",
    "12000": "[Channel %1: ] Block %2 address %3 programmed repeatedly",
    "12060": "[Channel %1: ] Block %2 same G group programmed repeatedly",
    "12080": "[Channel %1: ] Block %2 syntax error in text %3",
    "12090": "[Channel %1: ] Block %2 unexpected parameter %3",
    "12110": "[Channel %1: ] Block %2 block syntax cannot be interpreted",
    "12150": "[Channel %1: ] Block %2 operation %3 not compatible with data type",
    "12160": "[Channel %1: ] Block %2 Value %3 lies beyond the value range",
    "12170": "[Channel %1: ] Block %2 name %3 defined several times",
    "12210": "[Channel %1: ] Block %2 string %3 too long",
    "12320": "[Channel %1: ] Block %2 parameter %3 is no variable",
    "12450": "[Channel %1: ] Block %2 label defined twice",
    "12470": "[Channel %1: ] Block %2 G function %3 is unknown",
    "12550": "[Channel %1: ] Block %2 name %3 not defined or option/function not activated",
    "12640": "[Channel %1: ] Block %2 invalid nesting of control structures",
    "12641": "[Channel %1: ] Block %2 maximum nesting depth of control structures exceeded",
    "14000": "[Channel %1: ] Block %2 illegal end of file",
    "14011": "[Channel %1: ] Block %2 program %3 not existing or will be edited",
    "14012": "[Channel %1: ] Block %2 maximum subroutine level exceeded",
    "14013": "[Channel %1: ] Block %2 number of subroutine passes invalid",
    "14020": "[Channel %1: ] Block %2 wrong value or wrong number of parameters on function or procedure call",
    "14040": "[Channel %1: ] Block %2 error in end point of circle",
    "14045": "[Channel %1: ] Block %2 error in tangential circle programming",
    "14048": "[Channel %1: ] Block %2 wrong number of revolutions in circle programming",
    "14051": "[Channel %1: ] Block %2 arithmetic error in part program",
    "14080": "[Channel %1: ] Block %2 jump destination %3 not found",
    "14095": "[Channel %1: ] Block %2 radius for circle programming too small",
    "14096": "[Channel %1: ] Block %2 illegal type conversion",
    "14098": "[Channel %1: ] Block %2 conversion error: no valid number found",
    "14200": "[Channel %1: ] Block %2 negative polar radius",
    "14210": "[Channel %1: ] Block %2 polar angle too large",
    "14270": "[Channel %1: ] Block %2 pole programmed incorrectly",
    "14280": "[Channel %1: ] Block %2 polar coordinates programmed incorrectly",
    "14400": "[Channel %1: ] Block %2 tool radius compensation active at transformation switchover",
    "14500": "[Channel %1: ] Block %2 illegal DEF or PROC instruction in the part program",
    "14510": "[Channel %1: ] Block %2 PROC instruction missing on subroutine call",
    "14760": "[Channel %1: ] Block %2 auxiliary function of a group programmed repeatedly",
    "14770": "[Channel %1: ] Block %2 auxiliary function programmed incorrectly",
    "14800": "[Channel %1: ] Block %2 programmed path speed less or equal to zero",
    "14820": "[Channel %1: ] Block %2 negative value for maximum spindle speed programmed with constant "
             "cutting speed",
    "14840": "[Channel %1: ] Block %2 incorrect value range for constant cutting speed",
    "14900": "[Channel %1: ] Block %2 center point and end point programmed simultaneously",
    "14910": "[Channel %1: ] Block %2 invalid angle of aperture for programmed circle",
    "14920": "[Channel %1: ] Block %2 intermediate point of circle incorrect",
    "16410": "[Channel %1: ] Block %2 axis %3 is not a geometry axis",
    "16420": "[Channel %1: ] Block %2 axis %3 programmed repeatedly",
    "16500": "[Channel %1: ] Block %2 chamfer or rounding negative",
    "16720": "[Channel %1: ] Block %2 axis %3 thread pitch is zero",
    "16740": "[Channel %1: ] Block %2 no geometry axis programmed",
    "16763": "[Channel %1: ] Block %2 axis %3 programmed speed is illegal (zero or negative)",
    "17020": "[Channel %1: ] Block %2 illegal array index 1",
    "17050": "[Channel %1: ] Block %2 illegal value",
    "17070": "[Channel %1: ] Block %2 data is write-protected",
    "17160": "[Channel %1: ] Block %2 no tool selected",
    "17180": "[Channel %1: ] Block %2 illegal D number",
    "17181": "[Channel %1: ] Block %2 T no.= %3, D no.= %4 not existing",
    "17190": "[Channel %1: ] Block %2 illegal T number %3",
    "18310": "[Channel %1: ] Block %2 frame: illegal rotation",
    "18311": "[Channel %1: ] Block %2 frame: illegal instruction",
    "61000": "[Channel %1: ] Block %2: No tool offset active",
    "61001": "[Channel %1: ] Block %2: Thread pitch incorrectly defined",
    "61002": "[Channel %1: ] Block %2: Type of machining incorrectly defined",
    "61003": "[Channel %1: ] Block %2: No feed programmed in cycle",
    "61004": "[Channel %1: ] Block %2: Incorrect configuration of geometry axes",
    "61007": "[Channel %1: ] Block %2: Tool radius too small",
    "61008": "[Channel %1: ] Block %2: No tool active",
    "61010": "[Channel %1: ] Block %2: Finishing allowance too large",
    "61011": "[Channel %1: ] Block %2: Scaling not permissible",
    "61019": "[Channel %1: ] Block %2: Parameter %4 incorrectly defined",
    "61021": "[Channel %1: ] Block %2: Parameter %4 value too high",
    "61022": "[Channel %1: ] Block %2: Parameter %4 value too low",
    "61023": "[Channel %1: ] Block %2: Parameter %4 value must be unequal to zero",
    "61101": "[Channel %1: ] Block %2: Reference plane incorrectly defined",
    "61102": "[Channel %1: ] Block %2: No spindle direction programmed",
    "61103": "[Channel %1: ] Block %2: Number of holes is zero",
    "61104": "[Channel %1: ] Block %2: Contour violation of slots",
    "61105": "[Channel %1: ] Block %2: Milling cutter radius too large",
    "61107": "[Channel %1: ] Block %2: First drilling depth incorrectly defined",
    "61108": "[Channel %1: ] Block %2: Values for parameters _RAD1 and _DP1 not permissible",
    "61109": "[Channel %1: ] Block %2: Parameter _CDIR incorrectly defined",
    "61110": "[Channel %1: ] Block %2: Finishing allowance at bottom > depth infeed",
    "61111": "[Channel %1: ] Block %2: Infeed width > Tool diameter",
    "61112": "[Channel %1: ] Block %2: Tool radius negative",
    "61113": "[Channel %1: ] Block %2: Parameter _CRAD too large for corner radius",
    "61118": "[Channel %1: ] Block %2: Length or width = 0",
    "61120": "[Channel %1: ] Block %2: Thread type inside / outside not defined",
    "61153": "[Channel %1: ] Block %2: No 'Rotary axes direct' swivel mode possible --> error code: %4",
    "61182": "[Channel %1: ] Block %2: Name of swivel data record unknown",
    "61183": "[Channel %1: ] Block %2: Clearance mode _FR outside of value range 0..2.",
    "61184": "[Channel %1: ] Block %2: No solution possible with current input angle values",
    "61189": "[Channel %1: ] Block %2: invalid rotary axis positions",
    "61190": "[Channel %1: ] Block %2: unable to retract in tool direction --> error code: %4",
    "61601": "[Channel %1: ] Block %2: Finished part diameter too small",
    "61602": "[Channel %1: ] Block %2: Tool width incorrectly defined",
    "61603": "[Channel %1: ] Block %2: Recess type incorrectly defined",
    "61604": "[Channel %1: ] Block %2: Active tool violates programmed contour",
    "61605": "[Channel %1: ] Block %2: Contour incorrectly programmed",
    "61606": "[Channel %1: ] Block %2: Error during contour preparation",
    "61607": "[Channel %1: ] Block %2: Starting point incorrectly programmed",
    "61608": "[Channel %1: ] Block %2: Incorrect tool point direction programmed",
    "61609": "[Channel %1: ] Block %2: Shape incorrectly defined",
    "61610": "[Channel %1: ] Block %2: No infeed depth programmed",
    "61611": "[Channel %1: ] Block %2: No point of intersection found",
    "61612": "[Channel %1: ] Block %2: Thread finishing not possible",
    "TW0008": "PATH WAIT CAN NEVER BE RELEASED",
}

# words this dialect accepts (docs/dialects.md "Siemens 840D sl" lists them from here)
WORD_LIST = {
    "motion": "G0 G1 G2 G3 CIP CT G33 G331 G332, I J K (=AC() =IC()), I1 J1 K1, CR=, AR=, TURN=, AP= RP=",
    "plane_units_distance": "G17 G18 G19, G70 G71 G700 G710, G90 G91, AC() IC() DC() ACP() ACN(), DIAMON DIAMOF DIAM90",
    "feed_spindle": "F, G93 G94 G95 G96 G97, LIMS=, S, M3 M4 M5, SPOS=, M40-M45, G25/G26 S, CFC CFTCP CFIN",
    "frames": "G500 G54-G57 G505-G599, G53 G153 SUPA, TRANS ATRANS ROT AROT (RPL=) SCALE ASCALE MIRROR AMIRROR, "
              "G58 G59, G110 G111 G112, $P_PFRAME $P_UIFR[n] $P_IFRAME $P_ACTFRAME, CTRANS CROT CSCALE CMIRROR ':', "
              "G25 / G26 with axes (working area limitation) WALIMON WALIMOF",
    "path_control": "G60 G64 G641 ADIS= ADISPOS= G642 G643 G644 G645, G9, G601 G602 G603, SOFT BRISK DRIVE, FFWON "
                    "FFWOF, COMPOF COMPON COMPCURV COMPCAD COMPSURF, CYCLE832, CHF= CHR= RND= RNDM=",
    "tools": "T, M6, D0-D9, G40 G41 G42, G450 G451, NORM",
    "other_m": "M0 M1 M2 M30 M17 M7 M8 M9, machine-builder M codes (physics.controller.m_codes)",
    "programming": "R0-R99, R[n], DEF INT/REAL/BOOL/CHAR/STRING[n]/AXIS/FRAME, PROC (VAR) SAVE, EXTERN, "
                   "subprogram calls by name with P= and parameters, CALL, MCALL, M17, RET, IF/ELSE/ENDIF, WHILE/"
                   "ENDWHILE, FOR/TO/ENDFOR, LOOP/ENDLOOP, REPEAT/UNTIL, REPEAT label [label] P=, REPEATB label P=, "
                   "ENDLABEL, GOTOF GOTOB GOTO GOTOC (labels, N numbers), "
                   "MSG(), SETAL(), STOPRE, WAITM(), G4 F/S, G74, G75 FP=",
    "functions": "SIN COS TAN ASIN ACOS ATAN2 SQRT POT ABS TRUNC ROUND ROUNDUP LN EXP MINVAL MAXVAL BOUND NUMBER TOSTR "
                 "STRLEN, operators "
                 "+ - * / DIV MOD == <> > < >= <= AND OR XOR NOT B_AND B_OR B_XOR B_NOT <<",
    "system_variables": "$P_EP $P_F $P_S $P_TOOLNO $P_TOOL $P_TOOLP $P_GG $P_DRYRUN $P_SEARCH $P_AXN1-3 $P_UIFR "
                        "$P_PFRAME $P_IFRAME $P_ACTFRAME $AA_IW $AA_IM $A_YEAR $A_MONTH $A_DAY $A_HOUR $A_MINUTE "
                        "$A_SECOND $TC_DP1 $TC_DP3-$TC_DP6 $TC_DP12 $TC_DP15",
    "cycles": "CYCLE81 CYCLE82 CYCLE83 CYCLE84 CYCLE840 CYCLE85 CYCLE86 CYCLE87 CYCLE88 CYCLE89, HOLES1 HOLES2, "
              "POCKET3 POCKET4 SLOT1 SLOT2, CYCLE800 (_MODE axis by axis, solid angle, projection angle, rotary "
              "axes direct), CYCLE832, CYCLE95 CYCLE97 CYCLE93, TRAORI TRAFOOF (rotary axes or A3= B3= C3=)",
}

# standard cycles of the 840D sl (db/controllers/siemens-840d-sl.yaml words.cycles from the Job Planning
# manual, and the classic 840D cycles): a call of one the dialect does not model is TW0012 (not 14011)
# unless the project supplies a program of that name
STANDARD_CYCLES = frozenset({
    "CYCLE60", "CYCLE61", "CYCLE62", "CYCLE63", "CYCLE64", "CYCLE70", "CYCLE71", "CYCLE72", "CYCLE73", "CYCLE74",
    "CYCLE75", "CYCLE76", "CYCLE77", "CYCLE78", "CYCLE79", "CYCLE90", "CYCLE92", "CYCLE94", "CYCLE96", "CYCLE98",
    "CYCLE99", "CYCLE801", "CYCLE802", "CYCLE830", "CYCLE899", "CYCLE930", "CYCLE940", "CYCLE950", "CYCLE951",
    "CYCLE952", "LONGHOLE"})

__all__ = ["G_FUNCTIONS", "UNMODELLED", "STANDARD_CYCLES", "POWER_ON_MILL", "POWER_ON_LATHE", "KEYWORDS", "FRAME_WORDS", "ADDRESSES",
           "ALARM_CODES", "ALARM_TEXTS", "WORD_LIST"]
