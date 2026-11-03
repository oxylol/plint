"""FANUC 0i-F tables: G codes (M series and T series G code system A), alarm map, built-in texts.

Group numbers are FANUC's (they are what #4001-#4030 report). ``kind`` is the handler's
semantic key; ``cycle=True`` routes the code to the cycle hook (canned cycles of the cycles
agent), ``axis_words=True`` marks one-shot codes that consume the block's axis words.
"""

from __future__ import annotations

from .base import GSpec

M_GCODES: dict[str, GSpec] = {
    "G00": GSpec(1, "motion", "rapid"), "G01": GSpec(1, "motion", "linear"),
    "G02": GSpec(1, "motion", "cw"), "G03": GSpec(1, "motion", "ccw"),
    "G33": GSpec(1, "motion", "thread"),
    "G04": GSpec(0, "dwell"), "G05.1": GSpec(0, "ai_contour"), "G08": GSpec(0, "preview"),
    "G09": GSpec(0, "exact_once"), "G10": GSpec(0, "data", axis_words=True), "G11": GSpec(0, "data_end"),
    "G15": GSpec(17, "polar", "off"), "G16": GSpec(17, "polar", "on"),
    "G17": GSpec(2, "plane", "XY"), "G18": GSpec(2, "plane", "ZX"), "G19": GSpec(2, "plane", "YZ"),
    "G20": GSpec(6, "units", "inch"), "G21": GSpec(6, "units", "mm"),
    "G22": GSpec(4, "stroke", "on", axis_words=True), "G23": GSpec(4, "stroke", "off"),
    "G25": GSpec(19, "fluctuation", "off"), "G26": GSpec(19, "fluctuation", "on"),
    "G27": GSpec(0, "ref_check", axis_words=True), "G28": GSpec(0, "ref_return", axis_words=True),
    "G29": GSpec(0, "from_ref", axis_words=True), "G30": GSpec(0, "ref_return_n", axis_words=True),
    "G31": GSpec(0, "skip", axis_words=True), "G37": GSpec(0, "probe", "G37", axis_words=True),
    "G39": GSpec(0, "comp_corner"),
    "G40": GSpec(7, "comp", "G40"), "G41": GSpec(7, "comp", "G41"), "G42": GSpec(7, "comp", "G42"),
    "G43": GSpec(8, "length", "G43"), "G44": GSpec(8, "length", "G44"), "G49": GSpec(8, "length", "G49"),
    "G43.4": GSpec(8, "tcp", "fanuc:G43.4"), "G43.5": GSpec(8, "tcp", "fanuc:G43.5"),
    "G50": GSpec(11, "scale", "off"), "G51": GSpec(11, "scale", "on"),
    "G50.1": GSpec(22, "mirror", "off"), "G51.1": GSpec(22, "mirror", "on"),
    "G52": GSpec(0, "local", axis_words=True), "G53": GSpec(0, "machine", axis_words=True),
    "G54": GSpec(14, "wcs", "G54"), "G55": GSpec(14, "wcs", "G55"), "G56": GSpec(14, "wcs", "G56"),
    "G57": GSpec(14, "wcs", "G57"), "G58": GSpec(14, "wcs", "G58"), "G59": GSpec(14, "wcs", "G59"),
    "G54.1": GSpec(14, "wcs", "P"),
    "G60": GSpec(0, "single_direction", axis_words=True),
    "G61": GSpec(15, "exact", "exact_stop"), "G62": GSpec(15, "exact", "corner_override"),
    "G63": GSpec(15, "exact", "tapping"), "G64": GSpec(15, "exact", "blend"),
    "G65": GSpec(0, "macro_call"), "G66": GSpec(12, "modal_call", "on"), "G67": GSpec(12, "modal_call", "off"),
    "G68": GSpec(16, "rotation", "on"), "G69": GSpec(16, "rotation", "off"),
    "G68.2": GSpec(16, "tcp", "fanuc:G68.2"),
    "G73": GSpec(9, "cycle", "G73", cycle=True), "G74": GSpec(9, "cycle", "G74", cycle=True),
    "G76": GSpec(9, "cycle", "G76", cycle=True), "G80": GSpec(9, "cycle", "off"),
    "G81": GSpec(9, "cycle", "G81", cycle=True), "G82": GSpec(9, "cycle", "G82", cycle=True),
    "G83": GSpec(9, "cycle", "G83", cycle=True), "G84": GSpec(9, "cycle", "G84", cycle=True),
    "G85": GSpec(9, "cycle", "G85", cycle=True), "G86": GSpec(9, "cycle", "G86", cycle=True),
    "G87": GSpec(9, "cycle", "G87", cycle=True), "G88": GSpec(9, "cycle", "G88", cycle=True),
    "G89": GSpec(9, "cycle", "G89", cycle=True), "G81.1": GSpec(9, "cycle", "G81.1", cycle=True),
    "G90": GSpec(3, "distance", "abs"), "G91": GSpec(3, "distance", "inc"),
    "G92": GSpec(0, "coord_set", axis_words=True), "G92.1": GSpec(0, "coord_preset", axis_words=True),
    "G93": GSpec(5, "feed", "inverse_time"), "G94": GSpec(5, "feed", "per_min"), "G95": GSpec(5, "feed", "per_rev"),
    "G96": GSpec(13, "css", "on"), "G97": GSpec(13, "css", "off"),
    "G98": GSpec(10, "return", "initial"), "G99": GSpec(10, "return", "r"),
    "G07.1": GSpec(0, "tcp", "fanuc:G07.1"),
    "G12.1": GSpec(21, "tcp", "fanuc:G12.1"), "G13.1": GSpec(21, "tcp", "off"),
}

M_POWER_ON = {1: "G00", 2: "G17", 3: "G90", 4: "G22", 5: "G94", 6: "G21", 7: "G40", 8: "G49", 9: "G80",
              10: "G98", 11: "G50", 12: "G67", 13: "G97", 14: "G54", 15: "G64", 16: "G69", 17: "G15",
              19: "G25", 21: "G13.1", 22: "G50.1"}

T_GCODES: dict[str, GSpec] = {
    "G00": GSpec(1, "motion", "rapid"), "G01": GSpec(1, "motion", "linear"),
    "G02": GSpec(1, "motion", "cw"), "G03": GSpec(1, "motion", "ccw"),
    "G32": GSpec(1, "motion", "thread"), "G34": GSpec(1, "cycle", "G34", cycle=True),
    "G90": GSpec(1, "cycle", "G90", cycle=True), "G92": GSpec(1, "cycle", "G92", cycle=True),
    "G94": GSpec(1, "cycle", "G94", cycle=True),
    "G04": GSpec(0, "dwell"), "G07.1": GSpec(0, "tcp", "fanuc:G07.1"),
    "G10": GSpec(0, "data", axis_words=True), "G11": GSpec(0, "data_end"),
    "G12.1": GSpec(21, "tcp", "fanuc:G12.1"), "G13.1": GSpec(21, "tcp", "off"),
    "G17": GSpec(16, "plane", "XY"), "G18": GSpec(16, "plane", "ZX"), "G19": GSpec(16, "plane", "YZ"),
    "G20": GSpec(6, "units", "inch"), "G21": GSpec(6, "units", "mm"),
    "G22": GSpec(9, "stroke", "on", axis_words=True), "G23": GSpec(9, "stroke", "off"),
    "G25": GSpec(8, "fluctuation", "off"), "G26": GSpec(8, "fluctuation", "on"),
    "G27": GSpec(0, "ref_check", axis_words=True), "G28": GSpec(0, "ref_return", axis_words=True),
    "G29": GSpec(0, "from_ref", axis_words=True), "G30": GSpec(0, "ref_return_n", axis_words=True),
    "G31": GSpec(0, "skip", axis_words=True),
    "G36": GSpec(0, "probe", "G36", axis_words=True), "G37": GSpec(0, "probe", "G37", axis_words=True),
    "G40": GSpec(7, "comp", "G40"), "G41": GSpec(7, "comp", "G41"), "G42": GSpec(7, "comp", "G42"),
    "G50": GSpec(0, "coord_or_clamp", axis_words=True), "G50.3": GSpec(0, "coord_preset", axis_words=True),
    "G52": GSpec(0, "local", axis_words=True), "G53": GSpec(0, "machine", axis_words=True),
    "G54": GSpec(14, "wcs", "G54"), "G55": GSpec(14, "wcs", "G55"), "G56": GSpec(14, "wcs", "G56"),
    "G57": GSpec(14, "wcs", "G57"), "G58": GSpec(14, "wcs", "G58"), "G59": GSpec(14, "wcs", "G59"),
    "G65": GSpec(0, "macro_call"), "G66": GSpec(12, "modal_call", "on"), "G67": GSpec(12, "modal_call", "off"),
    "G70": GSpec(0, "cycle", "G70", cycle=True), "G71": GSpec(0, "cycle", "G71", cycle=True),
    "G72": GSpec(0, "cycle", "G72", cycle=True), "G73": GSpec(0, "cycle", "G73", cycle=True),
    "G74": GSpec(0, "cycle", "G74", cycle=True), "G75": GSpec(0, "cycle", "G75", cycle=True),
    "G76": GSpec(0, "cycle", "G76", cycle=True),
    "G80": GSpec(10, "cycle", "off"), "G83": GSpec(10, "cycle", "G83", cycle=True),
    "G84": GSpec(10, "cycle", "G84", cycle=True), "G85": GSpec(10, "cycle", "G85", cycle=True),
    "G87": GSpec(10, "cycle", "G87", cycle=True), "G88": GSpec(10, "cycle", "G88", cycle=True),
    "G89": GSpec(10, "cycle", "G89", cycle=True),
    "G96": GSpec(2, "css", "on"), "G97": GSpec(2, "css", "off"),
    "G98": GSpec(5, "feed", "per_min"), "G99": GSpec(5, "feed", "per_rev"),
}

T_POWER_ON = {1: "G00", 2: "G97", 5: "G99", 6: "G21", 7: "G40", 8: "G25", 9: "G22", 10: "G80", 12: "G67",
              14: "G54", 16: "G18", 21: "G13.1"}

# condition key -> FANUC alarm number
ALARM_CODES = {
    "too_many_digits": "PS0003", "no_data_after_address": "PS0005", "illegal_minus": "PS0006",
    "illegal_decimal_point": "PS0007", "improper_address": "PS0009", "improper_g_code": "PS0010",
    "feed_zero": "PS0011", "radius_tolerance": "PS0020", "radius_tolerance_r": "PS0020",
    "illegal_plane": "PS0021", "arc_no_radius": "PS0022", "illegal_radius": "PS0023",
    "illegal_offset_value": "PS0029", "illegal_offset_number": "PS0030", "g10_illegal_p": "PS0031",
    "g10_illegal_value": "PS0032", "g31_in_comp": "PS0035", "plane_change_in_comp": "PS0037",
    "ref_return_in_cycle": "PS0044", "cycle_q_missing": "PS0045", "illegal_reference_return": "PS0046",
    "program_not_found": "PS0076", "missing_p": "PS0076", "invalid_p": "PS0076", "nesting": "PS0077",
    "sequence_number_not_found": "PS0078", "g37_signal": "PS0080", "g37_offset_number": "PS0081",
    "g37_with_h": "PS0082", "g37_axis": "PS0083", "ref_check_error": "PS0092",
    "overflow_integer": "PS0110", "overflow_float": "PS0111", "zero_divide": "PS0112",
    "improper_command": "PS0113", "expression_format": "PS0114", "variable_out_of_range": "PS0115",
    "write_protected": "PS0116", "bracket_nesting": "PS0118", "argument_out_of_range": "PS0119",
    "modal_call_nesting": "PS0122", "missing_end": "PS0124", "macro_format": "PS0125",
    "illegal_loop_number": "PS0126", "nc_macro_same_block": "PS0127", "sequence_out_of_range": "PS0128",
    "g_as_argument": "PS0129", "illegal_scale": "PS0142", "wait_mismatch": "PS0160",
    "wait_illegal_p": "PS0161", "end_of_record": "PS5010", "sub_without_m99": "PS5010",
    "dwell_range": "PS0007", "multiple_codes": "PS0009",
}

# Built-in texts (used when db/controllers/<id>.yaml does not list the code): FANUC Series 0i-F
# PS alarm texts as generally documented (Smid 2005, appendix; FANUC 0i maintenance manuals).
ALARM_TEXTS = {
    "PS0003": "TOO MANY DIGIT", "PS0005": "NO DATA AFTER ADDRESS", "PS0006": "ILLEGAL USE OF MINUS SIGN",
    "PS0007": "ILLEGAL USE OF DECIMAL POINT", "PS0009": "IMPROPER NC-ADDRESS", "PS0010": "IMPROPER G-CODE",
    "PS0011": "FEED ZERO ( COMMAND )", "PS0020": "OVER TOLERANCE OF RADIUS", "PS0021": "ILLEGAL PLANE SELECT",
    "PS0022": "R OR I,J,K COMMAND NOT FOUND", "PS0023": "ILLEGAL RADIUS COMMAND",
    "PS0029": "ILLEGAL OFFSET VALUE", "PS0030": "ILLEGAL OFFSET NUMBER", "PS0031": "ILLEGAL P COMMAND IN G10",
    "PS0032": "ILLEGAL OFFSET VALUE IN G10", "PS0035": "CAN NOT COMMANDED G31",
    "PS0037": "CAN NOT CHANGE PLANE IN G41/G42", "PS0044": "G27-G30 NOT ALLOWED IN FIXED CYC",
    "PS0045": "ADDRESS Q NOT FOUND (G73/G83)", "PS0046": "ILLEGAL REFERENCE RETURN COMMAND",
    "PS0076": "PROGRAM NOT FOUND", "PS0077": "TOO MANY SUB,MACRO NESTING", "PS0078": "SEQUENCE NUMBER NOT FOUND",
    "PS0080": "G37 MEASURING POSITION REACHED SIGNAL IMPROPERLY INPUT", "PS0081": "G37 OFFSET NO. UNASSIGNED",
    "PS0082": "G37 SPECIFIED WITH H CODE", "PS0083": "G37 IMPROPER AXIS COMMAND",
    "PS0092": "ZERO RETURN CHECK (G27) ERROR", "PS0110": "OVERFLOW :INTEGER", "PS0111": "OVERFLOW :FLOAT",
    "PS0112": "ZERO DIVIDE", "PS0113": "IMPROPER COMMAND", "PS0114": "ILLEGAL EXPRESSION FORMAT",
    "PS0115": "VARIABLE NO. OUT OF RANGE", "PS0116": "WRITE PROTECTED VARIABLE",
    "PS0118": "TOO MANY BRACKET NESTING", "PS0119": "ARGUMENT VALUE OUT OF RANGE",
    "PS0122": "DUPLICATE MACRO MODAL-CALL", "PS0124": "MISSING END STATEMENT",
    "PS0125": "MACRO STATEMENT FORMAT ERROR", "PS0126": "ILLEGAL LOOP NUMBER",
    "PS0127": "DUPLICATE NC,MACRO STATEMENT", "PS0128": "ILLEGAL MACRO SEQUENCE NUMBER",
    "PS0129": "USE 'G' AS ARGUMENT", "PS0142": "ILLEGAL SCALE RATE", "PS0160": "MISMATCH WAITING M-CODE",
    "PS0161": "ILLEGAL P OF WAITING M-CODE", "PS5010": "END OF RECORD",
}

# Address letters and the largest number of significant digits of a literal (FANUC IS-B formats).
LETTERS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
MAX_DIGITS = {letter: 9 for letter in "ABCEIJKQRUVWXYZ"}
MAX_DIGITS.update({"F": 9, "G": 5, "H": 3, "D": 3, "M": 8, "S": 5, "T": 8, "P": 8, "L": 8})

# G65 argument specification I: letter -> local variable number
ARGUMENTS_I = {"A": 1, "B": 2, "C": 3, "I": 4, "J": 5, "K": 6, "D": 7, "E": 8, "F": 9, "H": 11, "M": 13,
               "Q": 17, "R": 18, "S": 19, "T": 20, "U": 21, "V": 22, "W": 23, "X": 24, "Y": 25, "Z": 26}
ARGUMENT_LETTER = {v: k for k, v in ARGUMENTS_I.items()}
LINEAR_ARGUMENTS = frozenset("EIJKQRUVWXYZ")
ROTARY_ARGUMENTS = frozenset("ABC")

# modal information variables #4102.. (letter -> offset from 4100)
MODAL_LETTERS = {"B": 2, "D": 7, "E": 8, "F": 9, "H": 11, "M": 13, "N": 14, "O": 15, "S": 19, "T": 20, "P": 30}
