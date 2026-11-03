O9812 (WEB / POCKET MEASUREMENT - CNC TWIN BUILT-IN PROBE MACRO)
(X OR Y: NOMINAL WIDTH, THE PROBE STARTS ON THE NOMINAL CENTRE; Z: PROBING HEIGHT OF A WEB, NO Z: POCKET;)
(R: CLEARANCE OUTSIDE A WEB; Q: OVERTRAVEL; S1-S6: WORK OFFSET G54-G59 MOVED TO THE MEASURED CENTRE;)
(H: TOLERANCE ON THE WIDTH; F: FEED. RESULTS: #135 OR #136 CENTRE, #138 WIDTH, #139 WIDTH ERROR, #140 CENTRE ERROR)
#30 = #4003
#31 = #{f_var}
#32 = 1.
IF [#4006 EQ 20] THEN #32 = 25.4
#27 = [#[{d_geom} + #4120] + #[{d_wear} + #4120]] * {d_factor}
IF [#27 LE 0] GOTO 92
IF [#9 EQ #0] THEN #9 = {measure_feed_mm_min} / #32
IF [#17 EQ #0] THEN #17 = {search_mm} / #32
IF [#18 EQ #0] THEN #18 = {clearance_mm} / #32
#2 = 0
IF [#24 NE #0] THEN #2 = 1
IF [#25 NE #0] THEN #2 = #2 + 2
IF [[#2 NE 1] AND [#2 NE 2]] GOTO 93
#3 = #[23 + #2]
IF [#3 LE 0] GOTO 93
#28 = #5001
#29 = #5002
#15 = #5003
#4 = #[27 + #2]
#16 = 0
G90
WHILE [#16 LT 2] DO 1
#5 = 1. - 2. * #16
#12 = #28
#13 = #29
#14 = #15
IF [#26 EQ #0] GOTO 10
#[11 + #2] = #4 + #5 * [#3 / 2 + #18 + #27]
G65 P9810 X#12 Y#13
G65 P9810 Z#26
#14 = #26
#1 = #[11 + #2]
#[11 + #2] = #4 + #5 * [#3 / 2 + #27 - #17]
GOTO 20
N10 #[11 + #2] = #4 + #5 * [#3 / 2 - #27 + #17]
N20 #6 = #[11 + #2]
G31 X#12 Y#13 Z#14 F#9
#7 = #[5060 + #2]
IF [ABS[#7 - #6] LT {count_mm} / #32] GOTO 91
IF [#26 EQ #0] THEN #[21 + #16] = #7 + #5 * #27
IF [#26 NE #0] THEN #[21 + #16] = #7 - #5 * #27
IF [#26 EQ #0] GOTO 30
#[11 + #2] = #1
G01 X#12 Y#13 Z#14 F#9
G65 P9810 Z#15
GOTO 40
N30 G01 X#28 Y#29 F#9
N40 #16 = #16 + 1
END 1
#10 = [#21 + #22] / 2
#12 = #21 - #22
#13 = #12 - #3
#14 = #10 - #4
#[134 + #2] = #10
#138 = #12
#139 = #13
#140 = #14
IF [#26 EQ #0] GOTO 50
G65 P9810 X#28 Y#29
N50 IF [#11 EQ #0] GOTO 60
IF [ABS[#13] GT #11] GOTO 94
N60 IF [#19 EQ #0] GOTO 70
IF [[#19 LT 1] OR [#19 GT 6]] GOTO 93
#[5200 + #19 * 20 + #2] = #[5200 + #19 * 20 + #2] + #14
N70 G#30 F#31
M99
N91 #3000 = {alarm_open} (PROBE OPEN)
N92 #3000 = {alarm_calibration} (PROBE NOT CALIBRATED)
N93 #3000 = {alarm_format} (FORMAT ERROR)
N94 #3000 = {alarm_tolerance} (OUT OF TOLERANCE)
M99
