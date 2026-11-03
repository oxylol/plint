O9814 (BORE / BOSS MEASUREMENT - CNC TWIN BUILT-IN PROBE MACRO)
(D: NOMINAL DIAMETER, THE PROBE STARTS ON THE NOMINAL CENTRE; Z: PROBING HEIGHT OF A BOSS, NO Z: BORE;)
(R: CLEARANCE OUTSIDE A BOSS; Q: OVERTRAVEL; S1-S6: WORK OFFSET G54-G59 MOVED TO THE MEASURED CENTRE;)
(H: TOLERANCE ON THE DIAMETER; F: FEED. X IS TOUCHED FIRST; THE Y PAIR THROUGH THE MEASURED X CENTRE GIVES)
(THE DIAMETER. RESULTS: #135 #136 CENTRE X Y, #138 DIAMETER, #139 DIAMETER ERROR, #140 CENTRE OFFSET)
#30 = #4003
#31 = #{f_var}
#32 = 1.
IF [#4006 EQ 20] THEN #32 = 25.4
#27 = [#[{d_geom} + #4120] + #[{d_wear} + #4120]] * {d_factor}
IF [#27 LE 0] GOTO 92
IF [#7 EQ #0] GOTO 93
IF [#7 LE 0] GOTO 93
IF [#9 EQ #0] THEN #9 = {measure_feed_mm_min} / #32
IF [#17 EQ #0] THEN #17 = {search_mm} / #32
IF [#18 EQ #0] THEN #18 = {clearance_mm} / #32
#24 = #5001
#25 = #5002
#28 = #24
#29 = #25
#15 = #5003
#16 = 0
G90
WHILE [#16 LT 4] DO 1
#2 = FIX[#16 / 2] + 1
#5 = 1. - 2. * [#16 - 2 * [#2 - 1]]
#4 = #[27 + #2]
#12 = #28
#13 = #29
#14 = #15
IF [#26 EQ #0] GOTO 10
#[11 + #2] = #4 + #5 * [#7 / 2 + #18 + #27]
G65 P9810 X#12 Y#13
G65 P9810 Z#26
#14 = #26
#1 = #[11 + #2]
#[11 + #2] = #4 + #5 * [#7 / 2 + #27 - #17]
GOTO 20
N10 IF [#16 NE 2] GOTO 15
G65 P9810 X#12 Y#13
N15 #[11 + #2] = #4 + #5 * [#7 / 2 - #27 + #17]
N20 #6 = #[11 + #2]
G31 X#12 Y#13 Z#14 F#9
#3 = #[5060 + #2]
IF [ABS[#3 - #6] LT {count_mm} / #32] GOTO 91
IF [#26 EQ #0] THEN #[20 + #16] = #3 + #5 * #27
IF [#26 NE #0] THEN #[20 + #16] = #3 - #5 * #27
IF [#26 EQ #0] GOTO 30
#[11 + #2] = #1
G01 X#12 Y#13 Z#14 F#9
G65 P9810 Z#15
GOTO 40
N30 G01 X#28 Y#29 F#9
N40 IF [#16 EQ 1] THEN #28 = [#20 + #21] / 2
#16 = #16 + 1
END 1
#10 = [#22 + #23] / 2
#12 = #22 - #23
#13 = #12 - #7
#14 = #28 - #24
#1 = #10 - #25
#135 = #28
#136 = #10
#138 = #12
#139 = #13
#140 = SQRT[#14 * #14 + #1 * #1]
G65 P9810 X#28 Y#10
IF [#11 EQ #0] GOTO 60
IF [ABS[#13] GT #11] GOTO 94
N60 IF [#19 EQ #0] GOTO 70
IF [[#19 LT 1] OR [#19 GT 6]] GOTO 93
#[5201 + #19 * 20] = #[5201 + #19 * 20] + #14
#[5202 + #19 * 20] = #[5202 + #19 * 20] + #1
N70 G#30 F#31
M99
N91 #3000 = {alarm_open} (PROBE OPEN)
N92 #3000 = {alarm_calibration} (PROBE NOT CALIBRATED)
N93 #3000 = {alarm_format} (FORMAT ERROR)
N94 #3000 = {alarm_tolerance} (OUT OF TOLERANCE)
M99
