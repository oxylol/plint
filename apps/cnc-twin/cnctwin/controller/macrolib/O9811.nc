O9811 (SINGLE SURFACE MEASUREMENT - CNC TWIN BUILT-IN PROBE MACRO)
(ONE OF X Y Z: NOMINAL SURFACE POSITION IN WORK COORDINATES; S1-S6: WORK OFFSET G54-G59 SHIFTED BY THE ERROR;)
(T: LENGTH OFFSET WHOSE WEAR TAKES THE Z ERROR; H: TOLERANCE ON THE ERROR; Q: OVERTRAVEL; F: FEED)
(RESULTS: #135-#137 MEASURED SURFACE X Y Z, #138-#140 ERROR X Y Z)
#30 = #4003
#31 = #{f_var}
#32 = 1.
IF [#4006 EQ 20] THEN #32 = 25.4
#27 = [#[{d_geom} + #4120] + #[{d_wear} + #4120]] * {d_factor}
IF [#27 LE 0] GOTO 92
IF [#9 EQ #0] THEN #9 = {measure_feed_mm_min} / #32
IF [#17 EQ #0] THEN #17 = {search_mm} / #32
#1 = 0
IF [#24 NE #0] THEN #1 = 1
IF [#25 NE #0] THEN #1 = #1 + 2
IF [#26 NE #0] THEN #1 = #1 + 4
#2 = 0
IF [#1 EQ 1] THEN #2 = 1
IF [#1 EQ 2] THEN #2 = 2
IF [#1 EQ 4] THEN #2 = 3
IF [#2 EQ 0] GOTO 93
#3 = #[23 + #2]
#4 = #[5000 + #2]
#5 = 1.
IF [#3 LT #4] THEN #5 = -1.
#12 = #5001
#13 = #5002
#14 = #5003
#6 = #3 + #5 * [#17 - #27]
#[11 + #2] = #6
G90 G31 X#12 Y#13 Z#14 F#9
#7 = #[5060 + #2]
IF [ABS[#7 - #6] LT {count_mm} / #32] GOTO 91
#8 = #7 + #5 * #27
#10 = #8 - #3
#[134 + #2] = #8
#[137 + #2] = #10
#[11 + #2] = #7 - #5 * {backoff_mm} / #32
G01 X#12 Y#13 Z#14 F#9
IF [#11 EQ #0] GOTO 20
IF [ABS[#10] GT #11] GOTO 94
N20 IF [#19 EQ #0] GOTO 30
IF [[#19 LT 1] OR [#19 GT 6]] GOTO 93
#[5200 + #19 * 20 + #2] = #[5200 + #19 * 20 + #2] + #10
N30 IF [#20 EQ #0] GOTO 40
IF [#2 NE 3] GOTO 93
#[{h_wear} + #20] = #[{h_wear} + #20] - #10
N40 G#30 F#31
M99
N91 #3000 = {alarm_open} (PROBE OPEN)
N92 #3000 = {alarm_calibration} (PROBE NOT CALIBRATED)
N93 #3000 = {alarm_format} (FORMAT ERROR)
N94 #3000 = {alarm_tolerance} (OUT OF TOLERANCE)
M99
