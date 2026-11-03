O9853 (BROKEN TOOL CHECK ON THE TOOL SETTER - CNC TWIN BUILT-IN SETTER MACRO)
(T: LENGTH OFFSET NUMBER OF THE TOOL IN THE SPINDLE, DEFAULT THE ACTIVE TOOL; H: TOLERANCE ON THE LENGTH.)
(THE TOOL IS FED ON THE SETTER SIGNAL TO ITS STORED LENGTH LESS H: NO CONTACT IS A BROKEN TOOL, A CONTACT)
(BEFORE THE CLEARANCE PLANE OR MORE THAN H EARLY IS OUT OF TOLERANCE. OFFSETS ARE NOT CHANGED)
#30 = #4003
#31 = #{f_var}
#29 = #4008
#28 = #{h_var}
#32 = 1.
IF [#4006 EQ 20] THEN #32 = 25.4
IF [#20 EQ #0] THEN #20 = #4120
IF [#20 LE 0] GOTO 93
IF [#11 EQ #0] THEN #11 = {broken_tolerance_mm} / #32
IF [#11 LE 0] GOTO 93
G49
G91 G28 Z0
G90 G53 G00 X[{setter_x_mm} / #32] Y[{setter_y_mm} / #32]
#1 = #5023 - #5003
#26 = {setter_z_mm} / #32 - #1
G43 H#20
#3 = #26 + {setter_clearance_mm} / #32
G31 P{setter_skip} Z#3 F[{fast_feed_mm_min} / #32]
#5 = 0
IF [ABS[#5063 - #3] GE {count_mm} / #32] THEN #5 = 2
IF [#5 NE 0] GOTO 20
#4 = #26 - #11
G31 P{setter_skip} Z#4 F[{measure_feed_mm_min} / #32]
#6 = #5063
IF [ABS[#6 - #4] LT {count_mm} / #32] THEN #5 = 1
IF [#6 GT #26 + #11] THEN #5 = 2
#135 = #6 - #26
N20 G91 G28 Z0
G#29 H#28
G#30 F#31
IF [#5 EQ 1] GOTO 95
IF [#5 EQ 2] GOTO 94
M99
N93 #3000 = {alarm_format} (FORMAT ERROR)
N94 #3000 = {alarm_tolerance} (OUT OF TOLERANCE)
N95 #3000 = {alarm_broken} (BROKEN TOOL)
M99
