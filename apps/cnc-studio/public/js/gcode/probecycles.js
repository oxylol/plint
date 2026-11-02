// Probing cycles built into the simulator (Iteration 2), in the style of Renishaw Inspection Plus,
// written in Macro B on top of G31. They are used when the job has no program of the same number.
// Conventions the simulator provides for them:
//   #5061–#5063  ball centre where the skip signal came (work coordinates)   #5090  1 = skip signal came
//   #5098        stylus ball radius of the active probe                      #27 #28 #29  1 = X / Y / Z given
//   #5221+20·n   work offset n (0 = G54): X, +1 Y, +2 Z                      #5001–#5003  current position
// Outputs as Inspection Plus: #135–#137 measured X/Y/Z, #138 size, #139–#141 position error, #142 size error.
// Arguments: X/Y/Z target (9811: the surface; 9812: the web/pocket size; 9814: D = bore/boss diameter),
// Q overtravel (default 4 mm), F feed (300), S work offset to update (1 = G54 … 6 = G59), T tolerance,
// Z on 9812/9814 = probe at this depth outside a web or boss (absent = inside a pocket or bore).
export const PROBE_CYCLES = [
  { name: 'O9810.nc', text: `O9810 (PROTECTED POSITIONING)
IF [#9 EQ 0] THEN #9 = 1000.
G31 X#24 Y#25 Z#26 F#9
IF [#5090 EQ 1] THEN #3000 = 92 (PROBE OBSTRUCTED - PATH HITS THE PART)
M99
` },
  { name: 'O9811.nc', text: `O9811 (SINGLE SURFACE)
IF [#9 EQ 0] THEN #9 = 300.
IF [#17 EQ 0] THEN #17 = 4.
#139 = 0
#140 = 0
#141 = 0
IF [#29 EQ 1] GOTO 30
IF [#27 EQ 1] GOTO 10
IF [#28 EQ 1] GOTO 20
#3000 = 93 (9811 NEEDS X Y OR Z)
N10 #32 = #5001
#31 = 1.
IF [#24 LT #32] THEN #31 = -1.
G31 X[#24 + #31 * #17] F#9
IF [#5090 EQ 0] THEN #3000 = 94 (PROBE NO SKIP SIGNAL - NOTHING TOUCHED IN X)
#135 = #5061 + #31 * #5098
#139 = #135 - #24
G01 X#32 F#9
#33 = #139
GOTO 80
N20 #32 = #5002
#31 = 1.
IF [#25 LT #32] THEN #31 = -1.
G31 Y[#25 + #31 * #17] F#9
IF [#5090 EQ 0] THEN #3000 = 94 (PROBE NO SKIP SIGNAL - NOTHING TOUCHED IN Y)
#136 = #5062 + #31 * #5098
#140 = #136 - #25
G01 Y#32 F#9
#33 = #140
GOTO 80
N30 #32 = #5003
#31 = 1.
IF [#26 LT #32] THEN #31 = -1.
G31 Z[#26 + #31 * #17] F#9
IF [#5090 EQ 0] THEN #3000 = 94 (PROBE NO SKIP SIGNAL - NOTHING TOUCHED IN Z)
#137 = #5063 + #31 * #5098
#141 = #137 - #26
G01 Z#32 F#9
#33 = #141
N80 IF [#20 LE 0] GOTO 85
IF [ABS[#33] GT #20] THEN #3000 = 95 (OUT OF TOLERANCE)
N85 IF [#19 LE 0] GOTO 90
#30 = #19 - 1
IF [#27 NE 1] GOTO 86
G10 L2 P#19 X[#[5221 + 20 * #30] + #139]
N86 IF [#28 NE 1] GOTO 87
G10 L2 P#19 Y[#[5222 + 20 * #30] + #140]
N87 IF [#29 NE 1] GOTO 90
G10 L2 P#19 Z[#[5223 + 20 * #30] + #141]
N90 M99
` },
  { name: 'O9812.nc', text: `O9812 (WEB OR POCKET - X OR Y SIZE, Z = PROBE DEPTH OUTSIDE A WEB)
IF [#9 EQ 0] THEN #9 = 300.
IF [#17 EQ 0] THEN #17 = 4.
#139 = 0
#140 = 0
#142 = 0
#1 = #5001
#2 = #5002
#3 = #5003
IF [#27 EQ 1] GOTO 10
IF [#28 EQ 1] GOTO 20
#3000 = 93 (9812 NEEDS X OR Y)
N10 #30 = #24 / 2
IF [#29 EQ 1] GOTO 15
G31 X[#1 + #30 + #17] F#9
IF [#5090 EQ 0] THEN #3000 = 94 (PROBE NO SKIP SIGNAL)
#11 = #5061 + #5098
G01 X#1 F#9
G31 X[#1 - #30 - #17] F#9
IF [#5090 EQ 0] THEN #3000 = 94 (PROBE NO SKIP SIGNAL)
#12 = #5061 - #5098
G01 X#1 F#9
GOTO 60
N15 G31 X[#1 + #30 + #5098 + #17] F1000.
IF [#5090 EQ 1] THEN #3000 = 92 (PROBE OBSTRUCTED)
G31 Z#26 F1000.
IF [#5090 EQ 1] THEN #3000 = 92 (PROBE OBSTRUCTED)
G31 X[#1 + #30 - #17] F#9
IF [#5090 EQ 0] THEN #3000 = 94 (PROBE NO SKIP SIGNAL)
#11 = #5061 - #5098
G31 X[#1 + #30 + #5098 + #17] F#9
G31 Z#3 F1000.
IF [#5090 EQ 1] THEN #3000 = 92 (PROBE OBSTRUCTED)
G31 X[#1 - #30 - #5098 - #17] F1000.
IF [#5090 EQ 1] THEN #3000 = 92 (PROBE OBSTRUCTED)
G31 Z#26 F1000.
IF [#5090 EQ 1] THEN #3000 = 92 (PROBE OBSTRUCTED)
G31 X[#1 - #30 + #17] F#9
IF [#5090 EQ 0] THEN #3000 = 94 (PROBE NO SKIP SIGNAL)
#12 = #5061 + #5098
G31 X[#1 - #30 - #5098 - #17] F#9
G31 Z#3 F1000.
G01 X#1 F1000.
GOTO 60
N20 #30 = #25 / 2
IF [#29 EQ 1] GOTO 25
G31 Y[#2 + #30 + #17] F#9
IF [#5090 EQ 0] THEN #3000 = 94 (PROBE NO SKIP SIGNAL)
#11 = #5062 + #5098
G01 Y#2 F#9
G31 Y[#2 - #30 - #17] F#9
IF [#5090 EQ 0] THEN #3000 = 94 (PROBE NO SKIP SIGNAL)
#12 = #5062 - #5098
G01 Y#2 F#9
GOTO 70
N25 G31 Y[#2 + #30 + #5098 + #17] F1000.
IF [#5090 EQ 1] THEN #3000 = 92 (PROBE OBSTRUCTED)
G31 Z#26 F1000.
IF [#5090 EQ 1] THEN #3000 = 92 (PROBE OBSTRUCTED)
G31 Y[#2 + #30 - #17] F#9
IF [#5090 EQ 0] THEN #3000 = 94 (PROBE NO SKIP SIGNAL)
#11 = #5062 - #5098
G31 Y[#2 + #30 + #5098 + #17] F#9
G31 Z#3 F1000.
IF [#5090 EQ 1] THEN #3000 = 92 (PROBE OBSTRUCTED)
G31 Y[#2 - #30 - #5098 - #17] F1000.
IF [#5090 EQ 1] THEN #3000 = 92 (PROBE OBSTRUCTED)
G31 Z#26 F1000.
IF [#5090 EQ 1] THEN #3000 = 92 (PROBE OBSTRUCTED)
G31 Y[#2 - #30 + #17] F#9
IF [#5090 EQ 0] THEN #3000 = 94 (PROBE NO SKIP SIGNAL)
#12 = #5062 + #5098
G31 Y[#2 - #30 - #5098 - #17] F#9
G31 Z#3 F1000.
G01 Y#2 F1000.
GOTO 70
N60 #138 = ABS[#11 - #12]
#135 = [#11 + #12] / 2
#139 = #135 - #1
#142 = #138 - #24
GOTO 80
N70 #138 = ABS[#11 - #12]
#136 = [#11 + #12] / 2
#140 = #136 - #2
#142 = #138 - #25
N80 IF [#20 LE 0] GOTO 85
IF [ABS[#142] GT #20] THEN #3000 = 95 (OUT OF TOLERANCE)
N85 IF [#19 LE 0] GOTO 90
#30 = #19 - 1
IF [#27 NE 1] GOTO 86
G10 L2 P#19 X[#[5221 + 20 * #30] + #139]
N86 IF [#28 NE 1] GOTO 90
G10 L2 P#19 Y[#[5222 + 20 * #30] + #140]
N90 M99
` },
  { name: 'O9814.nc', text: `O9814 (BORE OR BOSS - D DIAMETER, Z = PROBE DEPTH OUTSIDE A BOSS)
IF [#9 EQ 0] THEN #9 = 300.
IF [#17 EQ 0] THEN #17 = 4.
IF [#7 LE 0] THEN #3000 = 93 (9814 NEEDS D)
#1 = #5001
#2 = #5002
#3 = #5003
#30 = #7 / 2
IF [#29 EQ 1] GOTO 50
G31 X[#1 + #30 + #17] F#9
IF [#5090 EQ 0] THEN #3000 = 94 (PROBE NO SKIP SIGNAL)
#11 = #5061 + #5098
G01 X#1 F#9
G31 X[#1 - #30 - #17] F#9
IF [#5090 EQ 0] THEN #3000 = 94 (PROBE NO SKIP SIGNAL)
#12 = #5061 - #5098
G01 X#1 F#9
G31 Y[#2 + #30 + #17] F#9
IF [#5090 EQ 0] THEN #3000 = 94 (PROBE NO SKIP SIGNAL)
#13 = #5062 + #5098
G01 Y#2 F#9
G31 Y[#2 - #30 - #17] F#9
IF [#5090 EQ 0] THEN #3000 = 94 (PROBE NO SKIP SIGNAL)
#14 = #5062 - #5098
G01 Y#2 F#9
GOTO 80
N50 #4 = #30 + #5098 + #17
G31 X[#1 + #4] F1000.
IF [#5090 EQ 1] THEN #3000 = 92 (PROBE OBSTRUCTED)
G31 Z#26 F1000.
IF [#5090 EQ 1] THEN #3000 = 92 (PROBE OBSTRUCTED)
G31 X[#1 + #30 - #17] F#9
IF [#5090 EQ 0] THEN #3000 = 94 (PROBE NO SKIP SIGNAL)
#11 = #5061 - #5098
G31 X[#1 + #4] F#9
G31 Z#3 F1000.
G31 X[#1 - #4] F1000.
IF [#5090 EQ 1] THEN #3000 = 92 (PROBE OBSTRUCTED)
G31 Z#26 F1000.
IF [#5090 EQ 1] THEN #3000 = 92 (PROBE OBSTRUCTED)
G31 X[#1 - #30 + #17] F#9
IF [#5090 EQ 0] THEN #3000 = 94 (PROBE NO SKIP SIGNAL)
#12 = #5061 + #5098
G31 X[#1 - #4] F#9
G31 Z#3 F1000.
G31 X#1 Y[#2 + #4] F1000.
IF [#5090 EQ 1] THEN #3000 = 92 (PROBE OBSTRUCTED)
G31 Z#26 F1000.
IF [#5090 EQ 1] THEN #3000 = 92 (PROBE OBSTRUCTED)
G31 Y[#2 + #30 - #17] F#9
IF [#5090 EQ 0] THEN #3000 = 94 (PROBE NO SKIP SIGNAL)
#13 = #5062 - #5098
G31 Y[#2 + #4] F#9
G31 Z#3 F1000.
G31 Y[#2 - #4] F1000.
IF [#5090 EQ 1] THEN #3000 = 92 (PROBE OBSTRUCTED)
G31 Z#26 F1000.
IF [#5090 EQ 1] THEN #3000 = 92 (PROBE OBSTRUCTED)
G31 Y[#2 - #30 + #17] F#9
IF [#5090 EQ 0] THEN #3000 = 94 (PROBE NO SKIP SIGNAL)
#14 = #5062 + #5098
G31 Y[#2 - #4] F#9
G31 Z#3 F1000.
G01 X#1 Y#2 F1000.
N80 #135 = [#11 + #12] / 2
#136 = [#13 + #14] / 2
#138 = [ABS[#11 - #12] + ABS[#13 - #14]] / 2
#139 = #135 - #1
#140 = #136 - #2
#142 = #138 - #7
IF [#20 LE 0] GOTO 85
IF [ABS[#142] GT #20] THEN #3000 = 95 (OUT OF TOLERANCE)
N85 IF [#19 LE 0] GOTO 90
#30 = #19 - 1
G10 L2 P#19 X[#[5221 + 20 * #30] + #139] Y[#[5222 + 20 * #30] + #140]
N90 M99
` },
  { name: 'O9832.nc', text: 'O9832 (PROBE ON)\nM99\n' },
  { name: 'O9833.nc', text: 'O9833 (PROBE OFF)\nM99\n' },
];
export const PROBE_CYCLE_NUMBERS = PROBE_CYCLES.map((c) => parseInt(c.name.slice(1), 10));
