# cycle_time: no case (gap)

Wanted: the machining time of a stated NC program or toolpath measured on a real CNC (Fanuc,
Siemens, Heidenhain, Haas), ideally next to the CAM estimate or a feed-drive model, with the
program text or a complete toolpath description and the control's motion settings.

## What was tried (2026-11-03)

* The web-search budget of this workflow run was exhausted (shared limit across the parallel
  agents) before this quantity was reached, so only direct URLs and the reference lists of open
  papers already in hand could be used.
* Matras & Zebala, Materials 13 (2020) 1109 (key matras-2020-h13-cbn-milling, used for the
  roughness cases) gives tool-path lengths and machining times for zig, spiral, concentric and
  Z-level paths with and without feed optimisation (Table 8: 217/190, 238/208, 183/155,
  285/204 s). The text says these times were calculated with the CAM software, not recorded on
  the machine, and the programs are not given: not a measured cycle time.
* Papers known to report recorded cycle times (Heo et al. 2006, Robotics and CIM; So et al. 2007,
  JMPT; Erkorkmaz & Altintas on jerk-limited feed scheduling) are behind publisher paywalls.

## Why nothing qualified

No readable source pairs a complete program or toolpath with a time measured on a named control.

## Where to look next

Open-access Procedia CIRP / NAMRC papers on machining-time prediction with recorded times;
controller makers' application notes on look-ahead and cycle time (Heidenhain, Siemens SINUMERIK
"Advanced Surface", Fanuc AI contour control) when they publish test-part times; or a time
measured in house on one of the database machines with a published test part (e.g. ISO 10791-7).
