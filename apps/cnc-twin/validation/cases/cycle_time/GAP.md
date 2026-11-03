# cycle_time: what is covered

Two cases from Wilkinson, Sencer & Ward, Int. J. Adv. Manuf. Technol. 2025 (source in
db/bibliography/validation-4.yaml), both circular trochoids with the path geometry and the control's
acceleration and jerk limits stated:

* `cycletime-wilkinson2025-trochoid-840d-tnc640` (measured: true): five R = 10 mm circles at 3000 and
  6000 mm/min and 10-100 um tolerance, timed on a Hermle C52 with SINUMERIK 840D sl (G01 path,
  CYCLE832) and on a Heidenhain TNC640 programming station.
* `cycletime-wilkinson2025-trochoid-tnc640-r5-r10` (measured: false, controller-kernel result):
  R = 5 and 10 mm trochoids on the TNC640 interpolator, eight tests.

Still missing: a Fanuc or Haas control, a complete part program (pockets, contours, tool changes,
rapids) timed on the machine next to the CAM estimate, and the 840D segment length of the G01 path.
