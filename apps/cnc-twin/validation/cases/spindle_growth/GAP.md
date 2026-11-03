# spindle_growth: what is covered

Four measured cases (sources in db/bibliography/validation-4.yaml):

* `spindlegrowth-wei2023-vcenter55-ambient-cold`, `-mid`, `-warm`: Z thermal error of a Vcenter-55
  vertical machining centre, spindle idling at 2000, 4000 and 6000 rpm for at least 6 h (ISO 230-3,
  capacitive sensors), one case per ambient band (below 13.2, 13.2-22.7, above 22.7 degC): nine
  warm-up curves, 17-58 um plateaus, digitised from Wei et al., Sensors 2023.
* `spindlegrowth-yang2024-vmc850-constant-speed`: Z thermal deformation of a VMC850 at 2000 and
  4000 r/min for 2 h (eddy-current sensor), Yang et al., Sensors 2024.

Still missing: none of these papers describes the spindle itself (drive, bearings, preload, cooling,
maximum speed), and all measure the whole tool-to-table loop rather than the shaft. A case with a
named spindle unit and its bearing data, a cool-down curve, and a variable-speed (ISO 230-3 spectrum)
profile with a consistent speed record would still be useful.
