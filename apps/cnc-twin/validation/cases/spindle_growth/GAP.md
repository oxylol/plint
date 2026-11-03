# spindle_growth: no case (gap)

Wanted: measured axial thermal growth of a machine-tool spindle versus time at stated speeds
(warm-up and cool-down curves, steady-state growth, time constant), with the machine or spindle
named so that the run can be simulated.

## What was tried (2026-11-03)

* The web-search budget of this workflow run was exhausted (shared limit across the parallel
  agents) before this quantity was reached, so sources could only be found by direct URLs and by
  following the reference lists of open papers already in hand.
* Reference lists of the open-access papers read for the other part-2 quantities were scanned for
  thermal-error work. One candidate turned up: Li, F.; Li, T.; Wang, H.; Jiang, Y., "A temperature
  sensor clustering method for thermal error modeling of heavy milling machine tools", Applied
  Sciences 7 (2017) 82, doi 10.3390/app7010082 (measured spindle thermal error of a heavy milling
  machine). mdpi.com answered HTTP 403 to both fetch tools, and the paper is not in PubMed Central
  (PMC ID converter: "Identifier not found in PMC"), so it could not be read.
* The machine brochures already downloaded for db/machines (Haas, DMG MORI) give no measured growth
  curves; the database's `growth_um_at_max_rpm` values are all marked `src: est` for that reason.

## Why nothing qualified

No document with measured growth-versus-time data could be opened in this run. Values recalled
from memory (typical 10-40 um axial growth of 40-taper spindles) would be disguised estimates and
were not written as a case.

## Where to look next

CIRP Annals / Precision Engineering / IJMTM papers on spindle thermal error with ISO 230-3 tests
(e.g. Bossmanns & Tu 1999, Creighton et al. 2010 micro-milling spindle, Mayr et al. 2012 keynote),
open-access thermal-error papers in Applied Sciences, Sensors or Machines (Sensors is mirrored in
PMC and can be read through Europe PMC), and open datasets of spindle thermal-error measurements.
