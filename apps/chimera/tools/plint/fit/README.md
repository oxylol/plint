# Plint · what the fits are made of

- `ref-fit6.json`: what six real men do (TSP speakers MA MB MC MD MF MG), as `chimera realism
  reference` measures it. The realism judge scores a voice against it, and the fitter climbs it.
  The men MH MI MJ check a fit, and MK ML are read only at a gate; none of the five is in here.
  A copy of the rig's `hill/realism/ref-fit6.json`, which `tools/plint/realism_check.sh` rebuilds
  and checks against what the binary builds. Version 3 (2026-11-26: the consonants above 8 kHz;
  the judge refuses a file of another version, and the rig keeps the last one as
  `ref-fit6.v2.json`).
- `stage1-failed.json`: the first fit's tuning (stage (i), 2026-11-26), kept because it failed:
  it made the source real and let the vowels gain 4-6 dB on every consonant. Check (g) of
  `realism_check.sh` wants the judge to read it worse than the default on `consonants`. Not a voice
  to use. No TSP data.

The reference holds statistics derived from the TSP speech database (Copyright (c) 2018 Peter Kabal,
Simplified BSD). The notice and the licence's conditions are inside the file (`notice`) and in
full in `tools/tsp/README.md`. No audio is here.
