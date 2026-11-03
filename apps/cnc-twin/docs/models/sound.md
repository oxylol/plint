# Sound (`cnctwin.sound`)

What the operator in front of the closed machine (or a microphone inside the enclosure) would hear during the
run, synthesised sample by sample from the simulated physics: tooth passing and its harmonics with the force
fluctuation of the cut, chatter at the frequency the dynamics found, spindle unbalance and bearings, spindle and
axis motors, the coolant pump, splashing coolant and air jets, chips rattling on the enclosure, chips packed in
the flutes; silence of the cut in air. Output: `audio.wav` (44.1 kHz, 16-bit PCM, calibrated in pascal) and
`audio.json`; any time window can be rendered alone and equals the same samples of the whole run.

Package: `inputs.py` (what is read from the run directory: `SoundInputs`), `synth.py` (`Synth`, `render_window`,
`render_audio`), `acoustics.py` (radiation, propagation and source formulas), `dsp.py` (FIR design, block
convolution, WAV), `kernels.py` (Numba), `reference.py` (plain-Python twins), `params.py`, `__main__.py`.
Coefficients: `db/models/sound.yaml` (acoustics, listener, enclosure, structures, DSP) and
`db/models/sound-sources.yaml` (sources); sources: `db/bibliography/models-sound.yaml`.

```
python -m cnctwin.sound RUN_DIR                                   # audio.wav + audio.json
python -m cnctwin.sound RUN_DIR --t0 1.0 --t1 1.5 --out w.wav     # one window
python -m cnctwin.sound RUN_DIR --listener inside --sources cutting,chatter
```

## 1. Signal chain

```
 forces at the tool (T0, N)           contact: cutting, chatter, packing ---> SEA work + tool side + tool dipole --+
                                      spindle: unbalance, bearings ------> SEA tool side ----------------------------+
 airborne, cutting zone (Pa at r)     splash, air jets (band noise) ------------------------------------------------+--> x enclosure transmission (operator)
 forces on the enclosure sheet (N)    chip impacts (Hertz pulses) -------> SEA sheet, one side ----------------------> outside bus
 airborne, outside the enclosure      spindle motor, axis motors, pump ------------------------------------------------> outside bus
                                                                                                    p(t) = inside + outside (Pa)
```

Every force path is a zero-phase FIR filter whose magnitude is the pressure at the listener per newton of force
at that frequency; airborne sources are tones with a known phase and band noise with a known power, scaled to the
pressure their sound power gives at the listener.

### Radiation and propagation (`acoustics.py`)

* **Structure-borne (SEA, Lyon & DeJong 1995).** A force F at a plate-like structure (bending stiffness
  `B = E h^3 / (12 (1 - nu^2))`, surface mass `m'' = rho h`) injects `W_in = F_rms^2 Re(Y)` with the
  infinite-plate point mobility `Y = 1 / (8 sqrt(B m''))` (real, frequency independent: the frequency average of
  a finite plate's drive-point mobility; Cremer, Heckl & Petersson 2005, 5.4). The structure radiates the share
  `eta_rad / (eta_rad + eta_int)` with `eta_rad = rho0 c0 sigma / (omega m'')`; sigma is the radiation efficiency
  of the sphere of the same area, `sigma = (ka)^4 / (4 + (ka)^4)`, `a = sqrt(S / 4 pi)` (exact for a rigid
  oscillating sphere, Kinsler et al. 2000, Fahy & Gardonio 2007). Two such structures take the contact forces
  (action and reaction): the **work side** (table, vise and part: 60 mm cast iron, 1.4 m^2, loss factor 0.02)
  and the **tool side** (spindle head: 30 mm, 1.0 m^2). Spindle forces (unbalance, bearings) drive the tool
  side only.
* **Tool dipole.** The tool, holder and spindle nose also radiate their own vibration as a compact rigid sphere
  of radius 25 mm: `W = rho0 c0 (4 pi a^2 / 3) sigma U_rms^2`, `U = omega |H(omega)| F`, H the modal compliance of
  the machine's spindle `frf_modes` in that direction (x, y; z from the structure's z modes). Checked: the power
  from sigma equals the integral of the far-field intensity of `p = rho0 a^3 omega^2 U cos(theta) / (2 c0 r)`
  (Pierce 2019) to 0.1 %.
* **Listener.** `p_rms^2 = rho0 c0 W (Q / (4 pi r^2) + 4 / R)` (Hopkins & Stryker 1948), Q = 2 (above a
  reflecting floor), room constant `R = S a / (1 - a)` of the shop bay (400 m^3, 340 m^2, a = 0.12). Distances:
  operator 1 m in front of the machine (ISO 230-5), cutting zone 0.5 m behind the door, spindle motor 1.8 m,
  pump 2 m. The **inside** listener is 0.5 m from the cut and hears the enclosure's own reverberant field
  (R from the enclosure surface, a = 0.06).
* **Enclosure** between the cutting zone and the operator (or between the outside sources and the inside
  listener): field-incidence mass law of the 1.5 mm steel sheet and the 6 mm polycarbonate window
  (`TL = 10 log10(1 + (pi f m'' / (rho0 c0))^2) - 5 dB`, Bies & Hansen 2003), leaks (0.2 % open area, transmission
  1) and the interior absorption: `IL = TL_composite + 10 log10(a_enclosure)`, not below 0 (Beranek & Ver 1992).
  At 1 kHz this gives about 14 dB; leaks cap it near 27 dB.

## 2. Sources (`synth.py`)

All phases are functions of the simulated positions (stateless): the spindle angle comes from frames.bin
(unwrapped `angle + 2 pi rev`, cubic Hermite between the 1 ms records with central-difference slopes), a ballscrew
motor turns by `2 pi x / lead`, a fixed-frequency source by `2 pi f t`.

| source | model | parameters (est unless cited) |
|---|---|---|
| **cutting** | force on the tool, T0, x y z. **Fast tier** (`cutting.force_N` at L0) where recorded: band-limited (Blackman windowed sinc, 20 kHz, +-48 samples) resampling onto the audio grid, 2 ms raised-cosine crossfade at the window edges. **Elsewhere** the tick-tier `min/mean/max` of each component (smoothed over 5 ms: the per-tick statistics of a tooth-passing force beat with the 1 kHz tick rate) shaped by the tooth-period waveform `w(phi)` of the newest fast window at or before (else the first after): `F = mean + (max - min)(w(phi) - mean(w))`, `phi = N sign(omega) theta(t)` (64 phase bins, synchronous average). With no window at all: a raised-cosine pulse whose duty `d = 2 (mean - min)/(max - min)` reproduces the mean (light immersion: short pulses, rich harmonics; d > 1: the inverted pulse). No tool or zero envelope: zero force — **air cutting has no tooth pass**. | Altintas 2012 (tooth-passing forces) |
| **chatter** | where no fast window covers the time: vibration amplitude `x = 0.5 vibration_pp sqrt(chatter_index)` (the non-forced share of the vibration energy) at `chatter_freq_Hz`, phase integrated over the macro steps; the force that drives the dominant tool mode (the spindle mode nearest the median chatter frequency) to it, `F = x |k (1 - r^2 + 2 i zeta r)|`, rotating in x-y | dynamics channels; default mode 900 Hz, 50 N/um, zeta 0.03 |
| **unbalance** | `F = m G omega^2 / omega_max` rotating with theta (residual unbalance of grade G at the maximum speed; omega_max from the machine file, else 10 000 rpm) | m 12 kg; G 2.5 mm/s (ISO 21940-11) |
| **bearings** | varying compliance at the outer-race ball passage, phase `Z (1 - (D_w/d_m) cos alpha)/2 theta`, force `epsilon F_preload` | 20 balls, 11 / 90 mm, 15 deg (Harris & Kotzalas 2006); 1000 N, epsilon 0.02 (Sunnersjo 1978) |
| **packing** | tangential force `fluctuation x T_packing / r_tool` x unit band noise 300-6000 Hz | `chips.packing_torque_Nm`; fluctuation 0.5 |
| **spindle_motor** | sound power from `L_p(1 m) = 17 + 17 log10(P/kW) + 15 log10(n/rpm)` (P the peak of the machine's power curve, else 15 kW; P < 40 kW; `28 + 10 log10 P + 15 log10 n` above), n capped at 3600 rpm (the formula's data), -15 dB for a liquid-cooled fanless motor in the head; 30 % in the magnetic tone at `2 f_e` (`f_e = p n`), 15 % in the inverter tones `f_sw +- 2 f_e`, the rest pink noise 100 Hz - 8 kHz | Bies & Hansen 2003 (pp. 553-554); p = 2, f_sw 8 kHz (Gieras et al. 2006) |
| **axis_motors** | the same formula per linear axis at the motor speed `2 pi |v| / lead`, -10 dB (fanless servo motors behind covers); rotary axes as direct drives | 4 kW, p = 4, f_sw 10 kHz; lead from the machine file (16 mm default) |
| **pump** | `L_p(1 m) = 72 + 10 log10(P/kW)` while liquid coolant flows (the flood or through-spindle pump of the jets' kind), 30 % at the blade passage (6 vanes, 2900 rpm), the rest noise 200 Hz - 5 kHz | Bies & Hansen 2003 (pp. 523-524); pump power from the machine file |
| **splash** | `W = eta 0.5 rho Q U^2` of the liquid jets, band (Gaussian in log2 f, +-1.5 octaves) at the Minnaert frequency `f = sqrt(3 gamma p0 / rho) / (2 pi a)` of 1.5 mm bubbles (2.2 kHz) | eta 1e-6 (no measured value found, +-10 dB); Minnaert 1933 |
| **air_jets** | air blast and MQL: `W = K M^5 (1/2) rho0 U^3 pi D^2 / 4` (Lighthill 1952, K = 1e-4), hump at `f = 0.2 U / D` +-2 octaves | Fitzpatrick & Lee 1952 (Strouhal) |
| **impacts** | one chip per tooth pass while cutting, a share 0.5 striking the enclosure sheet; per 0.1 ms sub-slot a Bernoulli draw `u < rate x slot`; impulse `m (1 + e) v` (mass = chip mass rate / chip rate from `chips.emitted_kg`, else 2 mg; speed = the chip file's median, else 0.5 x the cutting speed omega r, spread uniformly over +-50 %), a Hertzian half-sine of duration `2.87 (m^2 / (R E*^2 v))^(1/5)` (Johnson 1985) integrated with 8 band-limited midpoints, on the sheet (SEA of 1.5 mm steel, half its radiation to the listener's side) | e 0.4; chip modulus and density from the material file (Akay 1978) |

## 3. Inputs (`inputs.py`)

| file / channel | use |
|---|---|
| frames.bin (scene.json header) | spindle angle (path 0), axis values per L1 tick |
| timeline.json + scene.json `tools[]` | tooth count `teeth` and radius `diameter_m / 2` of the active tool |
| series tick tier `cutting.force_N[i].min/.mean/.max` (or `cutting.force_T0_N`, `cutting.tool_force_N`) | envelopes |
| series fast tier, the same channel | full-rate force windows |
| series macro `dynamics.chatter_freq_Hz`, `dynamics.chatter_index`, `dynamics.vibration_pp_m` | chatter |
| `chips.packing_torque_Nm`, `chips.emitted_kg` (else `removal.removed_volume_m3` x density), `coolant.reach_fraction` | packing, chip mass rate, coolant on |
| keyframes `jet_*` (+ scene nozzles' `diameter_m`), chips `vel_M_m_s` | jets (kind, speed, flow, bore), chip speed |
| run.json project + `db/machines/<id>.yaml` (raw, the run's values win) | spindle power curve and maximum speed, `frf_modes`, structure z modes, ballscrew pitch, axis `motor_power_W`, coolant pumps, enclosure box |
| material and coolant database items | chip modulus, Poisson ratio, density; liquid density |

A missing channel silences its source and is listed in `SoundInputs.notes` (and `audio.json`). `SoundInputs` can
also be built by hand (tests, studies).

## 4. Levels and the WAV file

* **Calibration.** Sample value `s = round(32767 p / p_fs)` with `p_fs = full_scale_Pa = 20 Pa`: one LSB is
  0.61 mPa, a full-scale sine is 14.1 Pa rms = **117.0 dB SPL** (re 20 uPa, ISO 1683), the 16-bit floor about
  21 dB SPL. To get pascal from the file: `p = s x 20 / 32767`. Triangular dither of +-1 LSB before rounding
  (Lipshitz et al. 1992), from counter-based noise (`sound.dither`, by sample index). Samples beyond full scale are
  clipped and counted (`clipped_samples`), never rescaled silently.
* `audio.wav`: RIFF/WAVE, mono, 16-bit PCM, 44 100 Hz (IEC 60908), the calibration in a LIST/INFO ICMT comment
  (RIFF 1.0, ignored by players). `audio.json` (`format: cnctwin-audio`): calibration, listener, sources, levels
  (Leq, LAeq with the IEC 61672-1 A-weighting applied as a FIR, Leq of the inside and outside buses, peak), clipped
  samples, notes.
* The synthetic run (Al 6061, D10 4-flute at 10 400 rpm, flood): about 81 dB Leq at the operator, the spindle motor
  80 dB, the cut 68-69 dB through the enclosure, chatter 1927 Hz and the tooth passing at 696 Hz with harmonics.

## 5. Signal processing and determinism (`dsp.py`, `kernels.py`)

* Filters: zero-phase FIR, 4097 taps (10.8 Hz resolution), designed by frequency sampling of the real gain on a
  16 384-point grid and Blackman-windowed (Oppenheim & Schafer 2010, 7.5). Noise shapers are normalised to unit
  output power; the enclosure transmission is folded into the shapers and force filters of the sources it applies
  to (tones are scaled by `sqrt(T(f))` at their instantaneous frequency).
* **Fixed block grid.** Blocks of 16 384 samples at absolute indices; output block J is the sum, in this order,
  of the centred FFT convolutions of input blocks J-1, J, J+1 (taps - 1 < 2 B). Every source is a stateless
  function of the absolute sample index (positions, angles, counter-based noise `hash_normal(seed, stream, n, 0,
  0)`, impacts drawn per sub-slot index). Hence `render_window(t0, t1)` equals the same samples of the whole run
  bit for bit (tested on block edges), and the WAV equals the quantised windows.
* Kernels (explicit signatures, one specialisation each): `hash_noise`, `hash_uniforms`, `hash_tpdf`,
  `sinc_resample`, `place_impulses`, `hermite`, with plain-Python twins (direct sums, SciPy's Hermite spline).
* RNG streams: `sound.spindle_motor`, `sound.axis_motor.<axis>`, `sound.pump`, `sound.splash`, `sound.air_jets`,
  `sound.packing`, `sound.impacts`, `sound.dither`.
* Cost: about 0.3 s of one core per simulated second (all sources); memory a few blocks (~10 MB) whatever the run
  length (about 15 MB: block caches and filter spectra; the WAV is streamed).

## 6. Validity and error bounds

| quantity | bound | basis |
|---|---|---|
| frequencies (tooth passing, harmonics, unbalance, bearing, motor, inverter, pump tones) | exact to the spindle speed of the run (Hermite between 1 ms records; < 1e-9 rad for constant speed); bearing frequencies +-2 % (rolling-element slip, Randall & Antoni 2011) | kinematics |
| chatter frequency | as the dynamics channel (per macro step, 10 ms); its amplitude is an estimate (non-forced share, inverse receptance of one mode): +-10 dB | model |
| tooth-pass harmonic amplitudes | fast windows: the recorded force; elsewhere the newest window's waveform scaled to the tick envelope (error grows if the immersion changes inside a block without a window), or the pulse model (shape approximate, mean and peak-to-peak exact) | construction |
| structure-borne levels | SEA with infinite-plate mobility and sphere radiation: +-10 dB (plates below coincidence radiate less than the sphere predicts; modal peaks of a real machine are not resolved: the frequency average is used) | Lyon & DeJong; Fahy & Gardonio |
| motor and pump levels | +-5 dB inside the formulas' range (Bies & Hansen); spindle motors above 3600 rpm and fanless motors carry the extra offsets' uncertainty (+-10 dB) | empirical |
| splash | +-10 dB (assumed acoustic efficiency) | est |
| air jets | +-3 dB for K in 3e-5 .. 3e-4 | Lighthill |
| enclosure insertion loss | +-5 dB (mass law, assumed leak area and absorption) | Beranek & Ver |
| level at the listener | sum of the above: an informed estimate (+-10 dB overall) | - |
| time | exact (sample index = simulated time x 44 100) | construction |

What the sound is not: a measurement-grade prediction of machine noise emission (ISO 230-5); it is the physics
of the run made audible with calibrated, cited estimates — the frequencies, their timing and their changes
(entry and exit, air cutting, chatter onset, speed changes, rapids, coolant) are exact to the simulation, the
absolute levels are estimates within the bounds above.

## 7. Decisions and their consequences

* **Force-driven synthesis with SEA gains** (not a vibro-acoustic FE model): every source the run records turns
  into sound with cited relations and no tuning to a recording. Consequence: levels are estimates (+-10 dB);
  the machine's own modal colouring is averaged out.
* **Contact forces drive both sides, spindle forces the tool side**: action and reaction act on part and tool.
  Consequence: cutting is louder than unbalance of the same force amplitude.
* **Motor formula capped at 3600 rpm**: no extrapolation of an industrial-motor fit to spindle speeds.
  Consequence: the motor's level stops rising above 3600 rpm; its tones still follow the speed.
* **Zero-phase filters on a fixed block grid**: windows equal the whole run exactly. Consequence: filters are
  non-causal by half their length (46 ms): a sound can begin 46 ms before its cause in the audio, below audible
  pre-echo for these smooth gains.
* **One microphone, mono, no reverberation tail**: the reverberant field is in the level (room constant), not in
  time. Consequence: the audio sounds drier than a real shop.
* **Doors closed**: the enclosure transmission applies throughout. Consequence: the open-door sound of set-up
  phases is not modelled.

## 8. Tests (`tests/unit/sound/`)

`test_spectrum.py`: peaks at the tooth-passing frequency, its harmonic and the chatter frequency of a synthetic
signal; no tooth-pass line in air (exactly zero for the cutting source; the mix has only its noise floor there)
while the unbalance line stays; lighter immersion gives stronger harmonics; the fast tier resampled within 0.02 N
of the recorded sine with zero force outside the window; any window equals the whole run's samples; sources and
listeners checked, the enclosure attenuates the cut. `test_acoustics.py`: sphere radiation limits and the dipole
far field, plate mobility and SEA slopes (f^3 compact, 1/f damped), room equation, mass law (+6 dB/octave),
enclosure bounded by its leaks, A-weighting against the IEC table, motor and pump formulas, Lighthill U^8,
Minnaert 3.3 kHz mm, Hertz v^-1/5, BPFO, unbalance, chatter force. `test_kernels.py`: every kernel against its
reference, noise statistics and index-determinism, FIR and block convolution equal to direct convolution, one
signature each. `test_audio_file.py` (synthetic run): inputs read, audio.wav format, calibration (level of the
samples equals audio.json's Leq within 0.05 dB), determinism (identical bytes from a copy), the WAV equals the
quantised windows, the inside listener.

## References

Kinsler, Frey, Coppens & Sanders 2000; Pierce 2019; Fahy & Gardonio 2007; Cremer, Heckl & Petersson 2005; Lyon &
DeJong 1995; Beranek & Ver 1992; Bies & Hansen 2003 (via Arup Strutt 2026 for the formulas); Hopkins & Stryker
1948; Lighthill 1952; Fitzpatrick & Lee 1952; Johnson 1985; Akay 1978; Minnaert 1933; ISO 21940-11:2016; Sunnersjo
1978; Harris & Kotzalas 2006; Randall & Antoni 2011; Gieras, Wang & Lai 2006; Altintas 2012; Delio, Tlusty & Smith
1992; IEC 61672-1:2013; ISO 1683:2015; ISO 9613-1:1993; ISO 230-5:2000; IEC 60908:1999; RIFF 1.0 (1991); Oppenheim &
Schafer 2010; Lipshitz, Wannamaker & Vanderkooy 1992 (full entries in `db/bibliography/models-sound.yaml`).
