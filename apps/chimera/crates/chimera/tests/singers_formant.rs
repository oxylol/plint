//! The epilarynx control makes the singer's formant: narrowed along its tube above the ventricle
//! (not only at its outlet), the epilaryngeal resonance pulls F4 down near 3 kHz and the 2.4-3.4 kHz
//! band (the singer's ring) up against 0.3-0.8 kHz (Sundberg 1974; Titze & Story 1997).
use chimera::airway::frequency::{find_peaks, transfer_function};
use chimera::anatomy::human::{BodyName, NEUTRAL};
use chimera::phonetics::planner::vowel_table;
use chimera::voice::human_airway::HumanAirway;

/// Mean level of the flow transfer function in 2.4-3.4 kHz minus 0.3-0.8 kHz (dB), and each
/// vowel's F4 (Hz), over the adult male's a e i o u at the given epilarynx narrowing.
fn ring_and_f4(epi: f64) -> (f64, Vec<f64>) {
    let mut air = HumanAirway::new(96000.0, BodyName::Male.body());
    let freqs: Vec<f64> = (0..800).map(|i| 50.0 + 10.0 * i as f64).collect();
    let (mut ring, mut f4) = (0.0, vec![]);
    for v in ["a", "e", "i", "o", "u"] {
        let mut a = vowel_table()["male"][v].artic.apply_to(&NEUTRAL);
        a.epilarynx = epi;
        air.apply(&a, 0.0);
        let tf = transfer_function(&air.net.linear_elements(), air.pharynx.first_node(), &freqs, &[air.glottis_port]);
        let band = |lo: f64, hi: f64| {
            let s: Vec<f64> = freqs.iter().zip(&tf.mag).filter(|(f, _)| (lo..hi).contains(*f)).map(|(_, m)| 20.0 * chimera::math::log10(*m)).collect();
            s.iter().sum::<f64>() / s.len() as f64
        };
        ring += (band(2400.0, 3400.0) - band(300.0, 800.0)) / 5.0;
        f4.push(find_peaks(&freqs, &tf.mag, 150.0).get(3).unwrap_or_else(|| panic!("/{v}/ at epilarynx {epi}: fewer than four resonances")).freq);
    }
    (ring, f4)
}

#[test]
fn a_narrowed_epilarynx_puts_the_fourth_formant_near_3_khz() {
    let (r0, f0) = ring_and_f4(0.0);
    let (r1, f1) = ring_and_f4(0.85); // the Sing screen's "Ring on"
    println!("epilarynx 0: ring band {r0:.1} dB, F4 {f0:?}; 0.85: {r1:.1} dB, F4 {f1:?}");
    // Narrowing only the outlet (before) moved F4 down 190-390 Hz, to 3.45-3.65 kHz, and the band
    // +3.5 dB over rest; with the tube it drops 440-700 Hz, to 3.16-3.38 kHz, and +8.8 dB. The
    // margins are relative so a vowel recalibration doesn't flip the test.
    for (v, (a, b)) in ["a", "e", "i", "o", "u"].iter().zip(f0.iter().zip(&f1)) {
        assert!(a - b > 410.0, "/{v}/: F4 {a:.0} Hz at rest, {b:.0} Hz with the ring on");
        assert!(*b < 3500.0, "/{v}/: F4 {b:.0} Hz with the ring on");
    }
    assert!(r1 > r0 + 6.0, "ring band {r0:.1} dB at rest, {r1:.1} dB with the ring on");
}
