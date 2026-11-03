"""audio.wav of a run directory: 44.1 kHz mono 16-bit PCM, deterministic, calibrated, its samples equal the
rendered pressure windows; what the run directory gives the synthesis."""

import hashlib
import json
import math
import shutil
import struct

import numpy as np
import pytest

from cnctwin.sound import dsp as D
from cnctwin.sound import params as P
from cnctwin.sound.inputs import from_run
from cnctwin.sound.synth import render_audio, render_window


def test_inputs_from_the_synthetic_run(run_dir):
    inp = from_run(run_dir)
    assert inp.end_s == pytest.approx(2.32) and len(inp.frame_t_s) == 2320
    assert inp.teeth.max() == 4 and inp.teeth[0] == 0 and inp.tool_radius_m.max() == pytest.approx(0.005)
    assert len(inp.fast) == 7 and inp.fast[0].dt_s == pytest.approx(8e-6)
    assert inp.env_max_N.shape == (2320, 3) and inp.env_max_N.max() > 100.0
    assert inp.chatter_freq_Hz.max() > 1700.0 and inp.packing_torque_Nm.max() > 0.0
    assert inp.chip_rate_kg_s.max() > 0.0 and inp.coolant_on is not None and inp.coolant_on.any()
    assert [len(j) for j in inp.jets] == [0, 1, 1, 1, 1] and inp.jets[1][0].kind == "flood"
    assert inp.spindle_power_W == pytest.approx(34500.0)                         # from the machine database item
    assert inp.spindle_max_rad_s == pytest.approx(12000 * 2 * math.pi / 60)
    assert {m[3] for m in inp.spindle_modes} >= {"x", "y"} and inp.pumps["flood"] == pytest.approx(750.0)
    assert inp.axis_lead_m["X"] == pytest.approx(0.016) and inp.enclosure_area_m2 > 5.0
    assert inp.chip_youngs_Pa == pytest.approx(68.9e9) and inp.chip_density_kg_m3 == pytest.approx(2700.0)


def test_audio_wav_is_calibrated_deterministic_and_equals_the_windows(run_dir, tmp_path):
    first = render_audio(run_dir)
    wav = (run_dir / "audio.wav").read_bytes()
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE" and b"ICMT" in wav
    assert struct.unpack("<I", wav[4:8])[0] == len(wav) - 8
    samples, rate = D.read_wav(run_dir / "audio.wav")
    assert rate == 44100 and len(samples) == first["samples"] == int(math.ceil(2.32 * 44100))
    summary = json.loads((run_dir / "audio.json").read_text())
    assert summary == json.loads(json.dumps(first))
    full_scale = summary["calibration"]["full_scale_Pa"]
    assert summary["calibration"]["full_scale_sine_dB"] == pytest.approx(116.99, abs=0.01)
    pressure = samples.astype(np.float64) * full_scale / 32767.0
    level = 20 * math.log10(math.sqrt(np.mean(pressure ** 2)) / 2e-5)
    assert level == pytest.approx(summary["levels"]["Leq_dB"], abs=0.05)
    assert summary["clipped_samples"] == 0 and 50.0 < summary["levels"]["LAeq_dB"] < 110.0
    copy = tmp_path / "again"
    shutil.copytree(run_dir, copy)
    (copy / "audio.wav").unlink()
    render_audio(copy)
    assert hashlib.sha256((copy / "audio.wav").read_bytes()).digest() == hashlib.sha256(wav).digest()
    window = render_window(run_dir, 1.1, 1.3)
    quantised, _ = D.quantise(window.pressure_Pa, window.n0, from_run(run_dir).seed, full_scale,
                              bool(P.sound()["dither"]))
    assert np.array_equal(quantised, samples[window.n0:window.n0 + len(quantised)])


def test_inside_listener_and_a_subset_of_sources(run_dir, tmp_path):
    out = tmp_path / "inside.wav"
    summary = render_audio(run_dir, out, sources=("cutting", "chatter"), listener="inside")
    assert out.exists() and (tmp_path / "inside.json").exists()
    assert summary["listener"]["kind"] == "inside" and summary["sources"] == ["cutting", "chatter"]
    operator = render_audio(run_dir, tmp_path / "operator.wav", sources=("cutting", "chatter"))
    assert summary["levels"]["Leq_dB"] > operator["levels"]["Leq_dB"] + 10.0
