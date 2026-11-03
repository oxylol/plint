"""MachineLog: names, units, times, interpolation, files."""

import numpy as np
import pytest

from cnctwin.core.errors import InputError
from cnctwin.mdata.log import (Channel, LogBuilder, MachineEvent, MachineLog, check_channel_name, channel_unit,
                               conversion, format_time_ns, merge, parse_time_ns)


def test_parse_time_zones_and_fractions():
    base, zone = parse_time_ns("2026-11-03T10:15:30Z")
    assert zone and base == 1_793_700_930 * 10**9
    assert parse_time_ns("2026-11-03T10:15:30.123456789Z")[0] == base + 123_456_789
    assert parse_time_ns("2026-11-03T11:15:30.5+01:00")[0] == base + 500_000_000
    assert parse_time_ns("2026-11-03T05:15:30-0500")[0] == base
    naive, zone = parse_time_ns("2026-11-03 10:15:30,25")
    assert not zone and naive == base + 250_000_000
    assert format_time_ns(base + 1_500_000) == "2026-11-03T10:15:30.0015Z"
    assert format_time_ns(base) == "2026-11-03T10:15:30Z"
    with pytest.raises(InputError):
        parse_time_ns("03/11/2026 10:15")


def test_channel_names_and_units():
    assert channel_unit("spindle.load_frac") == "frac"
    assert channel_unit("axis.X.pos_machine_m") == "m"
    assert channel_unit("path.feed_m_s") == "m_s"
    assert channel_unit("temp.spindle_K") == "K"
    assert channel_unit("program.line") == ""
    for good in ("spindle.power_W", "axis.A.pos_machine_rad", "temp.axis_x_motor_K", "diag.d300.X", "modal.s"):
        check_channel_name(good)
    for bad in ("SpindleLoad", "axis.X.velocity_m", "spindle load"):
        with pytest.raises(InputError):
            check_channel_name(bad)


def test_unit_conversions():
    assert conversion("pct", "spindle.load_frac") == (0.01, 0.0)
    assert conversion("%", "axis.X.load_frac") == (0.01, 0.0)
    factor, _ = conversion("rpm", "spindle.speed_rad_s")
    assert factor == pytest.approx(2 * np.pi / 60)
    assert conversion("degC", "temp.spindle_K") == (1.0, 273.15)
    assert conversion("kW", "spindle.power_W") == (1000.0, 0.0)
    assert conversion("mm/min", "path.feed_m_s")[0] == pytest.approx(1e-3 / 60)
    assert conversion("mm/s", "path.feed_m_s")[0] == pytest.approx(1e-3)
    assert conversion(None, "spindle.power_W") == (1.0, 0.0)
    with pytest.raises(InputError):
        conversion("mm", "spindle.power_W")          # length into a power channel
    with pytest.raises(InputError):
        conversion("furlongs", "axis.X.pos_machine_m")


def test_sample_and_event_interpolation_with_gaps():
    sample = Channel("spindle.power_W", np.array([0.0, 1.0, 2.0, 10.0]), np.array([0.0, 10.0, np.nan, 30.0]))
    t = np.array([-1.0, 0.5, 1.5, 5.0, 10.0, 11.0])
    values = sample.at(t)
    assert np.isnan(values[0]) and np.isnan(values[-1])
    assert values[1] == pytest.approx(5.0)
    assert values[3] == pytest.approx(10.0 + 20.0 * 4.0 / 9.0)      # NaN dropped: 1.0 -> 10.0 interpolated
    gapped = sample.at(t, max_gap_s=2.0)
    assert np.isnan(gapped[3]) and gapped[1] == pytest.approx(5.0)
    event = Channel("program.line", np.array([0.0, 1.0, 3.0]), np.array([5.0, 6.0, 9.0]), "event")
    assert event.at(np.array([-0.5, 0.0, 2.9, 3.0, 99.0])).tolist()[1:] == [5.0, 6.0, 9.0, 9.0]
    text = Channel("program.block", np.array([0.0, 1.0]), np.array(["G00 X0", None], dtype=object), "event")
    assert text.at(np.array([0.5, 2.0])).tolist() == ["G00 X0", "G00 X0"]
    assert sample.resolution() == pytest.approx(10.0)


def test_builder_epoch_order_and_cap():
    builder = LogBuilder({"kind": "test"})
    base = parse_time_ns("2026-11-03T10:00:00Z")[0]
    builder.add("spindle.power_W", base + 2_000_000_000, 20.0)
    builder.add("spindle.power_W", base + 1_000_000_000, 10.0)
    builder.add("program.name", base + 1_500_000_000, "O1000")
    builder.event(base + 1_000_000_000, MachineEvent(0.0, "alarm", "PS0010", "IMPROPER G-CODE", "fault"))
    log = builder.build()
    assert log.epoch_ns == base + 1_000_000_000
    assert log.channels["spindle.power_W"].t_s.tolist() == [0.0, 1.0]
    assert log.channels["spindle.power_W"].values.tolist() == [10.0, 20.0]
    assert log.events[0].t_s == 0.0
    with pytest.raises(InputError):
        builder.build(max_observations=2)
    with pytest.raises(InputError):
        builder.add("spindle.power_W", 1.5, 1.0)       # relative time into an absolute log


def test_save_load_roundtrip_is_exact_and_deterministic(tmp_path):
    log = MachineLog(epoch_ns=parse_time_ns("2026-11-03T10:00:00Z")[0], source={"kind": "test"}, meta={"note": "x"})
    log.add("spindle.power_W", np.array([0.0, 0.1, 0.2]), np.array([1.0, np.nan, 3.25]))
    log.add("program.block", np.array([0.0, 0.15]), np.array(["G01 X1.", None], dtype=object))
    log.events.append(MachineEvent(0.1, "message", "3006", "CHECK TOOL", "info", "controller"))
    log.save(tmp_path / "a")
    log.save(tmp_path / "b")
    for name in ("machine_log.json", "machine_log.parquet"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    back = MachineLog.load(tmp_path / "a")
    assert back.epoch_ns == log.epoch_ns
    np.testing.assert_array_equal(back.channels["spindle.power_W"].values, log.channels["spindle.power_W"].values)
    assert back.channels["program.block"].values.tolist() == ["G01 X1.", None]
    assert back.events[0].text == "CHECK TOOL"


def test_merge_puts_logs_on_one_epoch():
    a = MachineLog(epoch_ns=1_000_000_000)
    a.add("spindle.power_W", np.array([0.0, 1.0]), np.array([1.0, 2.0]))
    b = MachineLog(epoch_ns=3_000_000_000)
    b.add("spindle.power_W", np.array([0.0]), np.array([3.0]))
    b.add("tool.number", np.array([0.5]), np.array([4.0]))
    merged = merge([a, b])
    assert merged.epoch_ns == 1_000_000_000
    assert merged.channels["spindle.power_W"].t_s.tolist() == [0.0, 1.0, 2.0]
    assert merged.channels["tool.number"].t_s.tolist() == [2.5]
    with pytest.raises(InputError):
        merge([a, MachineLog()])
