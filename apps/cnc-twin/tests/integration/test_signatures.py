"""(h) Every compiled kernel reached from core.hotloop.run_macro has exactly one signature (architecture 5.5): the
call graph is walked from run_macro's code (global names and module attributes that are Numba dispatchers),
recursively into every callee, and checked after a real step has run (no lazy second specialisation)."""

import dis
import types

import pytest
from numba.core.registry import CPUDispatcher

from cnctwin.core import hotloop
from cnctwin.core.drive import OperationDriver
from tests.integration import hotfix as HF


def _callees(disp) -> set:
    """Dispatchers named in a jitted function's code: globals and module attributes (``mod.fn``)."""
    func = disp.py_func
    found = set()
    g = func.__globals__
    code = func.__code__
    stack = [code]
    names = []
    while stack:
        co = stack.pop()
        ins = list(dis.get_instructions(co))
        for i, op in enumerate(ins):
            if op.opname in ("LOAD_GLOBAL", "LOAD_NAME"):
                names.append((op.argval, None))
            elif op.opname in ("LOAD_ATTR", "LOAD_METHOD") and i > 0 and ins[i - 1].opname == "LOAD_GLOBAL":
                names.append((ins[i - 1].argval, op.argval))
        stack.extend(c for c in co.co_consts if isinstance(c, types.CodeType))
    for name, attr in names:
        obj = g.get(name)
        if attr is not None and isinstance(obj, types.ModuleType):
            obj = getattr(obj, attr, None)
        if isinstance(obj, CPUDispatcher):
            found.add(obj)
    return found


def reached(root) -> list:
    seen = {}
    todo = [root]
    while todo:
        d = todo.pop()
        key = (d.py_func.__module__, d.py_func.__qualname__)
        if key in seen:
            continue
        seen[key] = d
        todo.extend(_callees(d))
    return sorted(seen.items())


def test_the_call_graph_is_found():
    names = {q for (_, q), _ in reached(hotloop.run_macro)}
    for expected in ("motion_tick", "cut_step", "turn_step", "modal_begin", "modal_correct", "modal_free",
                     "tooth_pass", "polar_cut", "helix_subtract", "l1_contact", "collide_envelope", "sweep_contact",
                     "engaged_intervals", "ring_begin", "refill_planes", "fk_poses", "record_tick"):
        assert expected in names, expected


def test_every_kernel_reached_from_run_macro_has_one_signature(tmp_path):
    drv = OperationDriver(HF.slot_project(tmp_path, 1.0, x1_mm=2.0), use_dynamics=True, dynamics_l2=False)
    while drv.state.clock.tick * 1e-6 < 4.3:          # into the cut: every path of the step runs once
        drv.step()
    kernels = reached(hotloop.run_macro)
    assert len(kernels) >= 40
    bad = [(m, q, len(d.signatures)) for (m, q), d in kernels if len(d.signatures) != 1]
    assert not bad, bad
