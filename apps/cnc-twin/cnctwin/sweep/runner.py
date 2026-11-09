"""The simulation entry that applies operator overrides: cnctwin.core.loop.run_project plus the overrides of
``options["operator"]`` (cnctwin.sweep.operator). The sweeps, ``twin run`` (when its options carry ``operator``) and
the API's default ``simulate`` entry (cnctwin.api.settings) run through it.

* :func:`run_project` ``(project_path, run_dir, options, progress_cb) -> dict``: without overrides (or with every
  override at 100 %) exactly the loop's run_project, so its runs share the loop's cache keys; with overrides the
  loop's ``Run`` with an :class:`~cnctwin.sweep.operator.OperatorHook` on every operation's motion domain.
* :func:`replay_window` ``(run_dir, tick0, tick1, options, progress_cb) -> dict``: ``twin replay --window`` of a run
  made with overrides (the CLI dispatches here when run.json ``options.operator`` is set): the loop's replay with the
  same hook, so the replayed window is the window of that run.

Overrides the project cannot take (:func:`cnctwin.sweep.operator.problems`: outside the machine's override ranges or
rapid keys, a tool the project does not have, a spindle override of a tap) are refused before the run starts
(InputError naming each). A run with overrides whose hook saw no command batch is refused (TwinError): the overrides
would have been ignored.
"""

from __future__ import annotations

import contextlib
import threading

from ..core.errors import InputError, TwinError
from .operator import OperatorHook, Overrides, normalise, problems

_PATCH_LOCK = threading.Lock()


def _run_class(overrides: Overrides):
    from ..core import loop

    class OverrideRun(loop.Run):
        """The loop's Run with the operator hook installed on every motion domain it creates."""

        instances: list = []

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            refused = problems(self.options.get("operator"), self.project)
            if refused:
                raise InputError("options.operator cannot be applied to this project: " + "; ".join(refused))
            self.operator_hook = OperatorHook(overrides, self.project)
            OverrideRun.instances.append(self)

        def _new_motion(self, op_index):
            motion = super()._new_motion(op_index)
            self.operator_hook.install(motion)
            return motion

    return OverrideRun


def _checked(run, result):
    hook = run.operator_hook
    if hook.calls == 0:
        raise TwinError("operator overrides were set but no command reached the hook (the loop's Run no longer "
                        "creates its motion domain through _new_motion): the run would ignore them")
    out = dict(result) if isinstance(result, dict) else {"result": result}
    out["operator"] = hook.summary()
    return out


@contextlib.contextmanager
def _patched_run(cls):
    """The loop's replay builds ``Run`` by its module name: it builds ``cls`` while this holds."""
    from ..core import loop
    with _PATCH_LOCK:
        original = loop.Run
        loop.Run = cls
        try:
            yield
        finally:
            loop.Run = original


def run_project(project_path, run_dir, options=None, progress_cb=None) -> dict:
    from ..core import loop
    options = dict(options or {})
    operator = normalise(options.get("operator"))
    if operator is None:
        options.pop("operator", None)
        return loop.run_project(project_path, run_dir, options, progress_cb)
    options["operator"] = operator
    overrides = Overrides.from_options(options)
    cls = _run_class(overrides)
    from pathlib import Path
    project_path = Path(project_path)
    if project_path.is_dir():
        project_path = project_path / "project.yaml"
    if options.get("replay"):
        with _patched_run(cls):
            result = loop.run_project(project_path, run_dir, options, progress_cb)
        if not cls.instances:
            raise TwinError("the loop's replay did not build its Run through cnctwin.core.loop.Run: the replay would "
                            "ignore the operator overrides")
        return _checked(cls.instances[-1], result)
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    run = cls(project_path, Path(run_dir), options, progress_cb)
    return _checked(run, run.execute())


def replay_window(run_dir, tick0: int, tick1: int, options=None, progress_cb=None) -> dict:
    """``twin replay --window`` of a run made with operator overrides (cnctwin.core.outputs.replay_window with
    :func:`run_project` as the runner, which reads the run's own options, overrides included)."""
    from ..core.outputs import replay_window as replay
    return replay(run_dir, int(tick0), int(tick1), options, progress_cb, runner=run_project)


__all__ = ["run_project", "replay_window"]
