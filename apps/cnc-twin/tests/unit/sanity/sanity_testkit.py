"""Helpers of the sanity tests (a module with a unique name: tests have no packages)."""

import copy
from pathlib import Path

from cnctwin.db.yamlio import load_file
from cnctwin.sanity import SanityContext

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def setup_data() -> dict:
    return copy.deepcopy(load_file(FIXTURES / "sanity_setup.yaml").data)


def context_with(mutate=None) -> SanityContext:
    data = setup_data()
    if mutate is not None:
        mutate(data)
    return SanityContext.from_project(data, 0)


def program(*lines: str) -> str:
    return "\n".join(lines) + "\n"


def fixture_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")
