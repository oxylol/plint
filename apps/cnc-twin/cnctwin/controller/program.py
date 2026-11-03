"""Parsed NC programs: immutable data the executor walks with an explicit program counter.

A program file is split into :class:`Program` objects (several per file, separated by ``O``
numbers or ``%``); each program is a tuple of :class:`Block`. Parsing never raises for a
program error: a block that cannot be read carries ``error = (alarm key, detail)`` and the
executor raises that alarm when (and only if) control reaches the block, as the real control
does. Nothing here is mutable, so the store is rebuilt from the program texts on resume and
the executor state only holds indices into it (``ProgramStore.digest`` guards the match).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Word:
    """One address word. Either a literal (``value``, ``point``) or a macro expression (``expr``)."""

    letter: str
    value: float | None = None       # literal value as written (sign included), program units
    point: bool = False              # the literal had a decimal point
    expr: tuple | None = None        # expression AST (cnctwin.controller.expr); value from a variable
    text: str = ""                   # literal text after the letter ("54.1", "-.5")
    column: int = 0


@dataclass(frozen=True, slots=True)
class Block:
    line: int                        # 1-based line in the file
    text: str                        # the block as written (without end-of-line)
    n: int | None = None             # sequence number
    delete: int = 0                  # block delete switch (1..9) of a leading '/', 0 = none
    words: tuple[Word, ...] = ()
    mid_delete_at: int = -1          # index into words where a mid-block '/n' starts (-1 = none)
    mid_delete: int = 0              # its switch number
    stmt: tuple | None = None        # macro statement AST
    comment: str = ""                # text of the ( ) comments, joined by a space
    error: tuple[str, str] | None = None   # (alarm key, detail) raised when the block is executed
    terms: int = 0                   # operators, functions and variable accesses (macro block-time model)
    o_number: int | None = None      # program-number block (O1234)
    lowercase: bool = False          # the block contained lower-case address letters (converted)
    extra: object = None             # dialect-specific payload (non-ISO dialects)


@dataclass(frozen=True)
class Program:
    number: int | None               # O number (None for an unnumbered main program)
    name: str                        # "O1000", or the file name for an unnumbered program
    file: str                        # file name as given in the project
    blocks: tuple[Block, ...]
    n_index: dict = field(default_factory=dict)   # sequence number -> tuple of block indices (ascending)

    def find_n(self, n: int, start: int = 0) -> int | None:
        """Index of the block with sequence number ``n``: the first at or after ``start``, else the
        first from the beginning (the control searches forward, then from the top)."""
        indices = self.n_index.get(n)
        if not indices:
            return None
        for index in indices:
            if index >= start:
                return index
        return indices[0]


@dataclass(frozen=True)
class ProgramSource:
    name: str                        # file name as given in the project
    text: str
    path: int = 0                    # controller path whose main program this is (-1: library only)
    main: bool = True                # the first program of this file is the path's main program


class ProgramStore:
    """Every program the control can call (main files of every path, then library files)."""

    def __init__(self, programs: list[Program], mains: dict[int, int], digest: str):
        self.programs = programs
        self.mains = mains            # path -> index of its main program
        self.digest = digest
        self._by_number: dict[int, int] = {}
        for index, program in enumerate(programs):
            if program.number is not None and program.number not in self._by_number:
                self._by_number[program.number] = index

    def by_number(self, number: int) -> int | None:
        return self._by_number.get(int(number))

    def __len__(self) -> int:
        return len(self.programs)


def source_digest(sources: list[ProgramSource]) -> str:
    """SHA-256 over the program sources in order (a resumed run must use the same texts)."""
    hasher = hashlib.sha256()
    for source in sources:
        hasher.update(source.name.encode("utf-8") + b"\0")
        hasher.update(str(source.path).encode("ascii") + b"\0")
        hasher.update(source.text.encode("utf-8") + b"\0")
    return hasher.hexdigest()
