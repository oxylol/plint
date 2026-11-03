"""Exceptions of the simulator. Physics never raises for a machining problem: problems are
findings. Exceptions are for invalid inputs and broken invariants."""


class TwinError(Exception):
    """Base class."""


class InputError(TwinError):
    """A project, database or program file is invalid (with file and line where known)."""

    def __init__(self, message: str, path: str = "", line: int = 0):
        location = f"{path}:{line}: " if path and line else (f"{path}: " if path else "")
        super().__init__(location + message)
        self.path = path
        self.line = line


class SourceError(InputError):
    """A database value has no resolvable source."""


class ConservationError(TwinError):
    """Mass, energy or time balance violated beyond tolerance: the run fails."""

    def __init__(self, quantity: str, expected: float, actual: float, tolerance: float, time_s: float):
        super().__init__(f"{quantity} balance violated at t={time_s:.6f} s: expected {expected:.9g}, "
                         f"got {actual:.9g} (relative tolerance {tolerance:g})")
        self.quantity = quantity
        self.expected = expected
        self.actual = actual
        self.tolerance = tolerance
        self.time_s = time_s


class CheckpointError(TwinError):
    """A checkpoint cannot be read or does not match the project."""
