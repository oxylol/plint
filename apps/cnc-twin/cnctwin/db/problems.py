"""Problems found while reading database and project files, and the estimated-value list."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..core.errors import InputError, SourceError

# Problem kinds (what failed):
#   yaml       the file is not valid YAML
#   format     format/version/id header wrong or unknown, file in the wrong folder
#   unknown    a key the schema does not have
#   required   a required key is missing
#   type       a value has the wrong type (text where a number is needed ...)
#   range      a number is outside its physically sane range, or a list has the wrong length
#   rule       a cross-field rule (min <= max, type-specific keys, unit vector ...)
#   source     a src that does not resolve, a number without a source, est without a note
#   reference  an id that names nothing (tool material, controller, holder ...)
#   duplicate  the same id twice, a bibliography key with conflicting entries
#   extends    an extends chain that is broken or circular
#   path       a project file path that does not exist
#   migration  a format version or a key form that cannot be migrated mechanically
#   calibration  a project's calibration overlay that cannot be read or applied, or an entry not applied
KINDS = ("yaml", "format", "unknown", "required", "type", "range", "rule", "source", "reference",
         "duplicate", "extends", "path", "migration", "calibration")


@dataclass(frozen=True)
class Problem:
    file: str
    line: int
    path: str            # dotted key path: "axes.X.travel_mm", "items[2].geometry.diameter_mm"
    message: str
    kind: str = "rule"
    severity: str = "error"   # error | warning
    obj: str = ""        # "<kind>:<id>" of the object the problem belongs to, if any

    def __str__(self) -> str:
        where = f"{self.file}:{self.line}" if self.line else self.file
        key = f" {self.path}:" if self.path else ""
        level = "" if self.severity == "error" else f" [{self.severity}]"
        return f"{where}:{key}{level} {self.message}"

    @property
    def is_error(self) -> bool:
        return self.severity == "error"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Estimate:
    """One number whose source is ``est`` (listed in the report)."""

    file: str
    line: int
    path: str
    value: object        # as written in the file (unit in the key name)
    note: str
    obj: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Migration:
    """One older key form moved forward at load time (the file itself is not changed)."""

    file: str
    line: int
    path_from: str       # key path as written ("spindles[0].thermal.growth_um_at_max_rpm")
    path_to: str         # key path after the migration ("spindles[0].thermal.growth_at_max_speed_um")
    rule: str            # id of the rule in cnctwin.db.migrate (docs/formats.md, migration table)
    note: str = ""
    obj: str = ""

    def __str__(self) -> str:
        where = f"{self.file}:{self.line}" if self.line else self.file
        return f"{where}: {self.path_from} -> {self.path_to} ({self.rule})"

    def to_dict(self) -> dict:
        return asdict(self)


def format_path(path) -> str:
    """("items", 2, "geometry", "diameter_mm") -> "items[2].geometry.diameter_mm"."""
    out = ""
    for part in path:
        if isinstance(part, int) and not isinstance(part, bool):
            out += f"[{part}]"
        else:
            out += ("." if out else "") + str(part)
    return out


def sort_problems(problems) -> list[Problem]:
    return sorted(problems, key=lambda p: (p.file, p.line, p.path, p.message))


def errors_only(problems) -> list[Problem]:
    return [p for p in problems if p.is_error]


def raise_for(problems, what: str = "") -> None:
    """Raise one exception that lists every error (``SourceError`` if all are source errors)."""
    errors = errors_only(problems)
    if not errors:
        return
    errors = sort_problems(errors)
    first = errors[0]
    if len(errors) == 1:
        message = f"{first.path}: {first.message}" if first.path else first.message
    else:
        head = f"{what}: " if what else ""
        message = f"{head}{len(errors)} problems\n" + "\n".join(str(p) for p in errors)
    cls = SourceError if all(p.kind == "source" for p in errors) else InputError
    error = cls(message, first.file, first.line)
    error.problems = errors
    error.key_path = first.path
    raise error
