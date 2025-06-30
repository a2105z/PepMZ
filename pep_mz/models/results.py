"""Domain models for PepMZ calculation requests and results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModificationSpec:
    """A modification applied at a 1-based residue position (or terminal)."""

    name: str
    """OpenMS/Unimod modification name, e.g. 'Oxidation' or 'Oxidation (M)'."""

    position: int | None = None
    """1-based residue index in the unmodified sequence. None for terminals."""

    terminus: str | None = None
    """'N' or 'C' for terminal modifications; mutually exclusive with position."""

    def describe(self) -> str:
        if self.terminus:
            return f"{self.name} @ {self.terminus}-term"
        if self.position is not None:
            return f"{self.name} @ residue {self.position}"
        return self.name


@dataclass(frozen=True)
class ChargeResult:
    charge: int
    mz: float
    precursor_mass: float
    """Mass of the charged precursor [M+nH]n+ (mono), i.e. mz * charge."""


@dataclass
class PeptideResult:
    sequence: str
    sequence_unimod: str
    sequence_unmodified: str
    formula: str
    mono_mass: float
    average_mass: float
    modifications: list[str]
    charges: list[ChargeResult]
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "sequence_unimod": self.sequence_unimod,
            "sequence_unmodified": self.sequence_unmodified,
            "formula": self.formula,
            "mono_mass": self.mono_mass,
            "average_mass": self.average_mass,
            "modifications": list(self.modifications),
            "charges": [asdict(c) for c in self.charges],
            "assumptions": list(self.assumptions),
            "warnings": list(self.warnings),
        }


@dataclass
class BatchRowResult:
    row_index: int
    input_sequence: str
    ok: bool
    result: PeptideResult | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "input_sequence": self.input_sequence,
            "ok": self.ok,
            "error": self.error,
            "result": self.result.to_dict() if self.result else None,
        }
