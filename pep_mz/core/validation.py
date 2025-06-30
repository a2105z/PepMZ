"""Input validation for peptide sequences, charges, and modification sites."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Canonical proteogenic amino acids used for PepMZ calculations.
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Ambiguous / special residues OpenMS may accept but PepMZ rejects by default.
AMBIGUOUS_AA = {
    "B": "Asparagine/Aspartate (ambiguous)",
    "J": "Isoleucine/Leucine (ambiguous)",
    "Z": "Glutamine/Glutamate (ambiguous)",
    "X": "Unspecified/Unknown residue",
}

# Rare but chemically defined; allowed only if explicitly enabled.
RARE_AA = {
    "U": "Selenocysteine",
    "O": "Pyrrolysine",
}


class ValidationError(ValueError):
    """Raised when input is scientifically ambiguous or invalid."""


@dataclass(frozen=True)
class ValidatedSequence:
    sequence: str
    """Uppercase unmodified letter sequence."""


def normalize_sequence(raw: str) -> str:
    if raw is None:
        raise ValidationError("Peptide sequence is empty.")
    seq = raw.strip().upper().replace(" ", "")
    # Strip common terminal dots used in OpenMS notation: .PEPTIDE.
    if seq.startswith("."):
        seq = seq[1:]
    if seq.endswith("."):
        seq = seq[:-1]
    return seq


def validate_sequence(
    raw: str,
    *,
    allow_rare: bool = False,
    allow_ambiguous: bool = False,
) -> ValidatedSequence:
    """Validate an unmodified peptide letter sequence.

    Rejects empty input, digits, punctuation (except we already strip dots),
    ambiguous residues (B/J/Z/X) unless allow_ambiguous=True, and rare residues
    unless allow_rare=True.
    """
    seq = normalize_sequence(raw)
    if not seq:
        raise ValidationError("Peptide sequence is empty.")

    if not re.fullmatch(r"[A-Z]+", seq):
        bad = sorted({ch for ch in seq if not ch.isalpha()})
        raise ValidationError(
            f"Sequence contains invalid characters {bad}. "
            "Use standard one-letter amino-acid codes only."
        )

    allowed = set(STANDARD_AA)
    if allow_rare:
        allowed |= set(RARE_AA)
    if allow_ambiguous:
        allowed |= set(AMBIGUOUS_AA)

    for i, aa in enumerate(seq, start=1):
        if aa in AMBIGUOUS_AA and not allow_ambiguous:
            raise ValidationError(
                f"Residue '{aa}' at position {i} is ambiguous ({AMBIGUOUS_AA[aa]}). "
                "PepMZ does not silently expand ambiguous codes — replace with a "
                "specific amino acid, or enable ambiguous residues explicitly "
                "(results may be scientifically unreliable)."
            )
        if aa in RARE_AA and not allow_rare:
            raise ValidationError(
                f"Residue '{aa}' at position {i} is {RARE_AA[aa]}. "
                "Enable rare residues explicitly if you intend to use U/O."
            )
        if aa not in allowed:
            raise ValidationError(
                f"Unsupported residue '{aa}' at position {i}."
            )

    return ValidatedSequence(sequence=seq)


def validate_charges(charges: list[int] | tuple[int, ...] | int) -> list[int]:
    if isinstance(charges, int):
        charges = [charges]
    if not charges:
        raise ValidationError("At least one charge state is required.")
    out: list[int] = []
    seen: set[int] = set()
    for z in charges:
        if not isinstance(z, int) or isinstance(z, bool):
            raise ValidationError(f"Charge state must be an integer, got {z!r}.")
        if z < 1:
            raise ValidationError(
                f"Charge state {z} is invalid. Charge must be a positive integer (≥ 1)."
            )
        if z > 20:
            raise ValidationError(
                f"Charge state {z} is outside the supported range (1–20)."
            )
        if z not in seen:
            out.append(z)
            seen.add(z)
    return out


def validate_modification_position(sequence: str, position: int) -> None:
    if position < 1 or position > len(sequence):
        raise ValidationError(
            f"Modification position {position} is out of range for sequence "
            f"length {len(sequence)} (positions are 1-based)."
        )
