"""Scientific calculation engine — pyOpenMS is the source of truth."""

from __future__ import annotations

from dataclasses import dataclass

import pyopenms as oms

from pep_mz.core.modifications import apply_modifications_to_sequence
from pep_mz.core.validation import (
    AMBIGUOUS_AA,
    ValidationError,
    validate_charges,
    validate_sequence,
)
from pep_mz.models.results import ChargeResult, ModificationSpec, PeptideResult


ASSUMPTIONS = [
    "Masses and m/z values are computed by pyOpenMS (OpenMS AASequence).",
    "getMZ(z) corresponds to the protonated precursor [M+zH]z+.",
    "mono_mass is the monoisotopic mass of the neutral peptide [M].",
    "average_mass is the average mass of the neutral peptide [M].",
    "Only standard amino acids A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y are accepted by default.",
    "Modifications are resolved against OpenMS ModificationsDB (Unimod-backed), not a custom mass table.",
]


@dataclass(frozen=True)
class CalculateRequest:
    sequence: str
    charges: list[int]
    modifications: list[ModificationSpec] | None = None
    allow_rare_residues: bool = False
    compare_unmodified: bool = False


def calculate_peptide(
    sequence: str,
    charges: list[int] | int,
    modifications: list[ModificationSpec] | None = None,
    *,
    allow_rare_residues: bool = False,
    allow_ambiguous_residues: bool = False,
) -> PeptideResult:
    """Calculate theoretical masses and m/z for one peptide."""
    validated = validate_sequence(
        sequence,
        allow_rare=allow_rare_residues,
        allow_ambiguous=allow_ambiguous_residues,
    )
    zs = validate_charges(charges)
    mods = modifications or []
    warnings: list[str] = []
    if allow_ambiguous_residues and any(ch in AMBIGUOUS_AA for ch in validated.sequence):
        warnings.append(
            "Sequence contains ambiguous residue codes; masses may be undefined or unreliable in OpenMS."
        )

    if mods:
        aa_seq, resolved = apply_modifications_to_sequence(validated.sequence, mods)
        mod_labels = [
            f"{r.full_id} (Δmono={r.diff_mono_mass:.6f} Da)"
            + (
                f" @ {r.spec.position}"
                if r.spec.position is not None
                else f" @ {r.spec.terminus}-term"
            )
            for r in resolved
        ]
    else:
        try:
            aa_seq = oms.AASequence.fromString(validated.sequence)
        except Exception as exc:  # noqa: BLE001
            raise ValidationError(
                f"OpenMS could not parse sequence '{validated.sequence}': {exc}"
            ) from exc
        mod_labels = []

    try:
        mono = float(aa_seq.getMonoWeight())
        average = float(aa_seq.getAverageWeight())
        formula = str(aa_seq.getFormula())
        unimod_str = str(aa_seq.toUniModString())
        unmodified = str(aa_seq.toUnmodifiedString())
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(f"OpenMS mass calculation failed: {exc}") from exc

    if mono <= 0:
        raise ValidationError(
            "Computed monoisotopic mass is non-positive; check the sequence and modifications."
        )

    charge_results: list[ChargeResult] = []
    for z in zs:
        try:
            mz = float(aa_seq.getMZ(z))
            precursor = float(aa_seq.getMonoWeight(oms.Residue.ResidueType.Full, z))
        except Exception as exc:  # noqa: BLE001
            raise ValidationError(
                f"OpenMS could not compute m/z for charge {z}: {exc}"
            ) from exc
        charge_results.append(
            ChargeResult(charge=z, mz=mz, precursor_mass=precursor)
        )

    return PeptideResult(
        sequence=validated.sequence,
        sequence_unimod=unimod_str,
        sequence_unmodified=unmodified,
        formula=formula,
        mono_mass=mono,
        average_mass=average,
        modifications=mod_labels,
        charges=charge_results,
        assumptions=list(ASSUMPTIONS),
        warnings=warnings,
    )


def compare_modified_vs_unmodified(
    sequence: str,
    charges: list[int] | int,
    modifications: list[ModificationSpec],
    *,
    allow_rare_residues: bool = False,
    allow_ambiguous_residues: bool = False,
) -> dict:
    """Return unmodified and modified results plus mono/m/z deltas."""
    if not modifications:
        raise ValidationError("Comparison requires at least one modification.")
    base = calculate_peptide(
        sequence,
        charges,
        None,
        allow_rare_residues=allow_rare_residues,
        allow_ambiguous_residues=allow_ambiguous_residues,
    )
    mod = calculate_peptide(
        sequence,
        charges,
        modifications,
        allow_rare_residues=allow_rare_residues,
        allow_ambiguous_residues=allow_ambiguous_residues,
    )
    deltas = {
        "delta_mono_mass": mod.mono_mass - base.mono_mass,
        "delta_average_mass": mod.average_mass - base.average_mass,
        "delta_mz_by_charge": {
            str(c.charge): (
                next(x.mz for x in mod.charges if x.charge == c.charge) - c.mz
            )
            for c in base.charges
        },
    }
    return {"unmodified": base, "modified": mod, "deltas": deltas}
