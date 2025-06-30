"""Theoretical fragment ions via OpenMS TheoreticalSpectrumGenerator."""

from __future__ import annotations

from dataclasses import dataclass

import pyopenms as oms

from pep_mz.core.modifications import apply_modifications_to_sequence
from pep_mz.core.validation import ValidationError, validate_charges, validate_sequence
from pep_mz.models.results import ModificationSpec


@dataclass(frozen=True)
class FragmentPeak:
    mz: float
    intensity: float
    ion: str


def theoretical_fragments(
    sequence: str,
    *,
    charges: list[int] | int = 1,
    modifications: list[ModificationSpec] | None = None,
    allow_rare_residues: bool = False,
    allow_ambiguous_residues: bool = False,
    include_a_ions: bool = False,
) -> list[FragmentPeak]:
    """Generate theoretical b/y (optional a) fragment peaks with OpenMS.

    This is a theoretical spectrum helper — not experimental spectrum matching.
    """
    validated = validate_sequence(
        sequence,
        allow_rare=allow_rare_residues,
        allow_ambiguous=allow_ambiguous_residues,
    )
    zs = validate_charges(charges)
    mods = modifications or []

    if mods:
        aa_seq, _ = apply_modifications_to_sequence(validated.sequence, mods)
    else:
        aa_seq = oms.AASequence.fromString(validated.sequence)

    tsg = oms.TheoreticalSpectrumGenerator()
    params = tsg.getParameters()
    params.setValue("add_metainfo", "true")
    params.setValue("add_b_ions", "true")
    params.setValue("add_y_ions", "true")
    params.setValue("add_a_ions", "true" if include_a_ions else "false")
    params.setValue("add_first_prefix_ion", "true")
    tsg.setParameters(params)

    spec = oms.MSSpectrum()
    try:
        tsg.getSpectrum(spec, aa_seq, min(zs), max(zs))
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(f"OpenMS theoretical spectrum generation failed: {exc}") from exc

    labels: list[str] = [""] * spec.size()
    arrays = spec.getStringDataArrays()
    if arrays:
        arr = arrays[0]
        for i in range(min(arr.size(), spec.size())):
            lab = arr[i]
            if isinstance(lab, (bytes, bytearray)):
                lab = lab.decode("utf-8", errors="replace")
            labels[i] = str(lab).strip()

    peaks: list[FragmentPeak] = []
    for i in range(spec.size()):
        peaks.append(
            FragmentPeak(
                mz=float(spec[i].getMZ()),
                intensity=float(spec[i].getIntensity()),
                ion=labels[i] or f"peak_{i+1}",
            )
        )
    peaks.sort(key=lambda p: p.mz)
    if not peaks:
        raise ValidationError("OpenMS returned no theoretical fragment peaks.")
    return peaks
