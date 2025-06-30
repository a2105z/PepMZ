"""Optional theoretical isotope envelope via pyOpenMS (not a search engine)."""

from __future__ import annotations

from dataclasses import dataclass

import pyopenms as oms

from pep_mz.core.validation import ValidationError, validate_sequence


@dataclass(frozen=True)
class IsotopePeak:
    mz: float
    intensity: float
    """Relative intensity from OpenMS coarse isotope pattern (sums ~1)."""


def theoretical_isotope_peaks(
    sequence: str,
    *,
    charge: int = 1,
    max_isotopes: int = 8,
    allow_rare_residues: bool = False,
) -> list[IsotopePeak]:
    """Return coarse theoretical isotope peaks for the peptide formula.

    Intensities are relative pattern weights from OpenMS.
    For charge > 1, m/z values are shifted to the protonated precursor series
    using AASequence.getMZ semantics via formula mass peaks / charge with proton
    adjustment through getMonoWeight path:

    We compute the neutral isotope distribution on the empirical formula, then
    convert each neutral mass peak to approximate [M+zH]z+ m/z by:
        mz = (neutral_isotope_mass + z * proton_mass_delta) / z
    where proton alignment uses OpenMS precursor mono mass difference.
    """
    if charge < 1:
        raise ValidationError("Charge must be ≥ 1 for isotope display.")
    if max_isotopes < 1 or max_isotopes > 30:
        raise ValidationError("max_isotopes must be between 1 and 30.")

    validated = validate_sequence(sequence, allow_rare=allow_rare_residues)
    aa = oms.AASequence.fromString(validated.sequence)
    formula = aa.getFormula()
    gen = oms.CoarseIsotopePatternGenerator(max_isotopes)
    dist = formula.getIsotopeDistribution(gen)

    mono = float(aa.getMonoWeight())
    precursor = float(aa.getMonoWeight(oms.Residue.ResidueType.Full, charge))
    # Mass added by charge carriers according to OpenMS.
    charge_offset = precursor - mono

    peaks: list[IsotopePeak] = []
    container = dist.getContainer()
    for i in range(dist.size()):
        p = container[i]
        neutral_mass = float(p.getMZ())
        intensity = float(p.getIntensity())
        mz = (neutral_mass + charge_offset) / charge
        peaks.append(IsotopePeak(mz=mz, intensity=intensity))
    if not peaks:
        raise ValidationError("OpenMS returned an empty isotope distribution.")
    return peaks
