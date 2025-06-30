"""Tests for pyOpenMS-backed calculations."""

import pyopenms as oms
import pytest

from pep_mz.core.calculator import calculate_peptide, compare_modified_vs_unmodified
from pep_mz.core.validation import ValidationError
from pep_mz.models.results import ModificationSpec


def test_peptide_matches_pyopenms_directly():
    result = calculate_peptide("PEPTIDE", [1, 2, 3])
    ref = oms.AASequence.fromString("PEPTIDE")
    assert result.mono_mass == pytest.approx(float(ref.getMonoWeight()), rel=0, abs=1e-9)
    assert result.average_mass == pytest.approx(float(ref.getAverageWeight()), rel=0, abs=1e-9)
    assert result.charges[0].mz == pytest.approx(float(ref.getMZ(1)), rel=0, abs=1e-9)
    assert result.charges[1].mz == pytest.approx(float(ref.getMZ(2)), rel=0, abs=1e-9)
    assert result.charges[2].mz == pytest.approx(float(ref.getMZ(3)), rel=0, abs=1e-9)
    assert result.formula == str(ref.getFormula())


def test_mz_decreases_with_charge():
    result = calculate_peptide("PEPTIDE", [1, 2, 4])
    mzs = [c.mz for c in result.charges]
    assert mzs[0] > mzs[1] > mzs[2]


def test_invalid_sequence_raises():
    with pytest.raises(ValidationError):
        calculate_peptide("PEPTIDEB", 2)


def test_oxidation_on_m_matches_openms_string():
    seq = "MYPEPTIDE"
    result = calculate_peptide(
        seq,
        2,
        [ModificationSpec(name="Oxidation (M)", position=1)],
    )
    ref = oms.AASequence.fromString("M(Oxidation)YPEPTIDE")
    assert result.mono_mass == pytest.approx(float(ref.getMonoWeight()), rel=0, abs=1e-6)
    assert result.charges[0].mz == pytest.approx(float(ref.getMZ(2)), rel=0, abs=1e-6)
    assert "Oxidation" in result.modifications[0] or "UniMod:35" in result.sequence_unimod


def test_compare_shows_positive_delta_for_oxidation():
    cmp = compare_modified_vs_unmodified(
        "MYPEPTIDE",
        [2],
        [ModificationSpec(name="Oxidation (M)", position=1)],
    )
    assert cmp["deltas"]["delta_mono_mass"] == pytest.approx(15.994915, abs=1e-4)
    assert cmp["deltas"]["delta_mz_by_charge"]["2"] > 0


def test_wrong_site_rejected():
    with pytest.raises(ValidationError):
        calculate_peptide(
            "PEPTIDE",
            2,
            [ModificationSpec(name="Oxidation (M)", position=1)],  # P, not M
        )
