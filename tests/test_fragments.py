"""Fragment spectrum tests."""

import pyopenms as oms
import pytest

from pep_mz.core.fragments import theoretical_fragments
from pep_mz.core.calculator import calculate_peptide
from pep_mz.models.results import ModificationSpec


def test_fragments_include_b_or_y():
    peaks = theoretical_fragments("PEPTIDE", charges=[1])
    assert len(peaks) >= 5
    labels = " ".join(p.ion for p in peaks)
    assert "b" in labels.lower() or "y" in labels.lower()


def test_c_term_amidation_matches_openms():
    result = calculate_peptide(
        "PEPTIDE",
        2,
        [ModificationSpec(name="Amidated (C-term)", terminus="C")],
    )
    ref = oms.AASequence.fromString("PEPTIDE.(Amidated)")
    assert result.mono_mass == pytest.approx(float(ref.getMonoWeight()), abs=1e-6)


def test_n_and_c_term_combined():
    result = calculate_peptide(
        "PEPTIDE",
        2,
        [
            ModificationSpec(name="Acetyl (N-term)", terminus="N"),
            ModificationSpec(name="Amidated (C-term)", terminus="C"),
        ],
    )
    ref = oms.AASequence.fromString(".(Acetyl)PEPTIDE.(Amidated)")
    assert result.mono_mass == pytest.approx(float(ref.getMonoWeight()), abs=1e-6)
