"""Isotope helper tests."""

import pytest

from pep_mz.core.isotopes import theoretical_isotope_peaks


def test_isotope_peaks_monotonic_mz():
    peaks = theoretical_isotope_peaks("PEPTIDE", charge=2, max_isotopes=5)
    assert len(peaks) == 5
    mz = [p.mz for p in peaks]
    assert mz == sorted(mz)
    assert sum(p.intensity for p in peaks) == pytest.approx(1.0, abs=0.05)
