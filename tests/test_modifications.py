"""Modification resolution tests."""

import pytest

from pep_mz.core.modifications import list_common_modifications, resolve_modification
from pep_mz.core.validation import ValidationError
from pep_mz.models.results import ModificationSpec


def test_resolve_oxidation_m():
    r = resolve_modification("MYPEPTIDE", ModificationSpec("Oxidation (M)", position=1))
    assert r.origin == "M"
    assert r.diff_mono_mass == pytest.approx(15.994915, abs=1e-6)


def test_bare_oxidation_with_m_context():
    r = resolve_modification("AMK", ModificationSpec("Oxidation", position=2))
    assert r.origin == "M"


def test_unknown_mod_fails():
    with pytest.raises(ValidationError, match="Could not resolve"):
        resolve_modification("PEPTIDE", ModificationSpec("NotARealModXYZ", position=1))


def test_common_list_nonempty():
    common = list_common_modifications()
    assert any(c["full_id"] == "Oxidation (M)" for c in common)
