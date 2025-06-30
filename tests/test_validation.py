"""Unit tests for validation."""

import pytest

from pep_mz.core.validation import ValidationError, validate_charges, validate_sequence


def test_valid_sequence():
    v = validate_sequence(" peptide ")
    assert v.sequence == "PEPTIDE"


def test_empty_rejected():
    with pytest.raises(ValidationError, match="empty"):
        validate_sequence("   ")


def test_ambiguous_rejected():
    with pytest.raises(ValidationError, match="ambiguous"):
        validate_sequence("PEPTIDEZ")


def test_digit_rejected():
    with pytest.raises(ValidationError, match="invalid characters"):
        validate_sequence("PEPT1DE")


def test_rare_u_requires_flag():
    with pytest.raises(ValidationError, match="Selenocysteine"):
        validate_sequence("ACDEFGHIKLMNPQRSTUVWY")
    v = validate_sequence("ACDU", allow_rare=True)
    assert "U" in v.sequence


def test_charges():
    assert validate_charges([2, 3, 2]) == [2, 3]
    with pytest.raises(ValidationError):
        validate_charges([0])
    with pytest.raises(ValidationError):
        validate_charges([])
