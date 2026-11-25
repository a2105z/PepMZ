"""The retention model ranks hydrophobicity and does not replace OpenMS masses."""

import pytest

from pep_mz.core.validation import ValidationError
from pep_mz.ml.retention import predict_retention


def test_hydrophobic_peptide_ranks_above_charged_peptide():
    oily = predict_retention("LLLLLL")
    mixed = predict_retention("PEPTIDE")
    charged = predict_retention("KKKKKK")
    assert oily.index > mixed.index > charged.index


def test_prediction_stays_near_kyte_doolittle():
    result = predict_retention("PEPTIDE")
    assert result.index == pytest.approx(result.reference_index, abs=0.1)


def test_rejects_ambiguous_sequence():
    with pytest.raises(ValidationError):
        predict_retention("PEPTIDEB")
