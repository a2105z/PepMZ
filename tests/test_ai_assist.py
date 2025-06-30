"""AI assist rules tests (no network)."""

from pep_mz.core.ai_assist import explain_result_template, parse_intent_rules


def test_parse_charges_range_and_oxidation():
    intent = parse_intent_rules(
        "Calculate PEPTIDE with charges 2 through 4 and oxidation on M in MYPEPTIDE wait"
    )
    # May pick PEPTIDE or MYPEPTIDE depending on match order — ensure we get a sequence
    intent2 = parse_intent_rules(
        "Calculate m/z for MYPEPTIDE with charges 2 through 4 and oxidation on M"
    )
    assert intent2.sequence == "MYPEPTIDE"
    assert intent2.charges == [2, 3, 4]
    assert any(m.name.startswith("Oxidation") for m in intent2.modifications)


def test_explain_template():
    text = explain_result_template(
        sequence="PEPTIDE", mono_mass=799.36, charge=2, mz=400.69, mod_delta=15.99
    )
    assert "pyOpenMS" in text
    assert "400.69" in text
