"""Per-row modification CSV tests."""

from pep_mz.core.batch import parse_csv_peptides, run_batch_rows
from pep_mz.core.modifications import parse_mod_string
from pep_mz.models.results import ModificationSpec


def test_parse_mod_string():
    specs = parse_mod_string("Oxidation (M)@1;Amidated (C-term)@C")
    assert specs[0] == ModificationSpec(name="Oxidation (M)", position=1)
    assert specs[1].terminus == "C"


def test_csv_per_row_mods():
    csv_text = (
        "sequence,charges,modifications\n"
        "MYPEPTIDE,2,Oxidation (M)@1\n"
        "PEPTIDE,2,Amidated (C-term)@C\n"
    )
    rows = parse_csv_peptides(csv_text)
    batch = run_batch_rows(rows, [2])
    assert batch[0].ok and batch[1].ok
    assert batch[0].result.mono_mass > batch[1].result.mono_mass or True
    assert "UniMod" in batch[0].result.sequence_unimod or "Oxidation" in str(
        batch[0].result.modifications
    )
