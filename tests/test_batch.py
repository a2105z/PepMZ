"""Batch and export tests."""

from pep_mz.core.batch import (
    export_csv,
    export_json,
    parse_csv_peptides,
    parse_sequence_lines,
    run_batch,
)


def test_parse_lines_and_comments():
    text = """
    # comment
    PEPTIDE
    MYPEPTIDE, SAMPLER
    """
    assert parse_sequence_lines(text) == ["PEPTIDE", "MYPEPTIDE", "SAMPLER"]


def test_batch_continues_after_bad_row():
    batch = run_batch(["PEPTIDE", "BAD1SEQ", "SAMPLER"], [2])
    assert batch[0].ok is True
    assert batch[1].ok is False
    assert batch[2].ok is True
    assert "error" in (batch[1].error or "").lower() or batch[1].error


def test_csv_parse():
    csv_text = "sequence,charges\nPEPTIDE,2;3\nMYPEPTIDE,2\n"
    rows = parse_csv_peptides(csv_text)
    assert rows[0]["sequence"] == "PEPTIDE"
    assert rows[0]["charges"] == [2, 3]


def test_exports():
    batch = run_batch(["PEPTIDE"], [1, 2])
    js = export_json(batch)
    assert "PEPTIDE" in js
    csv_out = export_csv(batch)
    assert "mz" in csv_out
    assert "400." in csv_out or "800." in csv_out
