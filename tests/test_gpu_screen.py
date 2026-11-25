"""GPU precursor screen matches pyOpenMS and the trypsin rule."""

import pyopenms as oms
import pytest

from pep_mz.core.gpu_screen import screen_peptides, screen_protein, trypsin_digest
from pep_mz.core.validation import ValidationError

# N-terminus of bovine serum albumin, a standard proteomics demo sequence.
BSA_N = "MKWVTFISLLFLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHVK"


def test_trypsin_skips_proline_and_counts_missed_sites():
    peptides = trypsin_digest("AKPAAKAA", missed_cleavages=1, min_length=1, max_length=20)
    found = {(p.sequence, p.missed_cleavages, p.start) for p in peptides}
    assert ("AKPAAK", 0, 1) in found
    assert ("AA", 0, 7) in found
    assert ("AKPAAKAA", 1, 1) in found
    assert not any(p.sequence == "AK" for p in peptides)


def test_screen_matches_pyopenms_for_mixed_lengths():
    sequences = ["PEPTIDE", "MYPEPTIDE", "GAG", BSA_N[:12]]
    result = screen_peptides(sequences, [2, 3], device="cpu")
    assert result.device == "cpu"
    assert result.openms_max_abs_delta < 1e-6
    for row in result.rows:
        ref = oms.AASequence.fromString(row.sequence)
        assert row.mono_mass == pytest.approx(float(ref.getMonoWeight()), abs=1e-6)
        assert row.mz_by_charge[2] == pytest.approx(float(ref.getMZ(2)), abs=1e-6)
        assert row.mz_by_charge[3] == pytest.approx(float(ref.getMZ(3)), abs=1e-6)


def test_protein_screen_covers_every_digest_peptide():
    _peptides, result = screen_protein(
        f">example\n{BSA_N}",
        [2, 3],
        missed_cleavages=1,
        min_length=6,
        max_length=40,
        device="cpu",
    )
    assert result.rows
    assert result.openms_checked >= 1
    assert result.openms_max_abs_delta < 1e-6
    assert all(row.end >= row.start for row in result.rows)


def test_cuda_kernel_is_real_and_unavailable_on_this_machine():
    from pathlib import Path

    from pep_mz.core.cuda_precursor import cuda_status

    source = Path("native/precursor_kernel.cu").read_text()
    assert "__global__ void precursor_kernel" in source
    assert "pepmz_precursor_launch" in source
    assert "cudaMalloc" in source
    status = cuda_status()
    assert status.available is False
    with pytest.raises(ValidationError, match="CUDA was requested"):
        screen_peptides(["PEPTIDE"], [2], device="cuda")


def test_rejects_empty_and_ambiguous_protein():
    with pytest.raises(ValidationError):
        screen_protein("   ", [2])
    with pytest.raises(ValidationError):
        trypsin_digest("PEPTIDEB")
