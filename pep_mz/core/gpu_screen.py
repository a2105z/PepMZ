"""Batched theoretical precursor m/z on PyTorch.

pyOpenMS remains the reference for a single peptide. This module digests a
protein with trypsin and scores every resulting peptide in one tensor pass.
The same code runs on CUDA when ``torch.cuda.is_available()`` and on CPU
otherwise. Residue masses and the proton mass are taken from pyOpenMS, so the
screen is a faster layout of that arithmetic, not a second mass table.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pyopenms as oms
import torch

from pep_mz.core.calculator import calculate_peptide
from pep_mz.core.validation import ValidationError, validate_charges, validate_sequence

STANDARD = "ACDEFGHIKLMNPQRSTVWY"


@dataclass(frozen=True)
class DigestPeptide:
    sequence: str
    start: int
    """1-based index in the protein."""
    end: int
    """1-based inclusive index in the protein."""
    missed_cleavages: int


@dataclass(frozen=True)
class PrecursorRow:
    sequence: str
    start: int
    end: int
    missed_cleavages: int
    mono_mass: float
    mz_by_charge: dict[int, float]


@dataclass(frozen=True)
class ScreenResult:
    device: str
    elapsed_ms: float
    charges: list[int]
    rows: list[PrecursorRow]
    openms_max_abs_delta: float
    openms_checked: int


_CONSTANTS: tuple[torch.Tensor, float, float] | None = None


def resolve_device(requested: str | None = None) -> torch.device:
    """Pick CUDA when it exists. An explicit device name is used as given."""
    if requested:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValidationError("CUDA was requested, but this PyTorch build cannot see a GPU.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _mass_constants() -> tuple[torch.Tensor, float, float]:
    """Residue mono masses (index 0 is padding), terminal water, and proton mass."""
    global _CONSTANTS
    if _CONSTANTS is not None:
        return _CONSTANTS

    mono_a = float(oms.AASequence.fromString("A").getMonoWeight())
    table = torch.zeros(27, dtype=torch.float64)
    for aa in STANDARD:
        pair = float(oms.AASequence.fromString(aa + "A").getMonoWeight())
        table[ord(aa) - ord("A") + 1] = pair - mono_a
    water = mono_a - float(table[ord("A") - ord("A") + 1])

    ref = oms.AASequence.fromString("PEPTIDE")
    proton = float(ref.getMZ(1)) - float(ref.getMonoWeight())
    _CONSTANTS = (table, water, proton)
    return _CONSTANTS


def parse_protein_text(text: str) -> str:
    """Join a pasted sequence, skipping blank lines, comments, and FASTA headers."""
    if text is None or not str(text).strip():
        raise ValidationError("Protein sequence is empty.")
    parts: list[str] = []
    for line in str(text).splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or raw.startswith(">"):
            continue
        parts.append(raw)
    if not parts:
        raise ValidationError("Protein sequence is empty.")
    return validate_sequence("".join(parts)).sequence


def trypsin_digest(
    sequence: str,
    *,
    missed_cleavages: int = 1,
    min_length: int = 6,
    max_length: int = 40,
) -> list[DigestPeptide]:
    """Digest after K or R, except when the next residue is P.

    ``missed_cleavages`` is how many cleavage sites a peptide may skip.
    Positions are 1-based in the original protein.
    """
    protein = validate_sequence(sequence).sequence
    if missed_cleavages < 0 or missed_cleavages > 5:
        raise ValidationError("Missed cleavages must be between 0 and 5.")
    if min_length < 1 or max_length < min_length or max_length > 80:
        raise ValidationError("Peptide length bounds must satisfy 1 ≤ min ≤ max ≤ 80.")

    sites = [0]
    for i, aa in enumerate(protein):
        if aa in "KR" and not (i + 1 < len(protein) and protein[i + 1] == "P"):
            sites.append(i + 1)
    if sites[-1] != len(protein):
        sites.append(len(protein))

    peptides: list[DigestPeptide] = []
    for left in range(len(sites) - 1):
        for missed in range(missed_cleavages + 1):
            right = left + 1 + missed
            if right >= len(sites):
                break
            start0 = sites[left]
            end0 = sites[right]
            pep = protein[start0:end0]
            if min_length <= len(pep) <= max_length:
                peptides.append(
                    DigestPeptide(
                        sequence=pep,
                        start=start0 + 1,
                        end=end0,
                        missed_cleavages=missed,
                    )
                )
    if not peptides:
        raise ValidationError(
            "Trypsin produced no peptides inside the length window. "
            "Widen the length limits or lower the missed-cleavage setting."
        )
    return peptides


def _encode(sequences: list[str]) -> torch.Tensor:
    width = max(len(seq) for seq in sequences)
    encoded = torch.zeros(len(sequences), width, dtype=torch.long)
    for row, seq in enumerate(sequences):
        for col, aa in enumerate(seq):
            encoded[row, col] = ord(aa) - ord("A") + 1
    return encoded


def screen_peptides(
    peptides: list[DigestPeptide] | list[str],
    charges: list[int] | int,
    *,
    device: str | None = None,
    openms_checks: int = 8,
) -> ScreenResult:
    """Compute neutral mono mass and [M+zH]z+ m/z for every peptide."""
    if not peptides:
        raise ValidationError("No peptides to screen.")
    zs = validate_charges(charges)
    spans: list[DigestPeptide] = []
    for item in peptides:
        if isinstance(item, DigestPeptide):
            spans.append(item)
        else:
            seq = validate_sequence(item).sequence
            spans.append(DigestPeptide(seq, 1, len(seq), 0))

    dev = resolve_device(device)
    table, water, proton = _mass_constants()
    table = table.to(dev)
    encoded = _encode([span.sequence for span in spans]).to(dev)
    charge_t = torch.tensor(zs, dtype=torch.float64, device=dev)

    if dev.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    mono = table[encoded].sum(dim=1) + water
    mz = (mono.unsqueeze(1) + charge_t * proton) / charge_t
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000

    mono_cpu = mono.detach().cpu().tolist()
    mz_cpu = mz.detach().cpu().tolist()
    rows = [
        PrecursorRow(
            sequence=span.sequence,
            start=span.start,
            end=span.end,
            missed_cleavages=span.missed_cleavages,
            mono_mass=float(mono_cpu[i]),
            mz_by_charge={z: float(mz_cpu[i][j]) for j, z in enumerate(zs)},
        )
        for i, span in enumerate(spans)
    ]

    check_n = min(max(openms_checks, 0), len(rows))
    if check_n == 0:
        delta = 0.0
    else:
        step = max(len(rows) // check_n, 1)
        checked_idx = list(range(0, len(rows), step))[:check_n]
        delta = 0.0
        for i in checked_idx:
            ref = calculate_peptide(rows[i].sequence, zs)
            delta = max(delta, abs(ref.mono_mass - rows[i].mono_mass))
            for charge in ref.charges:
                delta = max(delta, abs(charge.mz - rows[i].mz_by_charge[charge.charge]))
        check_n = len(checked_idx)

    return ScreenResult(
        device=str(dev),
        elapsed_ms=elapsed_ms,
        charges=zs,
        rows=rows,
        openms_max_abs_delta=delta,
        openms_checked=check_n,
    )


def screen_protein(
    sequence: str,
    charges: list[int] | int = (2, 3),
    *,
    missed_cleavages: int = 1,
    min_length: int = 6,
    max_length: int = 40,
    device: str | None = None,
) -> tuple[list[DigestPeptide], ScreenResult]:
    """Trypsin-digest a protein and screen every peptide."""
    protein = parse_protein_text(sequence)
    peptides = trypsin_digest(
        protein,
        missed_cleavages=missed_cleavages,
        min_length=min_length,
        max_length=max_length,
    )
    return peptides, screen_peptides(peptides, charges, device=device)
