"""Batch parsing, validation, calculation, and export helpers."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from pep_mz.core.calculator import calculate_peptide
from pep_mz.core.modifications import parse_mod_string
from pep_mz.core.validation import ValidationError
from pep_mz.models.results import BatchRowResult, ModificationSpec, PeptideResult


def parse_sequence_lines(text: str) -> list[str]:
    """Parse newline- or comma-separated peptide sequences (ignores blanks/# comments)."""
    if text is None:
        raise ValidationError("Batch text is empty.")
    sequences: list[str] = []
    for line in text.splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        sequences.extend(parts)
    if not sequences:
        raise ValidationError("No peptide sequences found in batch input.")
    return sequences


def parse_csv_peptides(text: str) -> list[dict[str, Any]]:
    """Parse CSV with required ``sequence`` column.

    Optional columns:
    - ``charges`` like ``2`` or ``2;3;4``
    - ``modifications`` like ``Oxidation (M)@1;Amidated (C-term)@C``
    """
    if not text or not text.strip():
        raise ValidationError("CSV input is empty.")
    reader = csv.DictReader(io.StringIO(text.strip()))
    if not reader.fieldnames:
        raise ValidationError("CSV has no header row.")
    fields = {f.strip().lower(): f for f in reader.fieldnames if f}
    if "sequence" not in fields:
        raise ValidationError(
            "CSV must include a 'sequence' column. Optional: 'charges', 'modifications'."
        )
    seq_key = fields["sequence"]
    charges_key = fields.get("charges")
    mods_key = fields.get("modifications")
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(reader, start=2):
        seq = (row.get(seq_key) or "").strip()
        if not seq:
            rows.append(
                {
                    "row_index": i,
                    "sequence": "",
                    "charges": None,
                    "modifications": [],
                    "empty": True,
                }
            )
            continue
        charges = None
        if charges_key and row.get(charges_key):
            raw_c = str(row[charges_key]).strip()
            charges = [int(x) for x in raw_c.replace(";", ",").split(",") if x.strip()]
        mods: list[ModificationSpec] = []
        if mods_key and row.get(mods_key):
            mods = parse_mod_string(str(row[mods_key]))
        rows.append(
            {
                "row_index": i,
                "sequence": seq,
                "charges": charges,
                "modifications": mods,
                "empty": False,
            }
        )
    if not rows:
        raise ValidationError("CSV contains no data rows.")
    return rows


def run_batch(
    sequences: list[str],
    charges: list[int],
    *,
    modifications: list[ModificationSpec] | None = None,
    allow_rare_residues: bool = False,
    allow_ambiguous_residues: bool = False,
) -> list[BatchRowResult]:
    """Process many peptides; per-row errors do not abort the batch."""
    results: list[BatchRowResult] = []
    for i, seq in enumerate(sequences):
        try:
            if not str(seq).strip():
                raise ValidationError("Empty sequence in batch row.")
            result = calculate_peptide(
                seq,
                charges,
                modifications,
                allow_rare_residues=allow_rare_residues,
                allow_ambiguous_residues=allow_ambiguous_residues,
            )
            results.append(
                BatchRowResult(
                    row_index=i,
                    input_sequence=seq,
                    ok=True,
                    result=result,
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                BatchRowResult(
                    row_index=i,
                    input_sequence=seq,
                    ok=False,
                    error=str(exc),
                )
            )
    return results


def run_batch_rows(
    rows: list[dict[str, Any]],
    default_charges: list[int],
    *,
    default_modifications: list[ModificationSpec] | None = None,
    allow_rare_residues: bool = False,
    allow_ambiguous_residues: bool = False,
) -> list[BatchRowResult]:
    """Run batch from parsed CSV-like row dicts with optional per-row charges/mods."""
    results: list[BatchRowResult] = []
    for row in rows:
        idx = int(row["row_index"])
        seq = row.get("sequence") or ""
        if row.get("empty") or not str(seq).strip():
            results.append(
                BatchRowResult(idx, seq, False, error="Empty sequence")
            )
            continue
        try:
            charges = row.get("charges") or default_charges
            mods = row.get("modifications")
            if mods is None or mods == []:
                mods = default_modifications
            result = calculate_peptide(
                seq,
                charges,
                mods or None,
                allow_rare_residues=allow_rare_residues,
                allow_ambiguous_residues=allow_ambiguous_residues,
            )
            results.append(BatchRowResult(idx, seq, True, result))
        except Exception as exc:  # noqa: BLE001
            results.append(BatchRowResult(idx, seq, False, error=str(exc)))
    return results


def results_to_table_rows(batch: list[BatchRowResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in batch:
        if not item.ok or item.result is None:
            rows.append(
                {
                    "row_index": item.row_index,
                    "input_sequence": item.input_sequence,
                    "ok": False,
                    "error": item.error,
                }
            )
            continue
        r = item.result
        for c in r.charges:
            rows.append(
                {
                    "row_index": item.row_index,
                    "input_sequence": item.input_sequence,
                    "ok": True,
                    "sequence_unimod": r.sequence_unimod,
                    "formula": r.formula,
                    "mono_mass": r.mono_mass,
                    "average_mass": r.average_mass,
                    "modifications": "; ".join(r.modifications),
                    "charge": c.charge,
                    "mz": c.mz,
                    "precursor_mass": c.precursor_mass,
                    "error": None,
                }
            )
    return rows


def export_json(batch: list[BatchRowResult]) -> str:
    return json.dumps([b.to_dict() for b in batch], indent=2)


def export_csv(batch: list[BatchRowResult]) -> str:
    rows = results_to_table_rows(batch)
    if not rows:
        return ""
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def write_export(path: str | Path, content: str) -> None:
    Path(path).write_text(content, encoding="utf-8")


def peptide_result_to_export_rows(result: PeptideResult) -> list[dict[str, Any]]:
    return results_to_table_rows(
        [
            BatchRowResult(
                row_index=0,
                input_sequence=result.sequence,
                ok=True,
                result=result,
            )
        ]
    )
