"""Optional AI assistance — never computes scientific values."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from pep_mz.models.results import ModificationSpec


@dataclass
class ParsedIntent:
    sequence: str | None
    charges: list[int]
    modifications: list[ModificationSpec]
    notes: list[str]
    source: str
    """'rules' or 'llm'."""


def parse_intent_rules(text: str) -> ParsedIntent:
    """Deterministic NL-ish parser for common calculator phrasings."""
    notes: list[str] = []
    raw = (text or "").strip()
    if not raw:
        return ParsedIntent(None, [], [], ["Empty prompt."], "rules")

    upper = raw.upper()

    # Sequence: longest A–Z run of length >= 2 that looks like a peptide.
    seq_matches = re.findall(r"\b([ACDEFGHIKLMNPQRSTVWY]{2,})\b", upper)
    sequence = max(seq_matches, key=len) if seq_matches else None
    if sequence is None:
        notes.append("Could not find a peptide sequence (standard AA letters).")

    charges: list[int] = []
    # "charges 2 through 4" / "charges 2-4" / "charge 2,3,4" / "z=2"
    m = re.search(
        r"charges?\s+(\d+)\s*(?:through|to|-|–)\s*(\d+)", raw, flags=re.I
    )
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        lo, hi = min(a, b), max(a, b)
        charges = list(range(lo, hi + 1))
    else:
        found = re.findall(r"(?:charge[s]?|z)\s*[=:]?\s*(\d+(?:\s*[,;/]\s*\d+)*)", raw, flags=re.I)
        for block in found:
            for part in re.split(r"[,;/]", block):
                part = part.strip()
                if part.isdigit():
                    charges.append(int(part))
    if not charges:
        charges = [2]
        notes.append("No charge stated; defaulted to +2 (please confirm).")

    modifications: list[ModificationSpec] = []
    # "oxidation on M" / "oxidized methionine"
    if re.search(r"oxidation|oxidized", raw, flags=re.I):
        # Find M position if sequence known
        if sequence and "M" in sequence:
            pos = sequence.index("M") + 1
            modifications.append(ModificationSpec(name="Oxidation (M)", position=pos))
            notes.append(f"Interpreted oxidation on first M at position {pos}.")
        else:
            notes.append("Oxidation mentioned but no M found in sequence — not applied.")

    if re.search(r"phospho|phosphorylation", raw, flags=re.I):
        if sequence:
            for aa, name in [("S", "Phospho (S)"), ("T", "Phospho (T)"), ("Y", "Phospho (Y)")]:
                if aa in sequence:
                    modifications.append(
                        ModificationSpec(name=name, position=sequence.index(aa) + 1)
                    )
                    notes.append(f"Interpreted phosphorylation on first {aa}.")
                    break
            else:
                notes.append("Phosphorylation mentioned but no S/T/Y found.")

    if re.search(r"n-?term(?:inal)?\s+acetyl|acetyl(?:ation)?\s+n-?term", raw, flags=re.I):
        modifications.append(ModificationSpec(name="Acetyl", terminus="N"))
        notes.append("Interpreted N-terminal acetylation.")

    return ParsedIntent(
        sequence=sequence,
        charges=charges,
        modifications=modifications,
        notes=notes,
        source="rules",
    )


def explain_result_template(
    *,
    sequence: str,
    mono_mass: float,
    charge: int,
    mz: float,
    mod_delta: float | None = None,
) -> str:
    """Plain-English explanation from deterministic numbers (no LLM)."""
    lines = [
        f"Peptide {sequence} has a monoisotopic mass of {mono_mass:.6f} Da (neutral [M]).",
        f"At charge state +{charge}, theoretical m/z is {mz:.6f} for the protonated precursor [M+{charge}H]{charge}+.",
        "m/z falls as charge increases because the same peptide mass is divided by a larger charge.",
    ]
    if mod_delta is not None:
        lines.append(
            f"The modification changed monoisotopic mass by {mod_delta:+.6f} Da relative to the unmodified peptide; "
            "m/z shifts by approximately that delta divided by the charge."
        )
    lines.append(
        "All numeric values were computed by pyOpenMS; this text only explains those results."
    )
    return "\n".join(lines)


def llm_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def parse_intent_with_optional_llm(text: str) -> ParsedIntent:
    """Try rules first; optionally refine with OpenAI if OPENAI_API_KEY is set.

    On any LLM failure, returns the rules parse unchanged.
    """
    base = parse_intent_rules(text)
    if not llm_available():
        return base
    try:
        refined = _openai_parse(text)
        if refined.sequence:
            return refined
    except Exception as exc:  # noqa: BLE001
        base.notes.append(f"LLM refine skipped: {exc}")
    return base


def _openai_parse(text: str) -> ParsedIntent:
    # Lightweight optional dependency path — urllib to avoid requiring openai package.
    import urllib.request

    api_key = os.environ["OPENAI_API_KEY"]
    prompt = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Extract peptide calculator fields as JSON with keys: "
                    "sequence (string), charges (int array), modifications "
                    "(array of {name, position|null, terminus|null}). "
                    "Use Unimod-style names like 'Oxidation (M)'. "
                    "Do not invent masses. If unsure, use nulls."
                ),
            },
            {"role": "user", "content": text},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(prompt).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode())
    content = payload["choices"][0]["message"]["content"]
    data: dict[str, Any] = json.loads(content)
    mods = []
    for m in data.get("modifications") or []:
        mods.append(
            ModificationSpec(
                name=str(m.get("name") or ""),
                position=m.get("position"),
                terminus=m.get("terminus"),
            )
        )
    return ParsedIntent(
        sequence=(data.get("sequence") or None),
        charges=[int(x) for x in (data.get("charges") or [])],
        modifications=mods,
        notes=["Parsed with optional LLM refine; verify fields before calculating."],
        source="llm",
    )
