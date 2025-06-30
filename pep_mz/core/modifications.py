"""OpenMS / Unimod modification resolution.

Never invents modification masses. Resolves names against ModificationsDB
in a residue-aware way so bare names like 'Oxidation' are not silently
applied to the wrong origin residue.
"""

from __future__ import annotations

from dataclasses import dataclass

import pyopenms as oms

from pep_mz.core.validation import ValidationError, validate_modification_position
from pep_mz.models.results import ModificationSpec


@dataclass(frozen=True)
class ResolvedModification:
    spec: ModificationSpec
    full_id: str
    openms_name: str
    origin: str
    diff_mono_mass: float
    unimod_accession: str


def _mdb() -> oms.ModificationsDB:
    return oms.ModificationsDB()


def list_common_modifications() -> list[dict[str, str | float]]:
    """Curated common search mods with residue-specific full IDs."""
    names = [
        "Oxidation (M)",
        "Carbamidomethyl (C)",
        "Phospho (S)",
        "Phospho (T)",
        "Phospho (Y)",
        "Acetyl (N-term)",
        "Amidated (C-term)",
        "Methyl (C-term)",
        "Deamidated (N)",
        "Deamidated (Q)",
        "Gln->pyro-Glu (N-term Q)",
        "Glu->pyro-Glu (N-term E)",
    ]
    out: list[dict[str, str | float]] = []
    mdb = _mdb()
    for full in names:
        try:
            idx = mdb.findModificationIndex(full)
            mod = mdb.getModification(idx)
            out.append(
                {
                    "full_id": str(mod.getFullId()),
                    "name": str(mod.getId()),
                    "origin": str(mod.getOrigin()),
                    "diff_mono_mass": float(mod.getDiffMonoMass()),
                    "unimod": str(mod.getUniModAccession()),
                }
            )
        except Exception:
            continue
    return out


def resolve_modification(
    sequence: str,
    spec: ModificationSpec,
) -> ResolvedModification:
    """Resolve a modification against ModificationsDB for this sequence context."""
    name = (spec.name or "").strip()
    if not name:
        raise ValidationError("Modification name is empty.")

    if spec.terminus and spec.position is not None:
        raise ValidationError(
            "Specify either a residue position or a terminus, not both."
        )
    if not spec.terminus and spec.position is None:
        raise ValidationError(
            "Modification requires a 1-based residue position or terminus ('N'/'C')."
        )

    mdb = _mdb()
    terminus = (spec.terminus or "").upper() or None

    if terminus:
        if terminus not in {"N", "C"}:
            raise ValidationError("Terminus must be 'N' or 'C'.")
        candidates = _candidate_full_ids(name, terminus=terminus)
        mod = _pick_modification(mdb, candidates, name)
        return ResolvedModification(
            spec=ModificationSpec(name=name, terminus=terminus),
            full_id=str(mod.getFullId()),
            openms_name=str(mod.getId()),
            origin=str(mod.getOrigin()),
            diff_mono_mass=float(mod.getDiffMonoMass()),
            unimod_accession=str(mod.getUniModAccession()),
        )

    assert spec.position is not None
    validate_modification_position(sequence, spec.position)
    aa = sequence[spec.position - 1]
    candidates = _candidate_full_ids(name, residue=aa)
    mod = _pick_modification(mdb, candidates, name)
    origin = str(mod.getOrigin())
    if origin and origin not in {"*", "X", ""} and origin != aa:
        raise ValidationError(
            f"Modification '{mod.getFullId()}' originates on '{origin}', "
            f"but position {spec.position} is '{aa}'. "
            "Choose a compatible site or a different modification."
        )
    return ResolvedModification(
        spec=ModificationSpec(name=name, position=spec.position),
        full_id=str(mod.getFullId()),
        openms_name=str(mod.getId()),
        origin=origin,
        diff_mono_mass=float(mod.getDiffMonoMass()),
        unimod_accession=str(mod.getUniModAccession()),
    )


def _candidate_full_ids(
    name: str,
    *,
    residue: str | None = None,
    terminus: str | None = None,
) -> list[str]:
    name = name.strip()
    out: list[str] = []
    if "(" in name and name.endswith(")"):
        out.append(name)
    if residue:
        out.append(f"{name} ({residue})")
    if terminus == "N":
        out.extend(
            [
                f"{name} (N-term)",
                f"{name} (Protein N-term)",
                name if "N-term" in name else f"{name} (N-term)",
            ]
        )
    if terminus == "C":
        out.extend(
            [
                f"{name} (C-term)",
                f"{name} (Protein C-term)",
                name if "C-term" in name else f"{name} (C-term)",
            ]
        )
    out.append(name)
    seen: set[str] = set()
    uniq: list[str] = []
    for c in out:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def _pick_modification(mdb: oms.ModificationsDB, candidates: list[str], original: str):
    for cand in candidates:
        try:
            idx = mdb.findModificationIndex(cand)
            return mdb.getModification(idx)
        except Exception:
            continue
    raise ValidationError(
        f"Could not resolve modification '{original}' against OpenMS ModificationsDB "
        f"(Unimod). Tried: {', '.join(candidates)}. "
        "Use a residue-qualified name such as 'Oxidation (M)' or 'Amidated (C-term)'."
    )


def apply_modifications_to_sequence(
    sequence: str,
    specs: list[ModificationSpec],
) -> tuple[oms.AASequence, list[ResolvedModification]]:
    """Build an AASequence with residue and terminal modifications via OpenMS."""
    resolved = [resolve_modification(sequence, s) for s in specs]

    n_term = [r for r in resolved if r.spec.terminus == "N"]
    c_term = [r for r in resolved if r.spec.terminus == "C"]
    residue_mods = [r for r in resolved if r.spec.position is not None]

    if len(n_term) > 1:
        raise ValidationError("Only one N-terminal modification is supported.")
    if len(c_term) > 1:
        raise ValidationError("Only one C-terminal modification is supported.")

    # Detect two residue mods on the same site.
    positions = [r.spec.position for r in residue_mods]
    if len(positions) != len(set(positions)):
        raise ValidationError(
            "Multiple modifications on the same residue position are not supported."
        )

    pieces: list[str] = []
    by_pos = {r.spec.position: r for r in residue_mods}
    for i, aa in enumerate(sequence, start=1):
        if i in by_pos:
            pieces.append(f"{aa}({by_pos[i].openms_name})")
        else:
            pieces.append(aa)
    core = "".join(pieces)

    if n_term:
        core = f".({n_term[0].openms_name}){core}"
    if c_term:
        core = f"{core}.({c_term[0].openms_name})"

    try:
        aa_seq = oms.AASequence.fromString(core)
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(
            f"OpenMS could not build modified sequence from '{core}': {exc}"
        ) from exc

    return aa_seq, resolved


def parse_mod_string(raw: str) -> list[ModificationSpec]:
    """Parse batch mod strings like ``Oxidation (M)@1;Amidated (C-term)@C;Acetyl@N``."""
    if raw is None or not str(raw).strip():
        return []
    specs: list[ModificationSpec] = []
    for part in str(raw).split(";"):
        token = part.strip()
        if not token:
            continue
        if "@" not in token:
            raise ValidationError(
                f"Modification '{token}' must look like 'Name@position' or 'Name@N' / 'Name@C'."
            )
        name, site = token.rsplit("@", 1)
        name = name.strip()
        site = site.strip().upper()
        if site in {"N", "C"}:
            specs.append(ModificationSpec(name=name, terminus=site))
        else:
            if not site.isdigit():
                raise ValidationError(
                    f"Modification site '{site}' must be a 1-based position or N/C."
                )
            specs.append(ModificationSpec(name=name, position=int(site)))
    return specs
