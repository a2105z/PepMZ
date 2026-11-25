"""PepMZ Streamlit application."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from pep_mz import __version__
from pep_mz.core.ai_assist import (
    explain_result_template,
    llm_available,
    parse_intent_with_optional_llm,
)
from pep_mz.core.batch import (
    export_csv,
    export_json,
    parse_csv_peptides,
    parse_sequence_lines,
    peptide_result_to_export_rows,
    run_batch,
    run_batch_rows,
)
from pep_mz.core.calculator import calculate_peptide, compare_modified_vs_unmodified
from pep_mz.core.fragments import theoretical_fragments
from pep_mz.core.gpu_screen import screen_protein
from pep_mz.core.isotopes import theoretical_isotope_peaks
from pep_mz.core.modifications import list_common_modifications
from pep_mz.core.validation import ValidationError
from pep_mz.models.results import BatchRowResult, ModificationSpec


st.set_page_config(
    page_title="PepMZ — Peptide m/z Calculator",
    page_icon="⚗️",
    layout="wide",
)


def _sidebar():
    st.sidebar.title("PepMZ")
    st.sidebar.caption(f"v{__version__} · powered by pyOpenMS")
    st.sidebar.markdown(
        "Theoretical peptide **m/z** for proteomics. "
        "Scientific values come from **pyOpenMS / OpenMS**, never from an LLM."
    )
    mode = st.sidebar.radio(
        "Workflow",
        ["Single peptide", "Batch", "GPU screen", "Tutorial", "Optional AI assist"],
        index=0,
    )
    with st.sidebar.expander("What is m/z?", expanded=False):
        st.write(
            "**m/z** (mass-to-charge) is the quantity measured in mass spectrometry. "
            "For a protonated peptide precursor [M+zH]z+, theoretical m/z is the "
            "charged precursor mass divided by charge z."
        )
    with st.sidebar.expander("Assumptions", expanded=False):
        st.markdown(
            """
- Standard amino acids only by default (A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y)
- Ambiguous codes B/J/Z/X rejected unless explicitly enabled
- Modifications via OpenMS ModificationsDB (Unimod-backed)
- `getMZ(z)` = theoretical [M+zH]z+ m/z from pyOpenMS
- Fragment ions from OpenMS `TheoreticalSpectrumGenerator` (theoretical only)
- PepMZ is **not** a peptide identification / database search engine
- Not an official OpenMS product
"""
        )
    return mode


def _charge_selector(default: list[int] | None = None) -> list[int]:
    default = default or [2, 3]
    return st.multiselect("Charge states", list(range(1, 9)), default=default)


def _mod_builder(sequence: str, key_prefix: str = "mod") -> list[ModificationSpec]:
    st.markdown("##### Modifications (optional)")
    common = list_common_modifications()
    labels = [c["full_id"] for c in common] or [
        "Oxidation (M)",
        "Carbamidomethyl (C)",
        "Amidated (C-term)",
        "Acetyl (N-term)",
    ]
    n = st.number_input(
        "Number of modifications",
        min_value=0,
        max_value=10,
        value=0,
        key=f"{key_prefix}_n",
    )
    specs: list[ModificationSpec] = []
    for i in range(int(n)):
        cols = st.columns([3, 1, 1])
        with cols[0]:
            name = st.selectbox(
                f"Mod {i+1}",
                options=labels,
                key=f"{key_prefix}_name_{i}",
            )
        with cols[1]:
            kind = st.selectbox(
                f"Site {i+1}",
                ["Residue", "N-term", "C-term"],
                key=f"{key_prefix}_kind_{i}",
            )
        with cols[2]:
            if kind == "Residue":
                pos = st.number_input(
                    f"Pos {i+1}",
                    min_value=1,
                    max_value=max(len(sequence), 1),
                    value=1,
                    key=f"{key_prefix}_pos_{i}",
                )
                specs.append(ModificationSpec(name=name, position=int(pos)))
            elif kind == "N-term":
                nterm_name = name if "N-term" in name else "Acetyl (N-term)"
                specs.append(ModificationSpec(name=nterm_name, terminus="N"))
            else:
                cterm_name = name if "C-term" in name else "Amidated (C-term)"
                specs.append(ModificationSpec(name=cterm_name, terminus="C"))
    return specs


def _show_result(result, *, compare=None, mods=None):
    st.success("Calculated with pyOpenMS")
    if result.warnings:
        for w in result.warnings:
            st.warning(w)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mono mass [M]", f"{result.mono_mass:.6f} Da")
    c2.metric("Average mass", f"{result.average_mass:.6f} Da")
    c3.metric("Formula", result.formula)
    c4.metric("Unimod string", result.sequence_unimod)

    if result.modifications:
        st.write("**Modifications**")
        for m in result.modifications:
            st.write(f"- {m}")

    rows = peptide_result_to_export_rows(result)
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.download_button(
        "Download CSV",
        data=export_csv([BatchRowResult(0, result.sequence, True, result)]),
        file_name="pepmz_result.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download JSON",
        data=json.dumps(result.to_dict(), indent=2),
        file_name="pepmz_result.json",
        mime="application/json",
    )

    if compare:
        st.markdown("##### Unmodified vs modified")
        dmono = compare["deltas"]["delta_mono_mass"]
        st.write(f"Δ mono mass: **{dmono:+.6f} Da**")
        delta_df = pd.DataFrame(
            [
                {"charge": int(k), "delta_mz": v}
                for k, v in compare["deltas"]["delta_mz_by_charge"].items()
            ]
        )
        st.dataframe(delta_df, use_container_width=True)

    with st.expander("Theoretical isotope envelope (coarse)"):
        z = st.selectbox(
            "Isotope charge",
            [c.charge for c in result.charges],
            key="iso_z",
        )
        try:
            peaks = theoretical_isotope_peaks(result.sequence_unmodified, charge=int(z))
            pdf = pd.DataFrame(
                [{"mz": p.mz, "relative_intensity": p.intensity} for p in peaks]
            )
            st.caption(
                "Coarse theoretical isotope pattern from the peptide empirical formula "
                "via pyOpenMS. Not experimental spectrum matching."
            )
            st.dataframe(pdf, use_container_width=True)
            st.bar_chart(pdf.set_index("mz")["relative_intensity"])
        except Exception as exc:  # noqa: BLE001
            st.warning(str(exc))

    with st.expander("Theoretical fragment ions (b/y)", expanded=False):
        st.caption(
            "Generated by OpenMS TheoreticalSpectrumGenerator. "
            "This is a theoretical helper, not peptide identification against measured spectra."
        )
        frag_z = st.multiselect(
            "Fragment charges",
            list(range(1, 4)),
            default=[1],
            key="frag_z",
        )
        include_a = st.checkbox("Include a-ions", value=False, key="frag_a")
        if frag_z:
            try:
                frags = theoretical_fragments(
                    result.sequence_unmodified,
                    charges=frag_z,
                    modifications=mods,
                    include_a_ions=include_a,
                )
                fdf = pd.DataFrame(
                    [{"ion": f.ion, "mz": f.mz, "intensity": f.intensity} for f in frags]
                )
                st.dataframe(fdf, use_container_width=True)
                st.bar_chart(fdf.set_index("mz")["intensity"])
                st.download_button(
                    "Download fragments CSV",
                    data=fdf.to_csv(index=False),
                    file_name="pepmz_fragments.csv",
                    mime="text/csv",
                )
            except Exception as exc:  # noqa: BLE001
                st.warning(str(exc))

    with st.expander("Plain-English explanation"):
        c0 = result.charges[0]
        mod_delta = compare["deltas"]["delta_mono_mass"] if compare else None
        st.write(
            explain_result_template(
                sequence=result.sequence_unimod,
                mono_mass=result.mono_mass,
                charge=c0.charge,
                mz=c0.mz,
                mod_delta=mod_delta,
            )
        )


def page_single():
    st.header("Single peptide")
    st.write(
        "Enter a peptide sequence, choose charge states, optionally add Unimod modifications "
        "(residue, N-term, or C-term)."
    )

    sequence = st.text_input("Peptide sequence", value="PEPTIDE")
    charges = _charge_selector([2, 3])
    allow_rare = st.checkbox("Allow rare residues U/O", value=False)
    allow_amb = st.checkbox(
        "Allow ambiguous residues B/J/Z/X (may be unreliable)",
        value=False,
    )
    compare = st.checkbox("Compare unmodified vs modified", value=True)
    mods = _mod_builder(sequence.strip().upper() or "X", key_prefix="single")

    if st.button("Calculate", type="primary"):
        try:
            if not charges:
                st.error("Select at least one charge state.")
                return
            kwargs = dict(
                allow_rare_residues=allow_rare,
                allow_ambiguous_residues=allow_amb,
            )
            if mods and compare:
                cmp = compare_modified_vs_unmodified(
                    sequence, charges, mods, **kwargs
                )
                _show_result(cmp["modified"], compare=cmp, mods=mods)
            else:
                result = calculate_peptide(
                    sequence, charges, mods or None, **kwargs
                )
                _show_result(result, mods=mods or None)
        except ValidationError as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Unexpected error: {exc}")


def page_batch():
    st.header("Batch processing")
    st.write(
        "Pipeline: parse → validate → resolve mods → pyOpenMS → export. "
        "CSV can include per-row `modifications` like `Oxidation (M)@1;Amidated (C-term)@C`."
    )
    tab1, tab2 = st.tabs(["Text / list", "CSV"])
    charges = _charge_selector([2])
    allow_rare = st.checkbox("Allow rare residues U/O", value=False, key="batch_rare")
    allow_amb = st.checkbox(
        "Allow ambiguous residues", value=False, key="batch_amb"
    )
    shared_mods = _mod_builder("MYPEPTIDE", key_prefix="batch")

    with tab1:
        text = st.text_area(
            "Peptides (one per line; # comments allowed)",
            value="PEPTIDE\nMYPEPTIDE\nSAMPLER\nDFPIAMGER\n",
            height=180,
        )
    with tab2:
        csv_text = st.text_area(
            "CSV with `sequence` (+ optional `charges`, `modifications`)",
            value=(
                "sequence,charges,modifications\n"
                "PEPTIDE,2;3,\n"
                "MYPEPTIDE,2,Oxidation (M)@1\n"
                "PEPTIDE,2,Amidated (C-term)@C\n"
                "DFPIAMGER,2,Oxidation (M)@6\n"
            ),
            height=160,
        )
        use_csv = st.checkbox("Use CSV input instead of text list", value=True)

    if st.button("Run batch", type="primary"):
        try:
            if use_csv:
                rows = parse_csv_peptides(csv_text)
                batch = run_batch_rows(
                    rows,
                    charges,
                    default_modifications=shared_mods or None,
                    allow_rare_residues=allow_rare,
                    allow_ambiguous_residues=allow_amb,
                )
            else:
                sequences = parse_sequence_lines(text)
                batch = run_batch(
                    sequences,
                    charges,
                    modifications=shared_mods or None,
                    allow_rare_residues=allow_rare,
                    allow_ambiguous_residues=allow_amb,
                )

            ok = sum(1 for b in batch if b.ok)
            st.write(f"Completed: **{ok}/{len(batch)}** rows succeeded.")
            summary = [
                {
                    "row": b.row_index,
                    "input": b.input_sequence,
                    "ok": b.ok,
                    "error": b.error,
                    "mono_mass": b.result.mono_mass if b.result else None,
                    "unimod": b.result.sequence_unimod if b.result else None,
                }
                for b in batch
            ]
            st.dataframe(pd.DataFrame(summary), use_container_width=True)
            st.download_button(
                "Download batch CSV",
                data=export_csv(batch),
                file_name="pepmz_batch.csv",
                mime="text/csv",
            )
            st.download_button(
                "Download batch JSON",
                data=export_json(batch),
                file_name="pepmz_batch.json",
                mime="application/json",
            )
        except ValidationError as exc:
            st.error(str(exc))


def page_tutorial():
    st.header("Tutorial")
    st.markdown(
        """
### 1. Single peptide m/z
1. Open **Single peptide**
2. Enter `PEPTIDE`
3. Select charges `2` and `3`
4. Click **Calculate**
5. Download CSV/JSON if needed

### 2. Methionine oxidation
1. Sequence: `MYPEPTIDE`
2. Add modification `Oxidation (M)` at position **1**
3. Enable **Compare unmodified vs modified**
4. Expect Δmono ≈ **+15.9949 Da**

### 3. C-terminal amidation
1. Sequence: `PEPTIDE`
2. Site **C-term**, mod `Amidated (C-term)`
3. Mass should drop slightly vs unmodified (amide vs carboxylic acid)

### 4. Batch with per-row mods
Use CSV:

```text
sequence,charges,modifications
PEPTIDE,2;3,
MYPEPTIDE,2,Oxidation (M)@1
PEPTIDE,2,Amidated (C-term)@C
```

### 5. Theoretical fragments
After a successful calculation, open **Theoretical fragment ions (b/y)**.
Peaks come from OpenMS — useful for teaching, not for searching real spectra.

### What PepMZ is not
- Not a database search / peptide ID engine
- Not an official OpenMS distribution
- Does not invent masses with AI
"""
    )


def page_ai():
    st.header("Optional AI assist")
    st.info(
        "AI only helps parse language or explain results. "
        "**pyOpenMS always calculates masses and m/z.**"
    )
    st.caption(
        f"OpenAI refine: {'available' if llm_available() else 'not configured — rules parser only'}"
    )
    prompt = st.text_area(
        "Describe a calculation",
        value="Calculate m/z for MYPEPTIDE with charges 2 through 4 and oxidation on M",
        height=100,
    )
    if st.button("Parse intent", type="primary"):
        intent = parse_intent_with_optional_llm(prompt)
        st.json(
            {
                "sequence": intent.sequence,
                "charges": intent.charges,
                "modifications": [
                    {"name": m.name, "position": m.position, "terminus": m.terminus}
                    for m in intent.modifications
                ],
                "notes": intent.notes,
                "source": intent.source,
            }
        )
        st.session_state["ai_intent"] = intent

    intent = st.session_state.get("ai_intent")
    if intent and intent.sequence and intent.charges:
        if st.button("Calculate with pyOpenMS using parsed fields"):
            try:
                result = calculate_peptide(
                    intent.sequence, intent.charges, intent.modifications or None
                )
                _show_result(result, mods=intent.modifications or None)
            except ValidationError as exc:
                st.error(str(exc))


def page_gpu():
    st.header("GPU precursor screen")
    st.write(
        "Trypsin digest, then one precursor m/z per peptide. On an NVIDIA GPU this "
        "launches native/precursor_kernel.cu, with one CUDA thread per peptide. "
        "On this machine the same arithmetic runs on the CPU unless that kernel "
        "has been built. A sample of rows is checked against pyOpenMS."
    )
    example = (
        ">example bovine serum albumin N-terminus\n"
        "MKWVTFISLLFLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHVK\n"
    )
    sequence = st.text_area("Protein sequence or FASTA", value=example, height=140)
    cols = st.columns(4)
    with cols[0]:
        missed = st.number_input("Missed cleavages", min_value=0, max_value=5, value=1)
    with cols[1]:
        min_len = st.number_input("Min length", min_value=1, max_value=80, value=6)
    with cols[2]:
        max_len = st.number_input("Max length", min_value=1, max_value=80, value=40)
    with cols[3]:
        charges = _charge_selector([2, 3])

    if st.button("Screen precursors", type="primary"):
        try:
            if not charges:
                st.error("Select at least one charge state.")
                return
            _peptides, result = screen_protein(
                sequence,
                charges,
                missed_cleavages=int(missed),
                min_length=int(min_len),
                max_length=int(max_len),
            )
        except ValidationError as exc:
            st.error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            st.error(f"Unexpected error: {exc}")
            return

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Peptides", len(result.rows))
        m2.metric("Device", result.device)
        m3.metric("Tensor time", f"{result.elapsed_ms:.2f} ms")
        m4.metric("Max |Δ| vs OpenMS", f"{result.openms_max_abs_delta:.2e}")
        st.caption(
            f"Checked {result.openms_checked} peptides against pyOpenMS. "
            "Masses are neutral [M]. m/z is the protonated precursor [M+zH]z+."
        )
        frame = pd.DataFrame(
            [
                {
                    "sequence": row.sequence,
                    "start": row.start,
                    "end": row.end,
                    "missed_cleavages": row.missed_cleavages,
                    "mono_mass": row.mono_mass,
                    **{f"mz_{z}+": row.mz_by_charge[z] for z in result.charges},
                }
                for row in result.rows
            ]
        )
        st.dataframe(frame, width="stretch", hide_index=True)
        st.download_button(
            "Download CSV",
            data=frame.to_csv(index=False),
            file_name="pepmz_precursors.csv",
            mime="text/csv",
        )


def main():
    mode = _sidebar()
    st.title("PepMZ")
    st.subheader("Peptide mass-to-charge calculator for proteomics")
    if mode == "Single peptide":
        page_single()
    elif mode == "Batch":
        page_batch()
    elif mode == "GPU screen":
        page_gpu()
    elif mode == "Tutorial":
        page_tutorial()
    else:
        page_ai()


if __name__ == "__main__":
    main()
