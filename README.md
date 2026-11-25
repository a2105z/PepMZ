# PepMZ

**Theoretical peptide m/z for proteomics — accessible UI, pyOpenMS underneath.**

PepMZ is a scientist-facing calculator for **theoretical peptide mass-to-charge (m/z)** values. It is built for wet-lab researchers, students, and anyone who needs accurate peptide masses without writing Python against OpenMS directly.

> **Scientific source of truth:** all masses and m/z values are computed by **[pyOpenMS](https://pyopenms.readthedocs.io/)** / OpenMS (`AASequence`). PepMZ never uses an LLM (or any other heuristic) to invent scientific numbers.

PepMZ is an independent open-source tool. It is **not** an official OpenMS product and is **not** a peptide identification / database search engine.

---

## Who it is for

- Proteomics / metabolomics trainees who need quick theoretical m/z checks
- Wet-lab scientists who prefer a form over a script
- Developers learning how OpenMS peptide mass APIs behave

## The scientific problem

In bottom-up proteomics, peptides are observed as ions in a mass spectrometer. Matching measured peaks to chemistry starts from **theoretical** monoisotopic mass and **m/z** at plausible charge states, often including **post-translational modifications (PTMs)** described in resources such as Unimod (exposed in OpenMS via `ModificationsDB`).

The arithmetic is well-defined; the friction is tooling. PepMZ removes that friction without hiding assumptions.

## Why PepMZ exists

Make powerful scientific computation accessible **without compromising the science**:

- Clear validation (no silent guessing of ambiguous residues)
- OpenMS/Unimod modification resolution
- Batch processing with per-row errors
- Exportable, reproducible results
- Optional explanations that only narrate deterministic outputs

---

## Features

| Feature | Status |
| --- | --- |
| Single-peptide mono / average mass + m/z | ✅ |
| Multiple charge states | ✅ |
| Unimod/OpenMS residue, N-term, and C-term mods | ✅ |
| Unmodified vs modified comparison | ✅ |
| Batch text / CSV pipeline (per-row mods) | ✅ |
| CSV / JSON export | ✅ |
| Theoretical coarse isotope envelope (pyOpenMS) | ✅ |
| Theoretical b/y fragment ions (OpenMS TSG) | ✅ |
| In-app tutorial | ✅ |
| Optional NL intent parse + explanations | ✅ (rules; optional OpenAI refine) |
| PyTorch retention index from the peptide sequence | ✅ |
| Experimental spectrum identification | ❌ (out of scope) |

---

## Architecture

```text
Input (UI / batch / optional NL parse)
        │
        ▼
 Input validation  (alphabet, charge, sites)
        │
        ▼
 PTM resolution    (OpenMS ModificationsDB / Unimod)
        │
        ▼
 Scientific engine (pyOpenMS AASequence)
        │
        ├── masses + m/z results
        ├── optional isotope peaks
        └── CSV / JSON export
```

Package layout:

```text
PepMZ/
├── app.py                 # streamlit run app.py
├── pep_mz/
│   ├── core/
│   │   ├── calculator.py  # pyOpenMS calculations
│   │   ├── validation.py
│   │   ├── modifications.py
│   │   ├── batch.py
│   │   ├── isotopes.py
│   │   └── ai_assist.py   # optional; never computes masses
│   ├── models/
│   ├── ui/
│   └── ...
├── tests/
├── examples/
└── README.md
```

---

## Retention index

After the mass is calculated, PepMZ also reports a **retention index**. A small PyTorch network reads the unmodified sequence and predicts how hydrophobic the peptide is, on the Kyte–Doolittle scale. A higher index should stick longer to a reversed-phase column and come off later. The network does not calculate mass, charge, or m/z. Those stay with pyOpenMS.

## How peptide m/z calculation works (conceptually)

1. Parse a peptide into residues (`AASequence`).
2. Apply modifications known to OpenMS/Unimod.
3. Compute neutral monoisotopic mass `[M]` via `getMonoWeight()`.
4. For charge \(z\), compute theoretical precursor m/z for \([M+zH]^{z+}\) via `getMZ(z)`.

PepMZ also reports average mass (`getAverageWeight()`) and chemical formula (`getFormula()`).

### Modifications

Modifications are resolved against OpenMS `ModificationsDB` using residue-qualified identifiers such as `Oxidation (M)`. Bare names are only accepted when they can be resolved safely in residue context — PepMZ will **error** rather than silently attach the wrong Unimod record.

---

## Installation

Requires Python 3.9+ (3.11 recommended) and a platform with pyOpenMS wheels.

```bash
cd PepMZ
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running the application

```bash
source .venv/bin/activate
streamlit run app.py
```

Open the local URL Streamlit prints (typically http://localhost:8501).

---

## Running tests

```bash
source .venv/bin/activate
pytest -q
```

Tests compare PepMZ outputs to direct pyOpenMS calls for reference peptides and modification cases.

---

## Example workflows

**Single peptide**

1. Enter `PEPTIDE`
2. Select charges `2` and `3`
3. Calculate → inspect mono mass, formula, and m/z table
4. Export CSV/JSON

**With modification**

1. Enter `MYPEPTIDE`
2. Add modification `Oxidation (M)` at position `1`
3. Enable compare unmodified vs modified
4. Confirm Δmono ≈ +15.9949 Da

**Batch**

```bash
# examples/peptides.txt or examples/peptides.csv in the Batch tab
```

**Library use**

```python
from pep_mz.core.calculator import calculate_peptide
from pep_mz.models.results import ModificationSpec

r = calculate_peptide("PEPTIDE", [2, 3])
print(r.mono_mass, r.charges[0].mz)

r2 = calculate_peptide(
    "MYPEPTIDE",
    2,
    [ModificationSpec(name="Oxidation (M)", position=1)],
)
print(r2.sequence_unimod, r2.mono_mass)
```

---

## Optional AI layer

Under **Optional AI assist**:

- A **rules parser** maps phrases like “charges 2 through 4” and “oxidation on M” into form fields
- If `OPENAI_API_KEY` is set, an optional refine step may help parsing
- You must **verify** parsed fields; **Calculate** still calls pyOpenMS

Without an API key, the entire scientific application remains fully usable.

---

## Limitations

- Not a search engine / ID pipeline (no PSM scoring against experimental peak lists)
- Ambiguous residues (B/J/Z/X) are rejected by default (optional opt-in with warnings)
- Isotope / fragment views are **theoretical** OpenMS outputs — not measured spectra
- pyopenmsviz is not required/used
- Multiple modifications on the same residue position are not supported

---

## Future work

- Richer terminal modification UX
- Per-row modifications in CSV batch
- Optional theoretical fragment ion tables via OpenMS `TheoreticalSpectrumGenerator`
- More educational walkthroughs

---

## License

MIT — see [`LICENSE`](LICENSE).

OpenMS / pyOpenMS are separate projects with their own licenses; PepMZ depends on them as libraries.
