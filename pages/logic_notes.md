<!--
## Cohort & Marker Auto-Detection Logic

### 1. Cohort Detection from Filenames
- Automatically detects cohort identifiers from filenames such as:
  - `A3M-CD8.qptiff` → `Cohort = "A"`, `Cohort_Sub = 3`
  - `E1-M-CD3.qptiff` → `Cohort = "E"`, `Cohort_Sub = 1`
- The **"M"** between cohort and marker (e.g., in `A3M-CD8`) is ignored.

### 2. Word-Based Cohorts
- Detects named cohort labels (case-insensitive) such as:
  - **HRT**, **PAS**, **CTRL**, etc.
- Sets:
  - `Cohort = "<word>"`

### 3. Ignore Tokens (Case-Insensitive)
- Any match in the **ignore list** is skipped.
- Default ignores include:
  - `CD`, `CD3`, `CD8`
- Prevents false positives:
  - A lone letter (e.g., `Z`) is *not* treated as a cohort unless followed by a digit (e.g., `Z3`).

### 4. Marker Extraction
- Extracts **Marker** information (e.g., `CD3`, `CD8`) from filenames.
- Enables filtered or overlayed visualization:
  - **CD3 only**
  - **CD8 only**
  - **Both** → Overlays two bars per cohort for comparison.

### 5. Plotting & Behavior
- Keeps all other data and structure unchanged.
- All plots and group-bys use the detected `Cohort` (letter or word).
- A single **Global Y-max control** is applied across all plots for consistent scaling.
-->
