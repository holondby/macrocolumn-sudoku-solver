# Learning to Search with a Simulated Cortical Macrocolumn: Code, Data, and Results

This repository contains the source code, Sudoku puzzle data, train/test split, experiment scripts, raw outputs, and analysis files associated with the manuscript:

**David Yeo, “Learning to Search with a Simulated Cortical Macrocolumn.”**

The project implements a biologically inspired macrocolumn reinforcement-learning Sudoku solver. The solver embeds a cortical-inspired action-value network within a depth-first search procedure. Forced Sudoku moves are propagated deterministically, while non-forced branching decisions are guided by learned action-value estimates, soft winner-take-all competition, and divisive normalization.

The repository is intended to support transparency and reproducibility for journal review and publication.

---

## Repository Contents

The repository is organized as follows:

```text
macrocolumn-sudoku-solver/
│
├── README.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── MANIFEST.md
│
├── src/
│   ├── macrocolumn_solver.py
│   ├── model.py
│   ├── sudoku_utils.py
│   └── statistics_utils.py
│
├── scripts/
│   ├── run_training.py
│   ├── run_test_evaluation.py
│   ├── reproduce_tables.py
│   └── reproduce_figures.py
│
├── data/
│   ├── puzzles/
│   └── train_test_split/
│
├── results/
│   ├── raw_outputs/
│   ├── processed_tables/
│   └── figures/
│
└── docs/
    └── reproducibility_notes.md
```

The exact file inventory is described in `MANIFEST.md`.

---

## Project Summary

The solver uses Sudoku as a compact constraint-satisfaction testbed for learned search guidance. The model combines:

* a 9 × 9 × 9 action space representing possible cell-digit assignments;
* deterministic propagation of forced cells;
* learned selection of non-forced branching cells;
* greedy ordering of admissible digits by estimated action value;
* multiple competing minicolumn pathways;
* divisive normalization;
* soft winner-take-all integration;
* potential-shaped learning targets;
* depth-first search with backtracking.

The main outcome measure is **backtrack count**, which measures the amount of wrong-path search effort that must be undone. Additional measures include move attempts, maximum selected domain size, and contradiction count.

---

## Main Experimental Components

The repository contains material for the following analyses reported in the manuscript:

1. Identification of forced-solution Sudoku puzzles.
2. Learning-rate comparison on the AI Escargot and Everest Sudoku puzzles.
3. Training-set evaluation across repeated learning trials.
4. Held-out test-set evaluation with learning disabled and exploration set to zero.
5. Generation of descriptive statistics, Wilcoxon signed-rank tests, Holm-Bonferroni corrections, and figures/tables used in the manuscript.

---

## Software Requirements

The code is written in Python.

Recommended setup:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install the required packages with:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should contain the Python packages needed to run the solver and reproduce the analyses, such as NumPy, pandas, SciPy, matplotlib, and TensorFlow.

---

## Reproducing the Main Analyses

From the top-level repository folder, run the following commands.

### 1. Run training

```bash
python scripts/run_training.py
```

This script trains the macrocolumn Sudoku solver on the training puzzle set.

### 2. Run held-out test evaluation

```bash
python scripts/run_test_evaluation.py
```

This script evaluates saved trained models on the held-out test puzzle set with learning disabled and exploration set to zero.

### 3. Reproduce statistical tables

```bash
python scripts/reproduce_tables.py
```

This script regenerates the descriptive and inferential statistical summaries reported in the manuscript.

### 4. Reproduce figures

```bash
python scripts/reproduce_figures.py
```

This script regenerates the figure files from the raw or processed result files.

---

## Data

Sudoku puzzle data are stored in:

```text
data/puzzles/
```

The train/test split is stored in:

```text
data/train_test_split/
```

The manuscript uses a 100-puzzle dataset. Forced-solution puzzles, defined by maximum selected domain size equal to 1, are identified separately because they contain no non-forced branching decisions. The held-out test set consists of 25 puzzles sampled from puzzles requiring non-forced branching decisions. The remaining puzzles are used for training.

The two difficult Sudoku stress-test puzzles, AI Escargot and Everest, are included for the learning-rate comparison.

---

## Results

Raw output files are stored in:

```text
results/raw_outputs/
```

Processed tables are stored in:

```text
results/processed_tables/
```

Generated figures are stored in:

```text
results/figures/
```

The raw outputs include backtrack counts, contradiction counts, maximum selected domain size values, and related performance measures used to generate the manuscript tables and figures.

---

## Random Seeds and Reproducibility

The experiments use fixed random-seed settings where applicable. The manuscript evaluates a fixed train/test split and fixed experimental configuration. Because neural-network training and numerical libraries may differ across systems, exact numerical reproduction may depend on the Python version, package versions, hardware, and TensorFlow configuration.

For the archived journal-submission version, the intended reproduction target is the set of reported statistical summaries, raw outputs, tables, and figures associated with the released repository version.

---

## Version Associated with the Manuscript

The version associated with the submitted manuscript is:

```text
Version: 1.0.0
```

After archival through Zenodo, the DOI will be listed here:

```text
DOI: https://doi.org/10.5281/zenodo.xxxxxxx
```

Replace the placeholder DOI with the actual DOI assigned by Zenodo.

---

## Citation

If you use this code, data, or results archive, please cite the archived repository:

```text
Yeo, D. (2026). Learning to Search with a Simulated Cortical Macrocolumn:
Code, Data, and Results (Version 1.0.0) [Software and data]. Zenodo.
https://doi.org/10.5281/zenodo.xxxxxxx
```

The citation metadata are also provided in `CITATION.cff`.

---

## License

The source code is released under the MIT License. See `LICENSE` for details.

Data and result files are provided to support transparency, review, and reproducibility of the associated manuscript. If a separate data license is added, it should be described here and in the relevant data documentation.

---

## Author

David Yeo
Independent Researcher
Peterborough, Ontario, Canada
ORCID: https://orcid.org/0009-0008-1226-3189

---

## Contact

For questions about the repository or the associated manuscript, contact:

David Yeo
Email: [holondby@gmail.com](mailto:holondby@gmail.com)

---

## Notes for Reviewers

This repository is intended to provide the implementation and supporting materials needed to inspect and reproduce the analyses reported in the manuscript. The repository includes the solver implementation, experimental scripts, data files, raw outputs, and table/figure generation files.

The manuscript itself should be cited separately from this repository. This repository provides the supporting software, data, and reproducibility materials associated with the manuscript.
