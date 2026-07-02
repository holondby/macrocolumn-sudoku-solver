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
│   ├── __init__.py
│   └── macrocolumn_solver.py
│
├── scripts/
│   ├── run_macrocolumn_solver.py
│   ├── descriptive_statistics.py
│   ├── Wilcoxon_signed_rank_test.py
│   └── overlay_line_plot.py
│
├── data/
│   ├── README.md
│   ├── puzzles/
│   │   ├── train/
│   │   ├── test/
│   │   └── stress_tests/
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

The code is written in Python. The programs were originally developed and run in an Anaconda Jupyter environment, but the final repository version has been converted to standard Python `.py` files for reproducibility and command-line execution.

The solver and analysis scripts require several external Python packages, including NumPy, pandas, SciPy, matplotlib, and TensorFlow.

A recommended setup is to create a dedicated conda environment for the repository:

```bash
conda create -n macrocolumn-sudoku python=3.10
```

Activate the environment with:

```bash
conda activate macrocolumn-sudoku
```

Install the required packages with:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file lists the Python packages needed to run the solver and reproduce the analyses.

Jupyter Notebook or JupyterLab is not required to run the final `.py` files. If the code is opened or tested in Jupyter, Jupyter should be launched from within the activated conda environment so that it uses the same installed packages.

---

## Reproducing the Main Analyses

From the top-level repository folder, run the relevant scripts below.

### 1. Run the macrocolumn solver

```bash
python scripts/run_macrocolumn_solver.py
```

This script runs the macrocolumn Sudoku solver using the selected puzzle directory, learning parameters, and output settings specified in the program. The same solver script can be used for training-set runs, stress-test learning-rate comparisons, and held-out test-set evaluations by using the appropriate input files and parameter settings.

For held-out test-set evaluation, learning should be disabled and exploration should be set to zero so that performance reflects the learned model state rather than continued training or random exploration.

### 2. Generate descriptive statistics

```bash
python scripts/descriptive_statistics.py
```

This script computes descriptive statistical summaries from the relevant raw output files.

### 3. Run Wilcoxon signed-rank analyses

```bash
python scripts/Wilcoxon_signed_rank_test.py
```

This script performs the Wilcoxon signed-rank analyses used in the manuscript.

### 4. Generate overlay line plots

```bash
python scripts/overlay_line_plot.py
```

This script generates overlay line plots from the relevant output files.

---

## Optional Runtime Output Settings

The solver includes optional display settings that control how much information is shown during a run.

The `SHOW_PLOTS` parameter controls whether the initial and final Sudoku states are displayed. Setting `SHOW_PLOTS = True` displays these Sudoku states, while setting `SHOW_PLOTS = False` suppresses this visual output.

The `SHOW_PATH` parameter controls whether the solution path and associated performance counts are displayed. Setting `SHOW_PATH = True` prints this additional run information, while setting `SHOW_PATH = False` suppresses it.

These settings affect only the amount of displayed output. They do not change the learning procedure, puzzle selection, train/test split, learned model state, or computed performance measures.

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

Additional information about the data files is provided in:

```text
data/README.md
```

The manuscript uses a 100-puzzle dataset. Forced-solution puzzles, defined by maximum selected domain size equal to 1, are identified separately because they contain no non-forced branching decisions. The held-out test set consists of 25 puzzles sampled from puzzles requiring non-forced branching decisions. The remaining puzzles are used for training.

The `data/puzzles/stress_tests/` directory contains the difficult Sudoku puzzles used for the learning-rate comparison experiments, namely Inkala’s AI Escargot and Everest puzzles. These puzzles were used to compare learning rates of 0.01, 0.001, and 0.0001 across the first 50 learning trials, with performance assessed primarily by backtrack count.

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

For practical convenience, some learning runs were conducted in consecutive 10-trial increments rather than as a single uninterrupted run. Thus, saved output files may correspond to trial blocks such as 1–10, 11–20, 21–30, and so on, up to the total number of trials used in a given experiment. These blocks should be interpreted as successive stages of the same learning procedure, not as independent 10-trial experiments. The learned model state was carried forward from one block to the next, so later blocks reflect continued learning from the earlier blocks.

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

