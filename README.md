# Learning to Search with a Simulated Cortical Macrocolumn: Code, Data, and Results

This repository contains the source code, Sudoku puzzle data, fixed training/test partition, analysis programs, numerical analysis inputs, tables, figures, and reproducibility documentation associated with the manuscript:

**David Yeo, “Learning to Search with a Simulated Cortical Macrocolumn.”**

The project implements a biologically inspired macrocolumn reinforcement-learning Sudoku solver. A cortical-inspired action-value network is embedded within a depth-first-search procedure. Forced Sudoku moves are propagated deterministically, while non-forced branching decisions are guided by learned action-value estimates, soft winner-take-all competition, and divisive normalization.

Sudoku is used as a compact constraint-satisfaction testbed for evaluating whether the proposed architecture can learn useful search guidance.

The repository is intended to support transparency, inspection, and reproducibility for journal review and publication.

---

## Repository Contents

The repository is organized as follows:

```text
macrocolumn-sudoku-solver/
│
├── .gitignore
├── README.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
│
├── src/
│   ├── __init__.py
│   └── macrocolumn_solver.py
│
├── scripts/
│   ├── run_macrocolumn_solver.py
│   ├── descriptive_statistics.py
│   ├── Friedman_and_Wilcoxon_tests.py
│   └── overlay_line_plot.py
│
├── data/
│   ├── README.md
│   ├── Inkala_data/
│   ├── training_data/
│   └── test_data/
│
├── results/
│   ├── analysis_inputs/
│   │   ├── README.md
│   │   ├── ai_escargot_counts.txt
│   │   ├── everest_counts.txt
│   │   ├── training_data_counts.txt
│   │   └── test_data_counts.txt
│   │
│   ├── tables/
│   │   ├── Table 1 - Forced_puzzles.txt
│   │   ├── Table 2 - AI_Escargot_descriptive_statistics.txt
│   │   ├── Table 3 - Everest_descriptive_statistics.txt
│   │   ├── Table 4 - Friedman_repeated-measures_comparison.txt
│   │   ├── Table 5 - AI_Escargot_Wilcoxon_signed-rank_test_results.txt
│   │   ├── Table 6 - Everest _Wilcoxon_signed-rank_test_results.txt
│   │   ├── Table 7 - Descriptive_analysis_of_training_puzzles.txt
│   │   ├── Table 8 - Pre-training_vs_post-training_comparison.txt
│   │   ├── Table 9 - Descriptive_analysis_of_test_puzzles.txt
│   │   └── Table 10 - Test_puzzle_ Wilcoxon_signed-rank_test_results.txt
│   │
│   └── figures/
│       ├── Figure 1 - Cortical_minicolumn.png
│       ├── Figure 2 - Simulated_macrocolumn_model.png
│       ├── Figure 3 - AI_Escargot_puzzle.png
│       ├── Figure 4 - Everest_puzzle.png
│       ├── Figure 5 - AI_Escargot_learning _rate_comparison.png
│       └── Figure 6 - Everest_learning_rate_comparison.png
│
└── docs/
    └── reproducibility_notes.md
```

Detailed experimental procedures and reproduction instructions are provided in:

```text
docs/reproducibility_notes.md
```

---

## Project Summary

The proposed solver learns while solving. A cortical-inspired macrocolumn action-value network is embedded directly within depth-first search so that search and learning are coupled during puzzle solving.

The model combines:

* a 9 × 9 × 9 action space representing 729 possible cell-digit assignments;
* deterministic propagation of forced moves;
* learned selection of non-forced branching cells;
* greedy ordering of admissible digits by estimated action value;
* fixed-ε exploration of the first attempted digit during training;
* multiple competing recurrent minicolumn pathways;
* divisive normalization;
* soft winner-take-all competition;
* potential-shaped TD(λ) learning targets;
* depth-first search with backtracking.

The primary performance measure reported in the manuscript is **backtrack count**, representing search effort spent pursuing branches that must subsequently be undone.

The central research question concerns the learning architecture rather than Sudoku itself. Sudoku provides a structured domain in which forced moves can be propagated deterministically and learned guidance can be evaluated at non-forced branching decisions.

---

## Main Experimental Components

The repository contains code, data, and analysis materials associated with:

1. Identification of forced-solution Sudoku puzzles.
2. Learning-rate comparisons using the AI Escargot and Everest puzzles.
3. Training of the macrocolumn model on a fixed 75-puzzle training set.
4. Evaluation of saved training checkpoints on a fixed 25-puzzle held-out test set.
5. Descriptive statistical analyses and Shapiro-Wilk tests.
6. Exploratory Friedman repeated-measures comparisons.
7. Wilcoxon signed-rank comparisons with Holm-Bonferroni correction.
8. Generation of the learning-rate comparison figures.

The learning-rate stress tests use 50 learning trials. The prior-experience experiment uses 50 training trials, with results evaluated at trials 0, 10, 20, 30, 40, and 50.

Detailed experimental settings are provided in:

```text
docs/reproducibility_notes.md
```

---

## Software Requirements

The code is written in Python and uses:

```text
NumPy
pandas
SciPy
Matplotlib
TensorFlow/Keras
```

The required packages are listed in:

```text
requirements.txt
```

A Python virtual environment can be created on Windows with:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

The final archived `requirements.txt` should record the exact package versions used to generate the reported results.

Jupyter Notebook or JupyterLab is not required. The repository programs are standard Python `.py` files that can be executed from a command prompt or terminal.

---

## Running the Programs

Run all programs from the top-level repository directory.

### 1. Run the macrocolumn solver

```bash
python scripts/run_macrocolumn_solver.py
```

This command-line wrapper launches the main solver implemented in:

```text
src/macrocolumn_solver.py
```

The solver prompts for:

* the directory containing the puzzle or puzzles;
* the number of training trials;
* the learning rate when training is requested.

Entering `0` for the number of trials evaluates the current model with learning and exploration disabled and then exits.

The same solver implementation is used for training and evaluation. Detailed parameter settings, checkpoint procedures, and experimental instructions are provided in:

```text
docs/reproducibility_notes.md
```

### 2. Generate descriptive statistics

```bash
python scripts/descriptive_statistics.py
```

The program prompts for a TXT filename or full file path. It excludes column 0 from the outcome analysis and calculates descriptive statistics and Shapiro-Wilk normality-test results for the remaining numerical columns.

### 3. Run Friedman and Wilcoxon analyses

```bash
python scripts/Friedman_and_Wilcoxon_tests.py
```

The program prompts for a TXT filename or full file path containing matched numerical observations.

It can:

* perform a Friedman comparison when at least three matched conditions are supplied;
* compare all numerical columns pairwise;
* compare the first numerical column only with each later column;
* calculate two-sided and directional Wilcoxon signed-rank results;
* apply Holm-Bonferroni correction within each selected family of comparisons.

### 4. Generate overlay line plots

```bash
python scripts/overlay_line_plot.py
```

The program prompts for a TXT filename or full file path containing trial numbers and backtrack counts for learning rates:

```text
0.01
0.001
0.0001
```

It generates and saves an overlay line plot as a PNG file.

The numerical files used for the reported analyses are archived in:

```text
results/analysis_inputs/
```

---

## Data

Sudoku puzzle data are stored in:

```text
data/
```

Additional information about the puzzle files and dataset organization is provided in:

```text
data/README.md
```

### `data/Inkala_data/`

This directory contains the AI Escargot and Everest puzzles used for the learning-rate stress tests.

Each puzzle was evaluated across 50 learning trials using learning rates:

```text
0.01
0.001
0.0001
```

### `data/training_data/`

This directory contains the fixed 75-puzzle training set used for the prior-experience experiment.

The first training trial begins with puzzles in case-insensitive filename order. The file names can therefore be used to specify the initial sequence, such as easy to hard. After each completed trial, the puzzle order is shuffled using Python’s seeded random-number generator.

The reported 50-trial training sequence was conducted in consecutive 10-trial increments within the same running program session, preserving the learned model state and evolving puzzle-order sequence.

### `data/test_data/`

This directory contains the fixed 25-puzzle held-out test set.

The test puzzles are not used for parameter updates during training. Held-out evaluation is conducted with:

```text
learning disabled
exploration disabled
epsilon = 0
```

The newly initialized trial-0 model and the trial-10, 20, 30, 40, and 50 checkpoints are evaluated on the same fixed test set.

The supplied training/test partition should not be regenerated when reproducing the reported experiment.

---

## Results

Files supporting the reported analyses are organized under:

```text
results/
```

### `results/analysis_inputs/`

This directory contains the numerical, whitespace-separated TXT files used as inputs to the analysis programs:

```text
ai_escargot_counts.txt
everest_counts.txt
training_data_counts.txt
test_data_counts.txt
```

The solver originally displayed the relevant performance counts on the console. The counts used for the manuscript analyses were manually transcribed into these archived analysis-input files.

The files preserve the numerical values used to generate the reported statistical summaries and learning-rate figures. They are direct analysis inputs, but they are not unedited console logs.

The format of each file is documented in:

```text
results/analysis_inputs/README.md
```

### `results/tables/`

This directory contains the ten archived manuscript tables:

* forced-puzzle results;
* AI Escargot and Everest descriptive statistics;
* exploratory Friedman results;
* AI Escargot and Everest Wilcoxon results;
* training-set descriptive and Wilcoxon results;
* held-out test-set descriptive and Wilcoxon results.

### `results/figures/`

This directory contains the six manuscript figures:

* the cortical minicolumn illustration;
* the simulated macrocolumn model;
* the AI Escargot and Everest puzzles;
* the AI Escargot and Everest learning-rate comparison plots.

The numerical inputs used to generate Figures 5 and 6 are provided in:

```text
results/analysis_inputs/
```

---

## Reproducibility

The experiments use the fixed random seed:

```text
SEED = 121252
```

The seed is applied to Python, NumPy, and TensorFlow where supported.

The prior-experience experiment uses:

```text
75 training puzzles
25 held-out test puzzles
learning rate = 0.0001
training trials = 50
```

The reported checkpoints are:

```text
trial 0
trial 10
trial 20
trial 30
trial 40
trial 50
```

For held-out evaluation, learning and exploration are disabled so that performance reflects the selected model checkpoint rather than continued training or random exploration.

Detailed reproduction instructions—including model parameters, puzzle ordering, checkpoint handling, learning-rate experiments, statistical procedures, and held-out evaluation—are provided in:

```text
docs/reproducibility_notes.md
```

Neural-network results can depend on software versions, numerical libraries, hardware, and TensorFlow configuration. Exact numerical replication may therefore vary across systems.

The archived analysis-input files preserve the numerical values used for the statistical analyses and figures reported in the manuscript.

---

## Version Associated with the Manuscript

The repository version intended for archival with the manuscript is:

```text
Version: 1.0.0
```

The Zenodo DOI will be added after archival:

```text
DOI: To be added after Zenodo archival
```

---

## Citation

Citation metadata are provided in:

```text
CITATION.cff
```

After the repository is archived through Zenodo and a DOI is assigned, the DOI and final archival citation should be added to this README and to `CITATION.cff`.

The manuscript should be cited separately from the supporting code, data, and results archive.

---

## License

The source code is released under the MIT License. See:

```text
LICENSE
```

The puzzle data, tables, and figures are provided to support inspection and reproduction of the experiments associated with the manuscript. Any externally sourced data or figures remain subject to their original attribution and reuse terms.

---

## Author

**David Yeo**
Independent Researcher
Peterborough, Ontario, Canada
ORCID: https://orcid.org/0009-0008-1226-3189

---

## Contact

For questions about the repository or associated manuscript:

**David Yeo**
Email: [holondby@gmail.com](mailto:holondby@gmail.com)

---

## Notes for Reviewers

This repository provides:

* the macrocolumn Sudoku solver;
* the fixed puzzle datasets and training/test partition;
* the statistical and plotting programs;
* the numerical analysis-input files;
* the manuscript tables and figures;
* detailed reproducibility documentation.

Complete experimental and analysis procedures are provided in:

```text
docs/reproducibility_notes.md
```

