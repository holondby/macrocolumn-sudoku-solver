# Learning to Search with a Simulated Cortical Macrocolumn: Code, Data, and Results

This repository contains the source code, Sudoku puzzle data, fixed train/test split, experiment scripts, numerical analysis inputs, and supporting analysis files associated with the manuscript:

**David Yeo, “Learning to Search with a Simulated Cortical Macrocolumn.”**

The project implements a biologically inspired macrocolumn reinforcement-learning Sudoku solver. A cortical-inspired action-value network is embedded within a depth-first search procedure. Forced Sudoku moves are propagated deterministically, while non-forced branching decisions are guided by learned action-value estimates, soft winner-take-all competition, and divisive normalization.

Sudoku is used as a compact constraint-satisfaction testbed for evaluating whether the proposed architecture can learn useful search guidance.

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
│
├── src/
│   ├── __init__.py
│   └── macrocolumn_solver.py
│
├── scripts/
│   ├── run_macrocolumn_solver.py
│   ├── descriptive_statistics.py
│   ├── Friedman_chi_square_test.py
│   ├── Wilcoxon_signed_rank_test.py
│   └── overlay_line_plot.py
│
├── data/
│   ├── README.md
│   ├── Inkala_data/
│   ├── training_data/
│   └── test_data/
│
results/
│
├── analysis_inputs/
│   ├── AI_Escargot_learning_rate_counts.txt
│   ├── Everest_learning_rate_counts.txt
│   └── test_set_checkpoint_backtrack_counts.txt
│
├── tables/
│   ├── Table 1 - Forced_puzzles.txt
│   ├── Table 2 - AI_Escargot_descriptive_statistics.txt
│   ├── Table 3 - Everest_descriptive_statistics.txt
│   ├── Table 4 - Friedman_repeated_measures_comparison.txt
│   ├── Table 5 - AI_Escargot_Wilcoxon_signed_rank_test_results.txt
│   ├── Table 6 - Everest_Wilcoxon_signed_rank_test_results.txt
│   ├── Table 7 - Descriptive_Analysis_of_training_puzzles.txt
│   ├── Table 8 - Pre-training_vs_post-training_comparsion.txt
│   ├── Table 9 - Descriptive_analysis_of_test_puzzles.txt
│   └── Table 10 - Test_puzzle_Wilcoxon_signed_rank_test_results.txt
│
└── figures/
|   ├── Figure 1 - Cortical_minicolumn.png
|   ├── Figure 2 - Simulated_macrocolumn_model.png
|   ├── Figure 3 - AI_Escargot_puzzle.png
|   ├── Figure 4 - Everest_puzzle.png
|   ├── Figure 5 - AI_Escargot_learning _rate_comparison.png
|   └── Figure 6 - Everest_learning_rate_comparison.png
│
└── docs/
    └── reproducibility_notes.md
```

Detailed experimental and reproduction procedures are provided in `docs/reproducibility_notes.md`.

---

## Project Summary

The proposed solver learns while solving. A cortical-inspired macrocolumn action-value network is embedded directly within depth-first search so that search and learning are coupled during puzzle solving.

The model combines:

* an action space of 9 × 9 × 9 = 729 possible cell-digit assignments;
* deterministic propagation of forced moves;
* learned selection of non-forced branching cells;
* greedy ordering of admissible digits by estimated action value;
* multiple competing minicolumn pathways;
* divisive normalization;
* soft winner-take-all competition;
* potential-shaped learning targets;
* depth-first search with backtracking.

The primary performance measure reported in the manuscript is **backtrack count**, representing search effort spent pursuing branches that must subsequently be undone.

The central research question concerns the learning architecture rather than Sudoku itself. Sudoku provides a structured search domain in which forced moves can be propagated deterministically and learned guidance can be evaluated at non-forced branching decisions.

---

## Main Experimental Components

The repository contains the code, data, and analysis materials associated with the following manuscript experiments:

1. Identification of forced-solution Sudoku puzzles.
2. Learning-rate comparison using the AI Escargot and Everest Sudoku puzzles.
3. Training of the macrocolumn model on a fixed 75-puzzle training set.
4. Evaluation of saved training checkpoints on a fixed 25-puzzle held-out test set.
5. Descriptive statistical analysis, Friedman tests, Wilcoxon signed-rank comparisons, Holm-Bonferroni corrections, and generation of learning-rate figures.

Detailed experimental settings and procedures are described in `docs/reproducibility_notes.md`.

---

## Software Requirements

The code is written in Python. The programs were originally developed and run in an Anaconda Jupyter environment, but the repository version is provided as standard Python `.py` files for command-line execution.

The solver and analysis scripts use external Python packages including NumPy, pandas, SciPy, matplotlib, and TensorFlow.

A recommended setup is to create a dedicated conda environment:

```bash
conda create -n macrocolumn-sudoku python=3.10
```

Activate the environment:

```bash
conda activate macrocolumn-sudoku
```

Install the required packages:

```bash
pip install -r requirements.txt
```

The required Python packages are listed in `requirements.txt`.

Jupyter Notebook or JupyterLab is not required to run the repository programs.

---

## Running the Programs

Run the programs from the top-level repository directory.

### 1. Run the macrocolumn solver

```bash
python scripts/run_macrocolumn_solver.py
```

This command-line wrapper launches the main macrocolumn Sudoku solver implemented in:

```text
src/macrocolumn_solver.py
```

The solver prompts for the puzzle directory and run parameters required for the selected experiment.

The same solver implementation is used for training and evaluation. For held-out test-set evaluation, learning and exploration are disabled so that performance reflects the saved learned model state rather than continued training or random exploration.

Detailed parameter settings for the manuscript experiments are provided in `docs/reproducibility_notes.md`.

### 2. Generate descriptive statistics

```bash
python scripts/descriptive_statistics.py
```

The program prompts for a TXT filename or full file path and computes descriptive statistics and Shapiro-Wilk normality-test results for the numerical columns in the supplied file.

### 3. Run Friedman and Wilcoxon signed-rank analyses

```bash
python scripts/Wilcoxon_signed_rank_test.py
```

The program prompts for a TXT filename or full file path containing matched numerical observations. It performs a Friedman omnibus test when at least three related conditions are supplied and performs pairwise Wilcoxon signed-rank comparisons with Holm-Bonferroni correction.

### 4. Generate overlay line plots

```bash
python scripts/overlay_line_plot.py
```

The program prompts for a TXT filename or full file path containing trial numbers and backtrack counts for learning rates 0.01, 0.001, and 0.0001. It generates an overlay line plot and saves the plot as a PNG file.

The numerical files used for the reported analyses are archived in `results/analysis_inputs/`.

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

The data directories are:

```text
data/Inkala_data/
data/training_data/
data/test_data/
```

### `data/Inkala_data/`

This directory contains the difficult Sudoku puzzles used for the learning-rate comparison experiments, specifically AI Escargot and Everest.

The puzzles were used to compare learning rates of:

```text
0.01
0.001
0.0001
```

across the first 50 learning trials.

### `data/training_data/`

This directory contains the fixed 75-puzzle training set used for the prior-experience experiment.

During training, the solver learns across the supplied puzzle directory. The training-puzzle order is shuffled after each completed trial using the fixed seeded random-number procedure.

Training is performed using the experimental settings described in `docs/reproducibility_notes.md`.

### `data/test_data/`

This directory contains the fixed 25-puzzle held-out test set.

The test puzzles are not used for parameter updates during training. Held-out evaluation is conducted with learning disabled and exploration disabled.

The untrained trial-0 baseline and saved training checkpoints are evaluated on the same fixed test set.

---

## Results

Files supporting the reported analyses are organized under:

```text
results/
```

### `results/analysis_inputs/`

This directory contains the numerical TXT files used as input to the analysis scripts.

The solver reports relevant performance counts during execution. Where the solver reports counts to the screen rather than writing them directly to a file, the numerical counts used in the manuscript analyses were manually transcribed into the archived TXT analysis-input files.

These files preserve the numerical values used for the reported statistical analyses and figures.

### `results/tables/`

This directory contains archived statistical summaries and test results associated with the manuscript analyses.

These materials may include descriptive statistics, Friedman test results, Wilcoxon signed-rank results, and Holm-Bonferroni-adjusted comparisons.

### `results/figures/`

This directory contains figures associated with the reported experiments, including learning-rate comparison plots.

The analysis-input files required to reproduce the figures are provided in `results/analysis_inputs/`.

---

## Reproducibility

The experiments use the fixed random seed:

```text
SEED = 121252
```

The prior-experience experiment uses a fixed 75-puzzle training set and a fixed 25-puzzle held-out test set. The supplied train/test partition should be preserved when reproducing the reported experiment.

Detailed reproduction instructions, including the learning-rate experiments, training procedure, checkpoint evaluations, trial-0 baseline, and held-out test procedure, are provided in:

```text
docs/reproducibility_notes.md
```

Because neural-network execution can depend on software versions, numerical libraries, hardware, and TensorFlow configuration, exact numerical replication on a different system may vary. The archived analysis-input files preserve the numerical results used for the statistical analyses and figures reported in the manuscript.

---

## Version Associated with the Manuscript

The repository version intended for archival with the manuscript is:

```text
Version: 1.0.0
```

The Zenodo DOI will be added after archival.

```text
DOI: To be added after Zenodo archival.
```

---

## Citation

Citation metadata are provided in:

```text
CITATION.cff
```

After the repository is archived through Zenodo and a DOI is assigned, the DOI and final archival citation should be added to this README and the citation metadata updated as required.

The manuscript should be cited separately from the supporting code, data, and results archive.

---

## License

The source code is released under the MIT License. See `LICENSE` for details.

The data and result files are provided to support inspection and reproduction of the experiments associated with the manuscript.

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

This repository provides the solver implementation, experimental scripts, fixed puzzle datasets, numerical analysis inputs, and supporting statistical and figure materials associated with the manuscript.

Detailed reproduction procedures are provided in `docs/reproducibility_notes.md`.
