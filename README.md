# Learning to Search with a Simulated Cortical Macrocolumn: Code, Data, and Results

This repository contains the source code, Sudoku puzzle data, fixed training/test partition, analysis programs, numerical analysis inputs, tables, figures, and reproducibility documentation associated with the manuscript:

**David Yeo, “Learning to Search with a Simulated Cortical Macrocolumn.”**

The project implements a biologically inspired macrocolumn reinforcement-learning Sudoku solver. A cortical-inspired action-value network is embedded within a depth-first-search procedure. Forced moves are propagated deterministically, while learned action values guide non-forced cell selection and digit ordering through soft winner-take-all competition and divisive normalization.

Sudoku is used as a compact constraint-satisfaction testbed for evaluating whether the proposed architecture can learn useful search guidance.

---

## Repository Contents

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

Detailed experimental procedures are provided in:

```text
docs/reproducibility_notes.md
```

---

## Project Components

The repository supports the following analyses:

1. Identification of forced-solution Sudoku puzzles.
2. Learning-rate comparisons using AI Escargot and Everest.
3. Training of the macrocolumn model on a fixed 75-puzzle training set.
4. Evaluation on a fixed 25-puzzle held-out test set.
5. Descriptive statistical and Shapiro-Wilk analyses.
6. Exploratory Friedman repeated-measures comparisons.
7. Wilcoxon signed-rank comparisons with Holm-Bonferroni correction.
8. Generation of the reported learning-rate figures.

Backtrack count is the principal measure of search efficiency used in the manuscript.

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

Install the required packages with:

```bash
pip install -r requirements.txt
```

The final archived `requirements.txt` should contain the exact package versions used to generate the reported results.

Jupyter Notebook or JupyterLab is not required. The repository programs are standard Python `.py` files.

---

## Running the Programs

Run all commands from the top-level repository directory.

### Run the macrocolumn solver

```bash
python scripts/run_macrocolumn_solver.py
```

This wrapper launches:

```text
src/macrocolumn_solver.py
```

The solver prompts for the puzzle directory, number of trials, and learning rate when training is requested.

### Generate descriptive statistics

```bash
python scripts/descriptive_statistics.py
```

### Run Friedman and Wilcoxon analyses

```bash
python scripts/Friedman_and_Wilcoxon_tests.py
```

### Generate learning-rate plots

```bash
python scripts/overlay_line_plot.py
```

Each analysis program prompts for the relevant TXT filename or full file path.

Exact model parameters, checkpoint procedures, puzzle ordering, experimental settings, analysis options, and data-column definitions are documented in:

```text
docs/reproducibility_notes.md
```

---

## Data

Sudoku puzzle files are stored under:

```text
data/
```

The subdirectories are:

```text
data/Inkala_data/
data/training_data/
data/test_data/
```

The supplied training and test directories contain the fixed partition used for the manuscript and should not be regenerated when reproducing the reported experiment.

Further information about the puzzle data is provided in:

```text
data/README.md
```

---

## Results

The `results/` directory contains the numerical analysis inputs, manuscript tables, and manuscript figures.

### Analysis inputs

```text
results/analysis_inputs/
```

The files in this directory preserve the numerical values used for the reported analyses and learning-rate plots:

```text
ai_escargot_counts.txt
everest_counts.txt
training_data_counts.txt
test_data_counts.txt
```

The solver originally displayed the relevant performance counts on the console. The manuscript values were manually transcribed into these analysis-input files. They are the direct numerical inputs to the archived analyses but are not unedited console logs.

Their formats are documented in:

```text
results/analysis_inputs/README.md
```

### Tables

```text
results/tables/
```

This directory contains the ten manuscript tables.

### Figures

```text
results/figures/
```

This directory contains the six manuscript figures.

---

## Reproducibility

The experiments use the fixed random seed:

```text
SEED = 121252
```

The repository includes the fixed puzzle datasets, analysis-input files, analysis programs, tables, figures, and experimental documentation required to inspect and reproduce the reported analyses.

Neural-network learning trajectories can vary across software versions, hardware, TensorFlow configurations, and low-level numerical implementations. The archived analysis-input files preserve the exact numerical values used to generate the manuscript’s statistical results and learning-rate figures.

Complete reproduction instructions are provided in:

```text
docs/reproducibility_notes.md
```

---

## Version Associated with the Manuscript

```text
Version: 1.0.0
DOI: To be added after Zenodo archival
```

The final repository release should be archived through Zenodo, and the resulting DOI should be added to this README, `CITATION.cff`, and the manuscript’s Data and Code Availability statement.

---

## Citation

Citation metadata are provided in:

```text
CITATION.cff
```

The manuscript should be cited separately from the supporting code, data, and results archive.

---

## License

The source code is released under the MIT License. See:

```text
LICENSE
```

Externally sourced data and figures remain subject to their original attribution and reuse terms.

---

## Author

**David Yeo**
Independent Researcher
Peterborough, Ontario, Canada
ORCID: https://orcid.org/0009-0008-1226-3189

---

## Contact

**David Yeo**
Email: [holondby@gmail.com](mailto:holondby@gmail.com)

