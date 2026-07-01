# Reproducibility Notes

This document describes how to reproduce the computational procedures and reported analyses for the manuscript:

**Learning to Search with a Simulated Cortical Macrocolumn**

The repository contains the source code, Sudoku puzzle datasets, train/test split information, experiment scripts, raw output files, statistical analysis scripts, and figure-generation scripts used to support the results reported in the manuscript.

## 1. Purpose of the Repository

This repository is intended to make the computational work reported in the manuscript inspectable and reproducible. It provides:

* the macrocolumn reinforcement-learning Sudoku solver;
* the Sudoku puzzle datasets used in the experiments;
* the training and held-out test split;
* scripts used to run the solver;
* scripts used to generate descriptive statistics;
* scripts used to perform Wilcoxon signed-rank tests and related corrections;
* scripts used to generate the learning-rate comparison line plots;
* raw or processed output files used to produce the manuscript tables and figures.

The main purpose of the archive is to allow reviewers and readers to trace the reported results back to the program, data, and analysis files from which they were generated.

## 2. Software Requirements

The code is written in Python. The main solver uses TensorFlow/Keras and NumPy. The analysis scripts use common scientific Python packages.

A typical environment requires:

```text
python
numpy
tensorflow
pandas
scipy
matplotlib
```

The package list should be recorded in:

```text
requirements.txt
```

A typical Windows setup is:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS or Linux, the virtual environment is usually activated with:

```bash
source .venv/bin/activate
```

## 3. Repository Layout

The intended repository structure is:

```text
macrocolumn-sudoku-solver/
│
├── README.md
├── CITATION.cff
├── LICENSE
├── MANIFEST.md
├── requirements.txt
│
├── src/
│   ├── __init__.py
│   └── macrocolumn_solver.py
│
├── scripts/
│   ├── run_macrocolumn_solver.py
│   ├── reproduce_descriptive_statistics.py
│   ├── reproduce_learning_rate_plots.py
│   └── reproduce_wilcoxon_tests.py
│
├── data/
│   ├── puzzles/
│   ├── train/
│   ├── test/
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

The main solver implementation is stored in:

```text
src/macrocolumn_solver.py
```

The command-line wrapper used to run the solver from the repository root is stored in:

```text
scripts/run_macrocolumn_solver.py
```

## 4. Main Solver

The main solver is a macrocolumn reinforcement-learning Sudoku solver embedded in depth-first search.

The solver:

* propagates forced Sudoku moves deterministically;
* uses learned action values to guide non-forced branching decisions;
* selects non-forced cells using soft winner-take-all competition and divisive normalization;
* orders admissible digits by estimated Q-value;
* samples only the first attempted digit during training from a fixed ε-greedy behavior policy;
* updates Q-values using potential-shaped TD(λ) learning;
* reports move attempts, maximum selected domain size, contradiction count, and backtrack count.

The principal implementation file is:

```text
src/macrocolumn_solver.py
```

To run the solver from the repository root:

```bash
python scripts/run_macrocolumn_solver.py
```

The program prompts for the puzzle directory:

```text
Puzzle directory:
```

It then prompts for the number of trials:

```text
Enter the number of trials (0 to test & exit):
```

Use:

```text
0
```

to run the solver in evaluation mode without learning.

Use a positive integer to run training. If training is selected, the program also prompts for the learning rate.

## 5. Sudoku Puzzle Format

Each Sudoku puzzle is stored as a plain text file containing a 9 × 9 grid of integers.

Use:

```text
0
```

for blank cells.

Example:

```text
0 0 3 0 2 0 6 0 0
9 0 0 3 0 5 0 0 1
0 0 1 8 0 6 4 0 0
0 0 8 1 0 2 9 0 0
7 0 0 0 0 0 0 0 8
0 0 6 7 0 8 2 0 0
0 0 2 6 0 9 5 0 0
8 0 0 2 0 3 0 0 9
0 0 5 0 1 0 3 0 0
```

The solver loads all valid 9 × 9 numeric text files from the directory supplied by the user.

## 6. Dataset Construction

The experimental dataset contained 100 Sudoku puzzles. The puzzles were drawn from newspaper and public-domain sources and ranged from easy to hard difficulty levels.

Two widely cited difficult Sudoku puzzles were also used as stress-test cases for learning-rate comparisons:

```text
AI Escargot
Everest
```

The paper reports that 17 of the 100 puzzles were forced-solution puzzles with maximum selected domain size equal to 1. These puzzles required no non-forced branching decision and therefore produced zero backtracks.

The forced-solution puzzles reported in the manuscript were:

```text
001
004
012
015
027
039
042
048
049
051
072
077
085
088
089
092
093
```

The prior-experience experiment used a 75/25 train-test split. The 25-puzzle test set was sampled only from puzzles with maximum selected domain size greater than 1, because forced-solution puzzles provide no opportunity to assess learned branching guidance.

The training and test directories should be preserved as separate folders, for example:

```text
data/train/
data/test/
```

The exact split should also be recorded in:

```text
data/train_test_split/
```

## 7. Main Experimental Parameters

The implemented macrocolumn model used seven competing minicolumns.

Each minicolumn contained:

```text
1 LSTM layer with 10 units
3 Dense layers with 30 ReLU units each
```

The resulting minicolumn capacity score was:

```text
10 + 3 × 30 = 100
```

The principal reinforcement-learning parameters were:

```text
GAMMA = 0.99
PHI_GAIN = 0.10
TD_LAMBDA = 0.95
EPSILON0 = 0.10
L1_REG = 1e-6
```

The fixed random seed used in the solver was:

```text
121252
```

This seed is applied to Python, NumPy, and TensorFlow where possible.

## 8. Solver Output Metrics

For each puzzle, the solver reports four main metrics:

```text
Move attempts
Maximum selected domain size
Contradiction count
Backtrack count
```

Backtrack count is the primary outcome measure in the manuscript because it directly measures wrong-path search effort that must later be undone.

The manuscript also uses the accounting relationship:

```text
initial blank cells = move attempts - backtrack count
```

after successful solution of a puzzle.

## 9. Checkpoint Notes

The solver saves Keras checkpoints using the prefix:

```text
macrocolumn_model
```

Checkpoint files are named in the form:

```text
macrocolumn_model (1).keras
macrocolumn_model (2).keras
macrocolumn_model (3).keras
...
```

When the solver starts, it attempts to load the most recent checkpoint in the current working directory.

For a clean trial-0 baseline, remove or relocate previous checkpoint files before running the baseline evaluation.

For trained evaluations, use the checkpoint corresponding to the desired training trial.

## 10. Reproducing the Learning-Rate Comparison

The learning-rate comparison was performed on the two Inkala stress-test puzzles:

```text
AI Escargot
Everest
```

The learning rates compared were:

```text
0.01
0.001
0.0001
```

Performance was assessed across the first 50 learning trials, excluding trial 0 from the descriptive and paired-comparison summaries.

The line plots reported as Figures 5 and 6 can be reproduced with:

```bash
python scripts/reproduce_learning_rate_plots.py
```

The descriptive statistics reported for the learning-rate comparison can be reproduced with:

```bash
python scripts/reproduce_descriptive_statistics.py
```

The Wilcoxon paired signed-rank comparisons can be reproduced with:

```bash
python scripts/reproduce_wilcoxon_tests.py
```

The output files used for these analyses should be stored in:

```text
results/raw_outputs/
```

Generated plots should be stored in:

```text
results/figures/
```

Processed statistical tables should be stored in:

```text
results/processed_tables/
```

## 11. Reproducing the Prior-Experience Experiment

The prior-experience experiment used the 75-puzzle training set and the 25-puzzle held-out test set.

The model was trained on the 75 training puzzles using:

```text
learning rate = 0.0001
```

Training-set performance was evaluated at:

```text
trial 0
trial 10
trial 20
trial 30
trial 40
trial 50
```

Held-out test-set performance was evaluated with learning disabled and exploration set to zero. The trial-0 baseline was obtained from a newly initialized model before training. Trained evaluations were obtained by loading the appropriate saved checkpoint.

The purpose of the test-set analysis was to determine whether the learned Q-values improved search guidance on novel puzzles, rather than merely improving performance on puzzles encountered during training.

## 12. Reproducing Descriptive Statistics

The descriptive statistics scripts should compute, where applicable:

```text
minimum
maximum
mean
median
standard deviation
median absolute deviation
Shapiro-Wilk W
Shapiro-Wilk p-value
```

The descriptive statistics reported in the manuscript include:

```text
Table 2: AI Escargot descriptive statistics
Table 3: Everest descriptive statistics
Table 7: training-set descriptive statistics
Table 9: test-set descriptive statistics
```

To run the descriptive statistics script:

```bash
python scripts/reproduce_descriptive_statistics.py
```

## 13. Reproducing Wilcoxon Tests

Because the backtrack-count distributions were non-normal, the manuscript used Wilcoxon signed-rank tests for paired comparisons.

The Wilcoxon analysis was used for:

```text
learning-rate comparisons
training-set trial-0 versus later trial comparisons
test-set trial-0 versus later trial comparisons
```

Where multiple paired comparisons were made, Holm-Bonferroni correction was applied.

The Wilcoxon-related tables reported in the manuscript include:

```text
Table 5: AI Escargot learning-rate comparisons
Table 6: Everest learning-rate comparisons
Table 8: training-set Wilcoxon comparisons
Table 10: test-set Wilcoxon comparisons
```

To run the Wilcoxon analysis script:

```bash
python scripts/reproduce_wilcoxon_tests.py
```

## 14. Notes on Statistical Interpretation

The learning-rate comparisons are exploratory because the observations within each learning trajectory are serially dependent. The Friedman and Wilcoxon tests are therefore useful for summarizing learning-rate trajectories, but they should not be interpreted as fully independent confirmatory tests.

The held-out test-set results showed a trend toward reduced backtracking after training, particularly around trials 30 and 40, but the trend did not remain statistically significant after Holm-Bonferroni correction.

The training-set results showed stronger evidence that the model acquired useful search guidance on puzzles encountered during training.

## 15. Expected Sources of Variation

The repository uses a fixed seed to improve reproducibility. However, exact numerical results may still vary across:

```text
operating systems
Python versions
TensorFlow versions
CPU versus GPU execution
hardware-specific numerical kernels
minor package-version differences
```

For this reason, the repository includes raw output files and processed table-generation files so that the manuscript results can be traced directly to the archived outputs.

## 16. Clean Reproduction Workflow

A typical reproduction workflow is:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Then run the solver:

```bash
python scripts/run_macrocolumn_solver.py
```

Then reproduce the analyses:

```bash
python scripts/reproduce_descriptive_statistics.py
python scripts/reproduce_wilcoxon_tests.py
python scripts/reproduce_learning_rate_plots.py
```

The generated results should correspond to the files stored under:

```text
results/
```

## 17. Notes for Reviewers

The repository is intended to support transparent review of the implementation and analyses. The key reproducibility materials are:

```text
src/macrocolumn_solver.py
data/
results/
scripts/
requirements.txt
README.md
MANIFEST.md
docs/reproducibility_notes.md
```

The solver implementation, training/test split, raw backtrack-count outputs, statistical summaries, and plotting scripts together provide the computational basis for the reported Tables 1–10 and Figures 5–6.

The results should be interpreted in the context stated in the manuscript: the solver demonstrates learned search guidance on training puzzles, while reliable generalization to novel puzzles remains an open question requiring additional experiments across repeated seeds, repeated train/test splits, larger puzzle sets, and additional constraint-satisfaction domains.
