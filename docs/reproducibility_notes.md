# Reproducibility Notes

This document describes how to reproduce the computational procedures and analyses reported in:

**David Yeo, “Learning to Search with a Simulated Cortical Macrocolumn.”**

The repository contains the solver, Sudoku datasets, fixed train/test split, analysis programs, archived outputs, and generated tables and figures used in the manuscript.

---

## 1. Repository Structure

The main repository directories are:

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
│   ├── overlay_line_plot.py
│   └── Wilcoxon_signed_rank_test.py
│
├── data/
│   ├── Inkala_data/
│   ├── training_data/
│   └── test_data/
│
├── results/
│   ├── analysis_inputs/
│   ├── tables/
│   └── figures/
│
└── docs/
    └── reproducibility_notes.md
```

The main solver is:

```text
src/macrocolumn_solver.py
---

## 2. Software Environment

The code is written in Python and uses NumPy, TensorFlow/Keras, pandas, SciPy, and Matplotlib.

Install the required packages from:

```text
requirements.txt
```

A typical Windows setup is:

```text
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```text
source .venv/bin/activate
```

For the closest numerical reproduction, use the Python and package versions recorded in the repository.

---

## 3. Sudoku Data

Each Sudoku puzzle is a plain-text 9 × 9 grid of integers. Zero represents a blank cell.

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

The solver loads all valid 9 × 9 puzzle files from the directory specified by the user.

The main dataset contains 100 puzzles. The fixed 75-puzzle training set and 25-puzzle held-out test set are stored separately in:

data/training_data/
data/test_data/

Seventeen puzzles with maximum selected domain size equal to 1 were excluded from the held-out test candidate set because they require no non-forced branching decision. These puzzles were retained in the training set.

---

## 4. Solver Configuration

The fixed random seed is:

```text
SEED = 121252
```

The implemented macrocolumn contains seven competing minicolumns. Each minicolumn contains:

```text
1 LSTM layer with 10 units
3 Dense layers with 30 ReLU units each
```

The principal learning parameters are:

```text
GAMMA = 0.99
PHI_GAIN = 0.10
TD_LAMBDA = 0.95
EPSILON0 = 0.10
L1_REG = 1e-6
```

The solver uses deterministic forced-move propagation, learned soft winner-take-all/divisive-normalization cell selection, fixed ε-greedy exploration during training, and potential-shaped TD(λ) learning.

---

## 5. Running the Solver

From the repository root, run:

```text
python scripts/run_macrocolumn_solver.py
```

The program first requests the puzzle directory:

```text
Puzzle directory:
```

It then requests the number of trials:

```text
Enter the number of trials (0 to test & exit):
```

Enter:

```text
0
```

to evaluate the current model without learning or exploration.

Entering a positive number starts training. The program then requests the learning rate.

For the reported experimental runs, the solver display settings were:


```text
SHOW_PLOT = False
SHOW_MOVES = False
```

For each puzzle, the solver reports:

```text
Move attempts
Maximum selected domain size
Contradiction count
Backtrack count
```

If no solution is found, the program reports "No solution found."

Backtrack count is the principal search-efficiency measure used in the manuscript.

The solver prints results to the console. The archived result files were manually prepared from the reported console output. 
---

## 6. Checkpoints

A checkpoint is saved after each completed training trial. The solver saves checkpoints using Keras files of the form:

```text
macrocolumn_model (1).keras
macrocolumn_model (2).keras
macrocolumn_model (3).keras
...
```

When the solver starts, it automatically loads the highest-numbered matching checkpoint in the current working directory.
Therefore:

* remove or relocate existing checkpoints before generating the untrained trial-0 baseline;
* retain the latest checkpoint when continuing training;
* isolate the intended checkpoint when evaluating a particular training trial.

If no checkpoint is present, a new model is created.

Separate working directories may be used to keep checkpoints from different experimental conditions independent.

---

## 7. Learning-Rate Stress Tests

AI Escargot and Everest were each evaluated for 50 training trials at:

```text
0.01
0.001
0.0001
```

Each learning-rate condition should begin from the intended initial model state and should not reuse a checkpoint generated under another learning rate.

The recorded backtrack counts are used to generate:

* descriptive statistics and Shapiro-Wilk tests;
* Friedman learning-rate comparisons;
* pairwise Wilcoxon signed-rank tests with Holm-Bonferroni adjustment;
* learning-rate comparison plots.

---

## 8. Train/Test Experiment

The prior-experience experiment uses the fixed 75-puzzle training set and 25-puzzle held-out test set.

The model is trained on:

```text
data/training_data/
```

using:

```text
learning rate = 0.0001
```

Training continues for 100 trials. The solver shuffles the training-puzzle order after each completed trial using the seeded random-number generator. Saved checkpoints are evaluated at:

```text
trial 10
trial 20
trial 30
trial 40
trial 50
trial 60
trial 70
trial 80
trial 90
trial 100
```

The untrained trial-0 baseline is obtained from a newly initialized model.

Each saved checkpoint is evaluated on:

```text
data/test_data/
```

with:

```text
trials = 0
learning disabled
exploration disabled
epsilon = 0
```

The held-out test analysis compares trial-0 backtrack counts with the corresponding counts obtained from each trained checkpoint.

The test set is used to assess whether training improves search guidance on puzzles not encountered during training.

---

## 9. Statistical Analysis

The analysis programs generate the descriptive and inferential statistics reported in the manuscript. The reported summary statistics and normality-test results are:

```text
minimum
maximum
mean
median
population standard deviation
median absolute deviation
Shapiro-Wilk W
Shapiro-Wilk p-value
```

The population standard deviation is computed with:

```text
ddof = 0
```

The median absolute deviation is computed with:

```text
scipy.stats.median_abs_deviation(..., scale="normal")
```

For learning-rate analyses, the manually prepared input TXT file contains:

```text
column 0: trial number
column 1: backtrack counts for learning rate 0.01
column 2: backtrack counts for learning rate 0.001
column 3: backtrack counts for learning rate 0.0001
```

Column 0 identifies the trial and is not interpreted as an experimental outcome. Reported descriptive statistics are therefore based on columns 1–3.

Paired comparisons use Wilcoxon signed-rank tests. Holm-Bonferroni adjustment is applied where multiple paired comparisons are performed. Friedman tests are used for the omnibus stress-test learning-rate comparisons.

Because successive observations within a learning trajectory are serially dependent, the stress-test inferential results are interpreted as exploratory.

---

## 10. Numerical Reproducibility

A fixed random seed is used for Python, NumPy, and TensorFlow where possible. Exact numerical results may nevertheless vary across operating systems, TensorFlow versions, hardware, and CPU/GPU numerical implementations.

For the closest reproduction, use the archived software versions and the fixed data split and preserve the checkpoint procedure described above.

The archived raw outputs and processed analysis files provide the direct computational basis for the manuscript results.

