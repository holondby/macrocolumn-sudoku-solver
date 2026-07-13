# Reproducibility Notes

This document describes the computational procedures and analyses associated with:

**David Yeo, “Learning to Search with a Simulated Cortical Macrocolumn.”**

The repository contains the macrocolumn Sudoku solver, the Sudoku puzzle datasets, the fixed training/test partition, general analysis programs, numerical analysis-input files, and the tables and figures associated with the manuscript.

The archived numerical files in `results/analysis_inputs/` preserve the values used for the reported analyses. Where the solver originally displayed results on the console rather than writing them directly to a data file, the relevant counts were manually transcribed into these analysis-input files. They should therefore be regarded as the direct inputs to the archived statistical analyses, but not as unedited console logs.

---

## 1. Repository Structure

The main repository directories and files are:

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

│
└── docs/
    └── reproducibility_notes.md
```

The main solver implementation is:

```text
src/macrocolumn_solver.py
```

The command-line wrapper used to launch it from the repository root is:

```text
scripts/run_macrocolumn_solver.py
```

---

## 2. Software Environment

The code is written in Python and uses the following external packages:

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

A compatible environment can be created on Windows with:

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

The final archived release should record the exact Python and package versions used to generate the reported results. Because TensorFlow and SciPy behaviour can vary across versions, exact numerical reproduction is most likely when the pinned versions in the archival `requirements.txt` file are used.

Jupyter Notebook or JupyterLab is not required. The repository programs are standard Python `.py` files and can be executed from a command prompt or terminal.

---

## 3. Sudoku Data Format

Each Sudoku puzzle is stored as a plain-text 9 × 9 grid of integers.

The value `0` represents a blank cell. For example:

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

The solver loads all valid puzzle files from the directory selected by the user.

The manuscript experiments use the following directories:

```text
data/Inkala_data/
data/training_data/
data/test_data/
```

### `data/Inkala_data/`

This directory contains the AI Escargot and Everest puzzles used for the learning-rate stress tests.

### `data/training_data/`

This directory contains the fixed 75-puzzle training set used for the prior-experience experiment.

### `data/test_data/`

This directory contains the fixed 25-puzzle held-out test set.

The 100-puzzle dataset contained 17 puzzles with a maximum selected domain size of 1. These puzzles required no non-forced branching and were excluded from the set of puzzles eligible for random test-set selection. They were retained in the training set.

The supplied training and test directories represent the fixed partition used for the manuscript. The partition should not be regenerated when reproducing the reported experiment.

---

## 4. Solver Configuration

The fixed random seed is:

```text
SEED = 121252
```

The seed is applied to Python, NumPy, and TensorFlow where supported. Python’s random-number generator controls puzzle-order shuffling, NumPy controls exploratory action sampling and minibatch ordering, and TensorFlow controls network initialization and applicable TensorFlow operations.

### Network Architecture

The input is a 9 × 9 × 9 one-hot representation of the Sudoku state.

The convolutional encoder uses:

```text
CONV_FILTERS = 36
KERNEL_SIZE = 3
```

Its structure is:

```text
Conv2D:
36 filters
3 × 3 kernel
ReLU activation
same padding

Conv2D:
36 filters
3 × 3 kernel
stride = 2
ReLU activation
valid padding

Reshape to a 16 × 36 sequence
Layer normalization
```

The macrocolumn contains seven competing minicolumns:

```text
N_COLUMNS = 7
```

Each minicolumn contains:

```text
LSTM_UNITS = 10
N_LAYERS = 3
DENSE_UNITS = 30
```

Thus, each minicolumn consists of one LSTM layer with 10 units followed by three Dense layers with 30 ReLU units each. Its capacity score is:

```text
10 + (3 × 30) = 100
```

### Minicolumn Competition and Normalization

The main competition and normalization parameters are:

```text
GAIN = 1.15
INH_SIGMA = 1.5
OFFSET_DENOM = 0.01
WTA_BETA = 2.0
RING_TOPOLOGY = True
```

The minicolumn feature vectors undergo ring-distance Gaussian divisive normalization. A separate linear 729-action Q-value head is then applied to each inhibited minicolumn output.

Each minicolumn is scored by its largest estimated Q-value among the currently legal actions. The seven Q-value heads are then combined using soft winner-take-all weights to produce the final 729-element action-value output.

Candidate Sudoku cells undergo a separate soft winner-take-all and divisive-normalization process. At this search-policy level, peer relationships are defined by shared rows, columns, or 3 × 3 boxes rather than by minicolumn ring distance.

### Reinforcement-Learning Parameters

The principal learning parameters are:

```text
GAMMA = 0.99
PHI_GAIN = 0.10
TD_LAMBDA = 0.95
EPSILON0 = 0.10
L1_REG = 1e-6
```

The sparse terminal rewards are:

```text
R_SOLVED = 1.0
R_CONTRA = -1.0
```

Queued TD optimization uses:

```text
TD_FLUSH_EVERY = 32
TD_BATCH_SIZE = 32
TD_BUFFER_MAX = 256
```

### Optimization

The network is trained using:

```text
Optimizer: Adam
Loss: Huber loss plus Dense-layer L1 regularization
Gradient clipping: clipnorm = 1.0
```

The learning rate is supplied by the user when training begins.

### Inference Cache and Numerical Constants

The depth-first-search inference cache and internal numerical constants are:

```text
CACHE_MAX = 20000
EPS = 1e-12
NEG_Q_CLAMP = -1e9
```

The cache stores predicted Q-values for previously encountered Sudoku states during a solve. It is cleared whenever the model is updated so that cached predictions do not become inconsistent with the current network parameters.

### Runtime Defaults

The current program defaults are:

```text
SHOW_MOVES = False
SHOW_PLOT = False
CKPT_PREFIX = "macrocolumn_model"
```

With these settings, the solver reports numerical performance measures without printing the full move sequence or displaying graphical Sudoku boards.

---

## 5. Search and Learning Procedure

The solver combines deterministic constraint propagation, learned branching decisions, and depth-first search.

Forced cells with only one legal digit are filled deterministically. Forced nonterminal moves are not learned separately. If a forced sequence ends in success or contradiction, only the final move receives a learning target.

At a non-forced decision state, the network estimates action values for legal cell-digit assignments. Legal-digit values are reduced to cell-level drives, and the candidate cells compete through soft winner-take-all activity and divisive normalization.

The cell with the largest normalized activity is selected. Row-column order is used to resolve exact ties.

The admissible digits for the selected cell are ordered by descending estimated Q-value. Smaller digits are preferred when Q-values are tied.

During training, only the first attempted digit is sampled through the fixed ε-greedy behaviour policy. Any remaining legal digits are then tried in greedy order. This preserves depth-first-search completeness.

For evaluation:

```text
learning disabled
exploration disabled
epsilon = 0
```

Learning uses potential-shaped, decision-to-decision TD(λ) targets. Forced moves occurring after a non-forced action are propagated before the next learned decision state is evaluated.

---

## 6. Running the Solver

Run all programs from the top-level repository directory.

Launch the solver with:

```bash
python scripts/run_macrocolumn_solver.py
```

The solver first requests the puzzle directory:

```text
Puzzle directory:
```

Enter the path to the directory containing the puzzle or puzzles to be processed.

The program then requests the number of trials:

```text
Enter the number of trials (0 to test & exit):
```

Enter:

```text
0
```

to evaluate the current model without learning or exploration.

Entering a positive trial count starts or continues training.

The program then requests the learning rate. The learning rates used in the manuscript experiments are described in Sections 8 and 9 below.

For each puzzle, the solver reports:

```text
Move attempts
Maximum selected domain size
Contradiction count
Backtrack count
```

If no solution is found, the program reports:

```text
No solution found.
```

Backtrack count is the principal search-efficiency outcome analyzed in the manuscript.

---

## 7. Checkpoint Handling

A Keras checkpoint is saved after each completed training trial.

Checkpoint filenames have the form:

```text
macrocolumn_model (1).keras
macrocolumn_model (2).keras
macrocolumn_model (3).keras
...
```

When training or evaluation begins, the solver searches the current working directory for matching checkpoints and loads the highest-numbered available checkpoint.

Consequently:

* Remove or relocate existing checkpoints before generating a newly initialized trial-0 baseline.
* Retain the latest checkpoint when continuing the same training sequence.
* Do not reuse a checkpoint from a different learning-rate condition.
* Isolate the intended checkpoint when evaluating a specific training trial.
* Use separate working directories when necessary to keep experimental conditions independent.

If no matching checkpoint is present, the solver creates a newly initialized model using the fixed seed.

For the prior-experience experiment, training was conducted in consecutive 10-trial increments. These increments were successive stages of one continuing 50-trial training sequence, not five independent experiments. The learned model state was carried forward between increments.

---

## 8. Learning-Rate Stress Tests

AI Escargot and Everest were each evaluated over 50 learning trials using:

```text
0.01
0.001
0.0001
```

Each puzzle and learning-rate condition must begin from the intended newly initialized model state. A checkpoint created under one learning rate must not be loaded for another learning-rate condition.

The three trajectories for each puzzle are matched by trial number:

```text
trial 1 through trial 50
```

Trial 0 is not included in the descriptive or inferential learning-rate analyses reported in Tables 2–6.

The recorded backtrack counts are stored in:

```text
results/analysis_inputs/ai_escargot_counts.txt
results/analysis_inputs/everest_counts.txt
```

These files support:

* descriptive statistics;
* Shapiro-Wilk normality tests;
* exploratory Friedman repeated-measures comparisons;
* exploratory pairwise Wilcoxon signed-rank comparisons;
* Holm-Bonferroni corrections;
* learning-rate overlay plots.

Because successive observations within each learning trajectory are serially dependent, the Friedman and Wilcoxon results are interpreted as exploratory rather than as independent confirmatory tests.

---

## 9. Prior-Experience Train/Test Experiment

The prior-experience experiment uses:

```text
75 training puzzles
25 held-out test puzzles
```

The model is trained on:

```text
data/training_data/
```

using:

```text
learning rate = 0.0001
training trials = 50
```

The first training trial begins with the puzzles in case-insensitive filename order. This permits a user-specified initial sequence, such as easy to hard, by naming the puzzle files so that they sort in the intended order.

After each completed training trial, the puzzle list is shuffled using Python’s seeded random-number generator. Because the 50-trial experiment was conducted in consecutive 10-trial increments, the same running program session was continued between increments, preserving both the learned model state and the evolving puzzle-order sequence.

Held-out evaluations process the test puzzles in case-insensitive filename order.

The manuscript reports performance at:

```text
trial 0
trial 10
trial 20
trial 30
trial 40
trial 50
```

The trial-0 baseline is obtained from a newly initialized model before training.

The trained model checkpoints evaluated in the manuscript are:

```text
trial 10
trial 20
trial 30
trial 40
trial 50
```

### Held-Out Evaluation

Each checkpoint is evaluated on:

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

When evaluating a particular checkpoint, that checkpoint must be the model selected by the solver’s checkpoint-loading procedure.

The held-out test analysis compares each puzzle’s trial-0 backtrack count with the corresponding backtrack count produced by the trial-10, 20, 30, 40, and 50 checkpoints.

The test puzzles are not used for parameter updates during training.

---

## 10. Analysis-Input Files

The files in `results/analysis_inputs/` are headerless, whitespace-separated text files.

The first column is always an identifier and is excluded from the numerical outcome analysis.

Zeros are valid backtrack counts and must not be treated as missing observations.

### `ai_escargot_counts.txt`

```text
column 0: trial number, 1–50
column 1: backtracks at learning rate 0.01
column 2: backtracks at learning rate 0.001
column 3: backtracks at learning rate 0.0001
```

### `everest_counts.txt`

```text
column 0: trial number, 1–50
column 1: backtracks at learning rate 0.01
column 2: backtracks at learning rate 0.001
column 3: backtracks at learning rate 0.0001
```

### `training_data_counts.txt`

```text
column 0: training-puzzle identifier
column 1: trial-0 backtrack count
column 2: trial-10 backtrack count
column 3: trial-20 backtrack count
column 4: trial-30 backtrack count
column 5: trial-40 backtrack count
column 6: trial-50 backtrack count
```

### `test_data_counts.txt`

```text
column 0: held-out test-puzzle identifier
column 1: trial-0 backtrack count
column 2: trial-10 backtrack count
column 3: trial-20 backtrack count
column 4: trial-30 backtrack count
column 5: trial-40 backtrack count
column 6: trial-50 backtrack count
```

The analysis-input files were manually prepared from the solver’s reported counts. They preserve the exact values used to generate the archived statistical summaries and learning-rate figures.

---

## 11. Descriptive Statistical Analysis

Run the descriptive-statistics program with:

```bash
python scripts/descriptive_statistics.py
```

The program requests the TXT filename or full path.

It ignores column 0 and calculates the following for each remaining numerical column:

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

The population standard deviation is calculated with:

```python
ddof=0
```

The median absolute deviation is calculated with:

```python
scipy.stats.median_abs_deviation(..., scale="normal")
```

Displayed probabilities smaller than `0.0001` are reported as:

```text
< 0.0001
```

The archived descriptive results correspond to:

```text
Table 2: ai_escargot_counts.txt
Table 3: everest_counts.txt
Table 7: training_data_counts.txt
Table 9: test_data_counts.txt
```

---

## 12. Friedman and Wilcoxon Analyses

Run the combined inferential-analysis program with:

```bash
python scripts/Friedman_and_Wilcoxon_tests.py
```

The program requests the TXT filename or full file path.

Column 0 is treated as an identifier and is excluded from the statistical analysis. Columns 1 onward are treated as matched count conditions.

The program allows either:

```text
1. all count columns to be compared pairwise; or
2. Count 1 to be compared only with each later count column.
```

### Learning-Rate Analyses

For:

```text
ai_escargot_counts.txt
everest_counts.txt
```

select all pairwise comparisons.

The Friedman test compares the three matched learning-rate trajectories:

```text
0.01
0.001
0.0001
```

The three Wilcoxon comparisons are:

```text
0.01 versus 0.001
0.01 versus 0.0001
0.001 versus 0.0001
```

Two-sided Wilcoxon signed-rank tests are used because the learning-rate analysis does not assume a direction before testing.

For each comparison, zero paired differences are removed before ranking, and tied absolute differences receive average ranks. The reported two-sided statistic is:

```text
W = min(W+, W−)
```

Holm-Bonferroni correction is applied across the three pairwise comparisons separately for each puzzle.

The corresponding archived results are reported in:

```text
Table 4: Friedman comparisons
Table 5: AI Escargot Wilcoxon comparisons
Table 6: Everest Wilcoxon comparisons
```

Because successive trials within each learning trajectory are serially dependent, these Friedman and Wilcoxon results are interpreted as exploratory.

### Training- and Test-Set Analyses

For:

```text
training_data_counts.txt
test_data_counts.txt
```

select the option that compares Count 1 with each later count column.

Count 1 represents trial 0. The later columns represent:

```text
trial 10
trial 20
trial 30
trial 40
trial 50
```

The directional hypothesis is:

```text
trial-0 backtrack count > post-training backtrack count
```

Accordingly, use the one-tailed output for:

```text
first column > second column
```

and report the positive-rank statistic:

```text
W+
```

Zero paired differences are removed before ranking, and tied absolute differences receive average ranks.

Holm-Bonferroni correction is applied across the five checkpoint comparisons separately for the training-set and test-set analyses.

The corresponding archived results are reported in:

```text
Table 8: training-set comparisons
Table 10: held-out test-set comparisons
```

### P-Value Calculation

Wilcoxon p-values are calculated using:

```python
scipy.stats.wilcoxon(..., method="exact")
```

The term `method="exact"` describes the SciPy program setting used for the archived analysis. When tied absolute differences are present, the resulting p-values should not be interpreted as mathematically exact permutation probabilities.

Holm-Bonferroni correction is applied separately within each selected family of comparisons.

---

## 13. Learning-Rate Figures

Run the overlay-plot program with:

```bash
python scripts/overlay_line_plot.py
```

The program requests the TXT filename or full path.

Use:

```text
results/analysis_inputs/ai_escargot_counts.txt
```

to generate the AI Escargot learning-rate plot, and:

```text
results/analysis_inputs/everest_counts.txt
```

to generate the Everest learning-rate plot.

The program interprets the columns as:

```text
column 0: trial number
column 1: learning rate 0.01
column 2: learning rate 0.001
column 3: learning rate 0.0001
```

The resulting manuscript figures are:

```text
Figure 5: AI Escargot learning-rate comparison
Figure 6: Everest learning-rate comparison
```

---

## 14. Interpretation of the Archived Results

The learning-rate trajectories contain serially dependent observations because each trial continues learning from the model state produced by earlier trials. The Friedman and Wilcoxon learning-rate comparisons are therefore exploratory.

The training-set Wilcoxon comparisons assess whether backtrack counts decreased relative to the untrained trial-0 model on puzzles used during training.

The held-out test-set Wilcoxon comparisons assess whether learned search guidance transferred to puzzles not encountered during training.

The held-out test results did not remain statistically significant after Holm-Bonferroni correction. They should therefore be interpreted as a preliminary transfer trend rather than as conclusive evidence of generalization.

---

## 15. Numerical Reproducibility

The fixed seed is used for Python, NumPy, and TensorFlow where possible. Nevertheless, exact neural-network results may vary because of:

* Python and package versions;
* operating system;
* processor or graphics hardware;
* CPU/GPU execution;
* TensorFlow kernels;
* parallel numerical operations;
* nondeterministic low-level implementations.

For the closest reproduction:

1. Use the final pinned Python and package versions.
2. Use the supplied Sudoku files without changing the training/test partition.
3. Preserve the fixed seed.
4. Keep checkpoints from different experimental conditions separate.
5. Follow the checkpoint-loading procedure exactly.
6. Disable learning and exploration during held-out evaluation.
7. Use the archived analysis-input files when reproducing the manuscript’s statistical tables and figures.

The archived analysis-input files provide the direct numerical basis for the reported analyses. Re-running neural-network training evaluates procedural reproducibility but may not produce bit-for-bit identical trajectories on a different software or hardware environment.

---

## 16. Archival Release

The repository version associated with the manuscript should be preserved as a formal GitHub release and archived through Zenodo.

After the archival release is created, record the final version and DOI in:

```text
README.md
CITATION.cff
the manuscript Data and Code Availability statement
```

The archived release should contain the final solver, scripts, fixed puzzle datasets, analysis-input files, tables, figures, pinned software dependencies, and this reproducibility document.

