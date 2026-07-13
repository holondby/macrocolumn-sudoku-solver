# Reproducibility Notes

This document describes the computational procedures and analyses associated with:

**David Yeo, “Learning to Search with a Simulated Cortical Macrocolumn.”**

The root `README.md` provides the repository overview, installation instructions, program entry points, directory structure, citation information, licensing information, and archival status. This document focuses on the exact model configuration, experimental procedures, checkpoint handling, data formats, and statistical analyses used for the manuscript.

---

## 1. Repository Resources

The main solver implementation is:

```text
src/macrocolumn_solver.py
```

It is launched from the repository root using:

```text
scripts/run_macrocolumn_solver.py
```

The Sudoku datasets are stored in:

```text
data/Inkala_data/
data/training_data/
data/test_data/
```

The numerical files used for the reported analyses are stored in:

```text
results/analysis_inputs/
```

The corresponding manuscript tables and figures are stored in:

```text
results/tables/
results/figures/
```

The formats of the archived analysis-input files are also summarized in:

```text
results/analysis_inputs/README.md
```

---

## 2. Software Environment

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

For the final archived release, `requirements.txt` should record the exact package versions used to generate the reported results. The exact Python version, operating system, and CPU or GPU execution environment should also be recorded.

Because TensorFlow and SciPy behaviour can vary across versions, the closest reproduction is expected when the archived software environment is used.

The repository programs are standard Python `.py` files. Jupyter Notebook or JupyterLab is not required.

---

## 3. Puzzle Data and Fixed Partition

Each Sudoku puzzle is stored as a plain-text 9 × 9 grid of integers. The value `0` represents a blank cell.

The solver loads all valid puzzle files from the directory specified by the user.

The learning-rate stress tests use:

```text
data/Inkala_data/
```

This directory contains the AI Escargot and Everest puzzles.

The prior-experience experiment uses:

```text
data/training_data/
data/test_data/
```

The original dataset contained 100 puzzles. Seventeen puzzles had a maximum selected domain size of 1 and therefore required no non-forced branching. These puzzles were excluded from the pool eligible for test-set sampling and were retained in the training set.

The resulting fixed partition contains:

```text
75 training puzzles
25 held-out test puzzles
```

The supplied training and test directories represent the partition used for the manuscript. The split should not be regenerated when reproducing the reported experiment.

---

## 4. Solver Configuration

The fixed random seed is:

```text
SEED = 121252
```

The seed is applied to Python, NumPy, and TensorFlow where supported.

Python’s random-number generator controls puzzle-order shuffling. NumPy controls exploratory first-action sampling and minibatch ordering. TensorFlow controls network initialization and applicable TensorFlow operations.

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

The macrocolumn contains:

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

The competition and normalization parameters are:

```text
GAIN = 1.15
INH_SIGMA = 1.5
OFFSET_DENOM = 0.01
WTA_BETA = 2.0
RING_TOPOLOGY = True
```

The minicolumn feature vectors undergo ring-distance Gaussian divisive normalization. A separate linear 729-action Q-value head is then applied to each inhibited minicolumn output.

Each minicolumn is scored by its largest estimated Q-value among the currently legal actions. The seven Q-value heads are combined using soft winner-take-all weights to produce the final 729-element action-value output.

Candidate Sudoku cells undergo a separate soft winner-take-all and divisive-normalization procedure. At this search-policy level, peer relationships are defined by shared rows, columns, or 3 × 3 boxes rather than by minicolumn ring distance.

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

The inference-cache and numerical settings are:

```text
CACHE_MAX = 20000
EPS = 1e-12
NEG_Q_CLAMP = -1e9
```

The cache stores predicted Q-values for previously encountered Sudoku states during a solve. It is cleared whenever the model is updated so that cached predictions remain consistent with the current network parameters.

### Runtime Defaults

The current program defaults are:

```text
SHOW_MOVES = False
SHOW_PLOT = False
CKPT_PREFIX = "macrocolumn_model"
```

---

## 5. Search and Learning Procedure

The solver combines deterministic constraint propagation, learned branching decisions, and depth-first search.

Forced cells with only one legal digit are filled deterministically. Forced nonterminal moves are not learned separately. If a forced sequence ends in success or contradiction, only the final move receives a learning target.

At a non-forced decision state, the network estimates action values for legal cell-digit assignments. Legal-digit values are reduced to cell-level drives, and candidate cells compete through soft winner-take-all activity and divisive normalization.

The cell with the largest normalized activity is selected. Row-column order resolves exact ties.

The admissible digits for the selected cell are ordered by descending estimated Q-value. Smaller digits are preferred when Q-values are tied.

During training, only the first attempted digit is sampled from the fixed ε-greedy behaviour policy. Any remaining admissible digits are tried in greedy order. This preserves depth-first-search completeness.

Learning uses potential-shaped, decision-to-decision TD(λ) targets. Forced moves occurring after a non-forced action are propagated before the next learned decision state is evaluated.

For evaluation:

```text
learning disabled
exploration disabled
epsilon = 0
```

---

## 6. Program Execution and Checkpoint Handling

Run the solver from the top-level repository directory with:

```bash
python scripts/run_macrocolumn_solver.py
```

The program requests:

```text
Puzzle directory
Number of trials
Learning rate, when training is requested
```

Entering `0` for the number of trials evaluates the current model with learning and exploration disabled and then exits.

Entering a positive number starts or continues training.

For each puzzle, the solver reports:

```text
Move attempts
Maximum selected domain size
Contradiction count
Backtrack count
```

Backtrack count is the primary outcome analyzed in the manuscript.

A Keras checkpoint is saved after each completed training trial. Checkpoint filenames have the form:

```text
macrocolumn_model (1).keras
macrocolumn_model (2).keras
macrocolumn_model (3).keras
...
```

When training or evaluation begins, the solver searches the current working directory for matching checkpoints and loads the checkpoint with the highest numerical suffix.

If no matching checkpoint is present, the solver creates a newly initialized model using the fixed seed.

To preserve experimental independence:

* remove or relocate existing checkpoints before generating a newly initialized trial-0 baseline;
* retain the latest checkpoint when continuing the same training sequence;
* do not reuse a checkpoint from a different learning-rate condition;
* isolate the intended checkpoint when evaluating a particular training trial;
* use separate working directories when necessary.

For the prior-experience experiment, training was conducted in consecutive 10-trial increments. These increments were successive stages of one continuing 50-trial sequence, not five independent experiments. The learned model state was carried forward between increments.

The same running program session was continued between increments, preserving both the learned model state and the evolving puzzle-order sequence.

### Puzzle Ordering

The first training trial begins with the puzzles in case-insensitive filename order. This permits a user-specified initial sequence, such as easy to hard, by naming the puzzle files so that they sort in the intended order.

After each completed training trial, the puzzle list is shuffled using Python’s seeded random-number generator.

Held-out evaluations process the test puzzles in case-insensitive filename order.

---

## 7. Learning-Rate Stress Tests

AI Escargot and Everest were each evaluated over 50 learning trials using:

```text
0.01
0.001
0.0001
```

Each puzzle and learning-rate condition begins from a newly initialized model state. A checkpoint created under one learning rate must not be loaded for another condition.

The observations are matched by trial number:

```text
trial 1 through trial 50
```

Trial 0 is excluded from the descriptive and inferential learning-rate analyses reported in Tables 2–6.

The archived backtrack counts are stored in:

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

## 8. Prior-Experience Train/Test Experiment

The model was trained on:

```text
data/training_data/
```

using:

```text
learning rate = 0.0001
training trials = 50
```

The manuscript reports performance at:

```text
trial 0
trial 10
trial 20
trial 30
trial 40
trial 50
```

The trial-0 baseline was obtained from a newly initialized model before training.

The trained checkpoints evaluated were:

```text
trial 10
trial 20
trial 30
trial 40
trial 50
```

Each model was evaluated on:

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

When evaluating a particular checkpoint, that checkpoint must be the highest-numbered matching checkpoint available to the solver.

The held-out analysis compares each puzzle’s trial-0 backtrack count with its corresponding count at trials 10, 20, 30, 40, and 50.

The test puzzles are not used for parameter updates during training.

---

## 9. Analysis-Input Files

The files in `results/analysis_inputs/` are headerless, whitespace-separated text files.

Column 0 is an identifier and is excluded from the numerical outcome analyses. Zeros are valid backtrack counts and must not be treated as missing values.

The solver originally displayed the relevant counts on the console. The values used for the manuscript were manually transcribed into the archived analysis-input files. These files are therefore the direct numerical inputs to the reported analyses, but they are not unedited console logs.

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

---

## 10. Analysis Procedures

### Descriptive Statistics

Run:

```bash
python scripts/descriptive_statistics.py
```

The program requests a TXT filename or full file path. It excludes column 0 and calculates:

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

The population standard deviation uses:

```python
ddof=0
```

The median absolute deviation uses:

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

### Friedman and Wilcoxon Analyses

Run:

```bash
python scripts/Friedman_and_Wilcoxon_tests.py
```

The program excludes column 0 and treats columns 1 onward as matched count conditions.

It permits either:

```text
1. all count columns to be compared pairwise; or
2. Count 1 to be compared only with each later count column.
```

#### Learning-Rate Analyses

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

The pairwise Wilcoxon comparisons are:

```text
0.01 versus 0.001
0.01 versus 0.0001
0.001 versus 0.0001
```

Two-sided Wilcoxon signed-rank tests are used because no directional difference was assumed before testing.

Zero paired differences are removed before ranking, and tied absolute differences receive average ranks.

The reported two-sided statistic is:

```text
W = min(W+, W−)
```

Holm-Bonferroni correction is applied across the three comparisons separately for each puzzle.

The corresponding results are:

```text
Table 4: Friedman comparisons
Table 5: AI Escargot Wilcoxon comparisons
Table 6: Everest Wilcoxon comparisons
```

#### Training- and Test-Set Analyses

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

Use the one-tailed result for:

```text
first column > second column
```

and report:

```text
W+
```

Zero paired differences are removed before ranking, and tied absolute differences receive average ranks.

Holm-Bonferroni correction is applied across the five checkpoint comparisons separately for the training-set and test-set analyses.

The corresponding results are:

```text
Table 8: training-set comparisons
Table 10: held-out test-set comparisons
```

#### P-Value Calculation

Wilcoxon p-values are calculated using:

```python
scipy.stats.wilcoxon(..., method="exact")
```

The term `method="exact"` describes the SciPy setting used for the archived analysis. When tied absolute differences are present, the resulting p-values should not be interpreted as mathematically exact permutation probabilities.

Holm-Bonferroni correction is applied separately within each selected family of comparisons.

### Learning-Rate Figures

Run:

```bash
python scripts/overlay_line_plot.py
```

Use:

```text
results/analysis_inputs/ai_escargot_counts.txt
```

to generate Figure 5, and:

```text
results/analysis_inputs/everest_counts.txt
```

to generate Figure 6.

The plotting program interprets the columns as:

```text
column 0: trial number
column 1: learning rate 0.01
column 2: learning rate 0.001
column 3: learning rate 0.0001
```

---

## 11. Interpretation of the Analyses

The learning-rate observations form serial learning trajectories because each trial continues from the model state produced by preceding trials. The Friedman and Wilcoxon learning-rate comparisons are therefore exploratory.

The training-set Wilcoxon comparisons assess whether backtrack counts decreased relative to the untrained trial-0 model on puzzles used during training.

The held-out test-set comparisons assess whether learned search guidance transferred to puzzles that were not used for parameter updates.

The held-out results did not remain statistically significant after Holm-Bonferroni correction. They should therefore be interpreted as a preliminary transfer trend rather than as conclusive evidence of generalization.

---

## 12. Numerical Reproducibility

The fixed seed is used for Python, NumPy, and TensorFlow where possible. Exact neural-network results may nevertheless vary because of:

* Python and package versions;
* operating system;
* processor or graphics hardware;
* CPU or GPU execution;
* TensorFlow kernels;
* parallel numerical operations;
* nondeterministic low-level implementations.

For the closest reproduction:

1. Use the exact archived Python and package versions.
2. Use the supplied Sudoku files without changing the training/test partition.
3. Preserve the fixed seed.
4. Keep checkpoints from different experimental conditions separate.
5. Follow the checkpoint-loading and puzzle-order procedures exactly.
6. Continue the 10-trial training increments within the same running session.
7. Disable learning and exploration during held-out evaluation.
8. Use the archived analysis-input files when reproducing the manuscript’s statistical tables and figures.

The analysis-input files provide the direct numerical basis for the reported analyses. Re-running neural-network training evaluates procedural reproducibility but may not produce bit-for-bit identical learning trajectories on a different software or hardware environment.
