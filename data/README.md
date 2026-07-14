# Data Directory

This directory contains the Sudoku puzzle datasets used in the experiments reported in the manuscript:

**Learning to Search with a Simulated Cortical Macrocolumn**

The puzzle files support reproduction of the learning-rate and prior-experience experiments described in the manuscript.

## Directory Structure

The data are organized as follows:

```text
data/
├── README.md
├── Inkala_data/
├── training_data/
└── test_data/
```

## `Inkala_data/`

The `Inkala_data/` directory contains the difficult Sudoku puzzles used in the learning-rate comparison experiments: Inkala’s AI Escargot and Everest puzzles.

These puzzles were used to compare learning rates of 0.01, 0.001, and 0.0001 across the first 50 learning trials. Performance was assessed primarily by backtrack count.

## `training_data/`

The `training_data/` directory contains the fixed 75-puzzle training set used in the prior-experience experiment.

During training, this directory is supplied to the solver as the puzzle-data directory. The solver trains across the puzzles in the directory and shuffles their order after each completed trial using the fixed seeded random-number procedure described in:

```text
docs/reproducibility_notes.md
```

Training continues for 50 trials using a learning rate of 0.0001. The manuscript reports performance at trial 0 and at the trained checkpoints corresponding to trials 10, 20, 30, 40, and 50.

The 50-trial training sequence was conducted in consecutive 10-trial increments. These increments represent successive stages of one continuing training sequence rather than five independent experiments. The learned model state was carried forward from one increment to the next.

## `test_data/`

The `test_data/` directory contains the fixed 25-puzzle held-out test set used to evaluate whether training improves search guidance on puzzles not encountered during training.

The trial-0 baseline is obtained from a newly initialized model before training. The models saved at trials 10, 20, 30, 40, and 50 are then evaluated on the same fixed test set with:

```text
trials = 0
learning disabled
exploration disabled
epsilon = 0
```

The test puzzles are not used for parameter updates during training.

## Fixed Training/Test Partition

The original dataset contained 100 puzzles. Seventeen puzzles had a maximum selected domain size of 1 and therefore required no non-forced branching. These puzzles were excluded from the pool eligible for test-set selection and were retained in the training set.

The resulting fixed partition contains:

```text
75 training puzzles
25 held-out test puzzles
```

The puzzle files supplied in `training_data/` and `test_data/` constitute the fixed partition used for the manuscript. This partition should not be regenerated or randomly changed when reproducing the reported experiment.

## Puzzle Files

Each Sudoku puzzle is stored as a plain-text file and supplied directly to the solver.

Puzzle files are named using the form:

```text
Sudoku_nnn.txt
```

where `nnn` identifies the puzzle number.

Each file contains a 9 × 9 grid of integers. The value `0` denotes an empty Sudoku cell.

The training and test sets are implemented by placing the corresponding puzzle files in separate directories and supplying the appropriate directory to the solver for training or evaluation.

## Notes on Reproducibility

Use the supplied puzzle files without changing the fixed training/test partition.

When continuing training:

* retain the latest checkpoint from the same experimental condition;
* do not restart the model between 10-trial increments;
* do not reuse checkpoints from a different learning-rate condition; and
* keep checkpoints from different experimental conditions separate.

When evaluating a particular trained checkpoint, ensure that the intended checkpoint is the model loaded by the solver.

The data files in this directory should be used together with:

```text
src/macrocolumn_solver.py
scripts/run_macrocolumn_solver.py
docs/reproducibility_notes.md
results/analysis_inputs/
```

The numerical files in `results/analysis_inputs/` contain the manually transcribed backtrack counts used for the reported statistical analyses, tables, and figures.

To reproduce the manuscript experiments, preserve the supplied datasets, fixed random seed, checkpoint sequence, and evaluation settings.
