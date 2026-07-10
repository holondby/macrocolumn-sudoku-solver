# Data Directory

This directory contains the Sudoku puzzle datasets used in the experiments reported in the manuscript:

**Learning to Search with a Simulated Cortical Macrocolumn**

The puzzle files are provided to support reproduction of the learning-rate and prior-experience experiments described in the manuscript.

## Directory Structure

The data are organized as follows:

```text
data/
├── README.md
├── Inkala_data/
├── Training_data/
└── Test_data/
```

## Inkala_data/

The `Inkala_data/` directory contains the difficult Sudoku puzzles used in the learning-rate comparison experiments, specifically Inkala's AI Escargot and Everest puzzles.

These puzzles were used to compare learning rates of 0.01, 0.001, and 0.0001 across the first 50 learning trials. Performance was assessed primarily by backtrack count.

## Training_data/

The `Training_data/` directory contains the fixed 75-puzzle training set used in the prior-experience experiment.

During training, the directory is supplied to the solver as the puzzle-data directory. The solver trains across the puzzles in the directory and shuffles the puzzle order after each completed trial using the fixed seeded random-number procedure described in `reproducibility_notes.md`.

Training continues for 100 trials using a learning rate of 0.0001. Model checkpoints are saved at the specified evaluation trials.

## Test_data/

The `Test_data/` directory contains the fixed 25-puzzle held-out test set used to evaluate whether training improves search guidance on puzzles not encountered during training.

The test puzzles are evaluated with learning disabled and exploration disabled (`epsilon = 0`). The untrained trial-0 baseline is obtained from a newly initialized model. Saved training checkpoints are then evaluated on the same fixed test set.

The test set is not used for parameter updates during training.

## Puzzle Files

Each Sudoku puzzle is stored as a plain-text file and supplied directly to the solver.

Puzzle files are named using the form:

```text
Sudoku_nnn.txt
```

where `nnn` identifies the puzzle number.

The training and test sets are implemented by placing the corresponding puzzle files in separate directories and supplying the appropriate directory to the solver for training or evaluation.

Puzzle files use `0` to denote an empty Sudoku cell.

## Notes on Reproducibility

The training and test puzzle sets supplied in this directory are the fixed datasets used for the reported prior-experience experiment. They should not be randomly repartitioned when reproducing the manuscript results.

For practical convenience, learning runs were conducted in consecutive 10-trial increments rather than as a single uninterrupted run. These increments represent successive stages of the same learning procedure, not independent 10-trial experiments. The learned model state was carried forward from one increment to the next, so later trials reflect continued learning from earlier trials.

The data files in this directory should be used together with the source code, experiment scripts, and `reproducibility_notes.md` provided elsewhere in the repository.

To reproduce the manuscript experiments, use the supplied puzzle files and preserve the fixed training and test datasets.
