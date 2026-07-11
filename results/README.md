# Results Directory

This directory contains the analysis inputs, figures, and statistical tables associated with the experiments reported in the manuscript, **“Learning to Search with a Simulated Cortical Macrocolumn.”**

The directory is organized as follows:

```text
results/
├── README.md
├── input_analysis/
├── figures/
└── tables/
```

## `input_analysis/`

This subdirectory contains the analysis-ready input files used by the statistical and plotting programs.

These files are prepared from the backtrack counts and other numerical results reported by the solver. Where the solver reports results to the screen rather than writing them directly to a data file, the reported values are transcribed into plain-text input files for subsequent analysis.

Each file should retain the row and column structure required by the corresponding analysis program. The meaning of the columns is documented either in the file itself, in the associated analysis script, or in the reproducibility documentation.

The files in this directory represent inputs to the analysis procedures rather than newly generated experimental results.

## `figures/`

This subdirectory contains figures generated from the experimental results, including plots used to examine training performance, learning-rate effects, and held-out test-set performance.

Where applicable, figure filenames should correspond clearly to the experiment or comparison shown. The source data for each figure are stored in `input_analysis/`, and the programs used to generate the figures are stored in the repository’s analysis-script directory.

Figures included in the manuscript may have been resized or reformatted for publication, but their underlying numerical content is unchanged.

## `tables/`

This subdirectory contains tables generated from the statistical analyses.

These may include:

* descriptive statistics;
* Shapiro–Wilk normality-test results;
* Friedman test results;
* Wilcoxon signed-rank test results;
* Holm–Bonferroni corrected p-values; and
* summaries of training-set and held-out test-set performance.

The tables are derived from the files in `input_analysis/` using the corresponding analysis programs. Table numbering in filenames may follow the numbering used in the manuscript where appropriate.

## Reproducibility

The general workflow is:

```text
solver output
    ↓
analysis-ready files in input_analysis/
    ↓
statistical and plotting programs
    ↓
figures/ and tables/
```

The repository’s main `README.md`, analysis scripts, and reproducibility notes provide the commands, parameter settings, data formats, and experimental procedures needed to reproduce the reported results.
Create results directory
