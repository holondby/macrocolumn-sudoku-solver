# Results Directory

This directory contains the analysis inputs, figures, and statistical tables associated with the experiments reported in the manuscript, **“Learning to Search with a Simulated Cortical Macrocolumn.”**

The directory is organized as follows:

```text
results/
├── README.md
├── analysis_inputs/
├── figures/
└── tables/
```

## `analysis_inputs/`

This subdirectory contains the analysis-ready data files used by the statistical and plotting programs.

Where the solver reports results to the screen rather than writing them directly to a data file, the reported values are transcribed into plain-text files in the format required by the corresponding analysis program.

These files may contain backtrack counts, trial numbers, checkpoint results, learning-rate comparisons, or other numerical values used in the reported analyses. The column structure and interpretation of each file are documented in the associated analysis script or supporting documentation.

## `figures/`

This subdirectory contains figures generated from the experimental results.

The figures may include plots of:

* performance across training trials;
* comparisons among learning rates;
* training-set performance; and
* held-out test-set performance.

Figure filenames should clearly identify the experiment or comparison shown. The corresponding analysis inputs are stored in `analysis_inputs/`, and the programs used to generate the figures are stored in the repository’s analysis-script directory.

## `tables/`

This subdirectory contains tables summarizing the experimental and statistical results.

These may include:

* descriptive statistics;
* Shapiro–Wilk normality-test results;
* Friedman test results;
* Wilcoxon signed-rank test results;
* Holm–Bonferroni corrected p-values; and
* summaries of training-set and held-out test-set performance.

Where appropriate, filenames may follow the table numbering used in the manuscript. The values in these tables are derived from the files stored in `analysis_inputs/` using the corresponding analysis programs.
