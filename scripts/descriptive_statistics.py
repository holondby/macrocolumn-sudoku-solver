#!/usr/bin/env python
# coding: utf-8

# In[2]:


"""
Compute descriptive statistics and Shapiro-Wilk normality tests for the
measurement columns in a user-specified whitespace-separated TXT file.

The first column, column 0, is assumed to contain a row identifier, such as
a trial number or puzzle number. It is not included in the statistical
analysis.

Example of an appropriate input file:

0    4532     4532     4532
1    40723    103      4545
2    36607    0        17861
3    7277     0        0
4    78661    0        0

In this example:

Column 0: row identifier e.g., trial or puzzle number.
Column 1: backtrack counts for learning rate 0.01
Column 2: backtrack counts for learning rate 0.001
Column 3: backtrack counts for learning rate 0.0001

The file should not contain a header row. Columns may be separated by
spaces, tabs, or mixed whitespace.

For each numeric measurement column after column 0, the program reports:

- Minimum
- Maximum
- Mean
- Median
- Standard deviation
- Median absolute deviation
- Shapiro-Wilk statistic and p-value

Missing values are omitted before the statistics are computed.
"""

import pandas as pd
import numpy as np
from scipy.stats import shapiro, median_abs_deviation


def main():
    """
    Read a whitespace-separated TXT file and analyze each numeric
    measurement column after the identifier column.
    """

    # Ask the user to enter the TXT filename or full file path.
    file_path = input("Enter the TXT filename or full file path: ").strip()

    # Stop if no filename or path was entered.
    if not file_path:
        print("No filename was entered.")
        return

    # Read the TXT file.
    #
    # sep=r"\s+" permits spaces, tabs, or mixed whitespace between columns.
    # header=None treats every row as data and assigns column numbers
    # beginning with 0.
    try:
        df = pd.read_csv(
            file_path,
            sep=r"\s+",
            header=None
        )
    except Exception as e:
        print(f"Error reading the file: {e}")
        return

    # A valid file must contain column 0 as the row identifier and at least
    # one additional column containing measurement data.
    if df.shape[1] < 2:
        print(
            "The file must contain an identifier column and at least "
            "one measurement column."
        )
        return

    # Exclude column 0 because it contains identifiers such as trial numbers
    # or puzzle numbers rather than experimental measurements.
    measurement_df = df.iloc[:, 1:]

    # Retain only measurement columns that pandas recognizes as numeric.
    numeric_df = measurement_df.select_dtypes(include=[np.number])

    # Stop if no numeric measurement columns remain after column 0 is removed.
    if numeric_df.empty:
        print("No numeric measurement columns were found after column 0.")
        return

    print("\nStatistics for numeric measurement columns:\n")

    # Significance level for the Shapiro-Wilk normality test.
    alpha = 0.05

    # Analyze each numeric measurement column separately.
    for col in numeric_df.columns:

        # Remove missing values before computing the statistics.
        data = numeric_df[col].dropna().values

        # Skip a column if it contains no usable observations.
        if len(data) == 0:
            print(f"Column: {col}")
            print("  No non-missing numeric observations.\n")
            continue

        # Compute the descriptive statistics.
        min_val = np.min(data)
        max_val = np.max(data)
        mean_val = np.mean(data)
        median_val = np.median(data)

        # Compute the population standard deviation.
        # Use ddof=1 instead if a sample standard deviation is required.
        std_val = np.std(data, ddof=0)

        # Compute the scaled median absolute deviation.
        #
        # scale="normal" makes the MAD comparable to the standard deviation
        # when the observations are approximately normally distributed.
        mad_val = median_abs_deviation(data, scale="normal")

        # The Shapiro-Wilk test requires at least three observations.
        # Store NaN values when the test cannot be performed.
        if len(data) >= 3:
            try:
                shapiro_stat, shapiro_p = shapiro(data)
            except Exception:
                shapiro_stat, shapiro_p = np.nan, np.nan
        else:
            shapiro_stat, shapiro_p = np.nan, np.nan

        # Print the results for the current measurement column.
        # The displayed column number is the original file-column number.
        print(f"Column: {col}")
        print(f"  Min: {min_val:.4f}")
        print(f"  Max: {max_val:.4f}")
        print(f"  Mean: {mean_val:.4f}")
        print(f"  Median: {median_val:.4f}")
        print(f"  Standard Deviation: {std_val:.4f}")
        print(f"  Median Absolute Deviation: {mad_val:.4f}")
        print(f"  Shapiro-Wilk Statistic: {shapiro_stat:.4f}")
        print(f"  Shapiro-Wilk p-value: {shapiro_p:.4f}")

        # Interpret the Shapiro-Wilk p-value.
        #
        # A nonsignificant result does not prove normality. It indicates only
        # that the test did not detect a statistically significant departure
        # from normality.
        if np.isnan(shapiro_p):
            print(
                "  Interpretation: test not performed; at least three "
                "observations are required.\n"
            )
        elif shapiro_p >= alpha:
            print(
                f"  p = {shapiro_p:.4f} ≥ {alpha}; "
                "no significant departure from normality detected.\n"
            )
        else:
            print(
                f"  p = {shapiro_p:.4f} < {alpha}; "
                "significant departure from normality detected.\n"
            )


# Run main() only when this file is executed directly.
# This prevents the program from running automatically if it is imported
# into another Python program.
if __name__ == "__main__":
    main()


# In[ ]:




