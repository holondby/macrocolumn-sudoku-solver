"""
Compute descriptive statistics and Shapiro-Wilk normality tests
for numeric columns in a user-specified TXT file.

For each numeric column, the program reports:
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
    Read a TXT file, identify numeric columns, and print descriptive
    statistics and Shapiro-Wilk normality-test results for each column.
    """

    # Ask the user to enter the TXT filename or full file path.
    file_path = input("Enter the TXT filename or full file path: ").strip()

    # Stop if no filename or path was entered.
    if not file_path:
        print("No filename was entered.")
        return

    # Read the TXT file into a pandas DataFrame.
    # header=None treats all rows as data rather than column labels.
    try:
        df = pd.read_table(file_path, header=None)
    except Exception as e:
        print(f"Error reading the file: {e}")
        return

    # Retain only columns that pandas recognizes as numeric.
    numeric_df = df.select_dtypes(include=[np.number])

    # Stop if the file does not contain any numeric columns.
    if numeric_df.empty:
        print("No numeric columns found in the TXT file.")
        return

    print("\nStatistics for numeric columns:\n")

    # Significance level for the Shapiro-Wilk normality test.
    alpha = 0.05

    # Analyze each numeric column separately.
    for col in numeric_df.columns:

        # Remove missing values before computing statistics.
        data = numeric_df[col].dropna().values

        # Compute basic descriptive statistics.
        min_val = np.min(data)
        max_val = np.max(data)
        mean_val = np.mean(data)
        median_val = np.median(data)

        # Compute the population standard deviation.
        # Use ddof=1 instead if the sample standard deviation is desired.
        std_val = np.std(data, ddof=0)

        # Compute the scaled median absolute deviation.
        # scale='normal' makes the MAD comparable to the standard deviation
        # for approximately normally distributed data.
        mad_val = median_abs_deviation(data, scale='normal')

        # Perform the Shapiro-Wilk normality test.
        # If the test cannot be performed, store NaN values.
        try:
            shapiro_stat, shapiro_p = shapiro(data)
        except Exception:
            shapiro_stat, shapiro_p = np.nan, np.nan

        # Print the results for the current column.
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
        # A nonsignificant result does not prove normality;
        # it only indicates that significant non-normality was not detected.
        if np.isnan(shapiro_p):
            print("  Interpretation: test not performed.\n")
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
# This prevents the program from running automatically if imported elsewhere.
if __name__ == "__main__":
    main()
