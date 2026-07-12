#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ──────────────────────────────────────────────────────────────────────────────
# Friedman omnibus test and Holm-corrected Wilcoxon pairwise tests
#
# Expected input file:
#     A headerless numeric table separated by tabs, spaces, or mixed whitespace.
#
#     Column 0:
#         Trial number. This column identifies the matched observations and is
#         not included in the statistical analysis.
#
#     Columns 1 onward:
#         Related experimental conditions measured during each trial.
#
# Example:
#
#     1    337490    29830    163300
#     2    106364    23764    102948
#     3    142682        0      2254
#
# In this example:
#     Column 0 contains the trial number.
#     Columns 1–3 contain measurements for three related conditions.
#
# Analysis:
#     The Friedman test compares all related condition columns together.
#     The Wilcoxon signed-rank tests compare every unique pair of condition
#     columns. The trial-number column is excluded from all statistical tests.
#
# Output:
#     Friedman chi-square statistic, degrees of freedom, and p-value.
#     Pairwise Wilcoxon W, W+, W-, raw p-values, and Holm-corrected p-values.
# ──────────────────────────────────────────────────────────────────────────────

import re
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import friedmanchisquare, rankdata, wilcoxon


# ──────────────────────────────────────────────────────────────────────────────
# Settings
# ──────────────────────────────────────────────────────────────────────────────

# Ask the user for the input file to analyze.
input_file = input("Enter the TXT filename or full file path: ").strip()

# Stop if the user presses Enter without entering a filename.
if not input_file:
    print("No filename was entered.")
    raise SystemExit

# Significance level used for the Friedman and Wilcoxon tests.
ALPHA = 0.05


# ──────────────────────────────────────────────────────────────────────────────
# File reader
# ──────────────────────────────────────────────────────────────────────────────

def read_counts_file(filename):
    """
    Read a headerless, whitespace-delimited numeric table.

    Column 0 is treated as the trial-number identifier.
    Columns 1 onward are treated as related experimental conditions.

    Comma-formatted integers, such as 23,085, are accepted.
    Rows with inconsistent column counts are ignored.
    """

    # Convert the supplied filename to a Path object.
    path = Path(filename)

    # Stop if the requested file does not exist.
    if not path.exists():
        raise FileNotFoundError(f"Could not find file: {filename}")

    # Store usable numeric rows from the file.
    rows = []

    # Store the expected number of columns after reading the first usable row.
    expected_cols = None

    # Open the file as plain text.
    with open(path, "r", encoding="utf-8") as file:
        for line in file:

            # Remove leading and trailing whitespace.
            line = line.strip()

            # Skip blank lines.
            if not line:
                continue

            # Extract integers, including comma-formatted values such as 23,085.
            values = re.findall(
                r"-?\d{1,3}(?:,\d{3})+|-?\d+",
                line
            )

            # Skip lines that contain no usable numeric values.
            if not values:
                continue

            # Remove commas and convert the extracted values to integers.
            values = [int(value.replace(",", "")) for value in values]

            # Use the first usable row to establish the expected column count.
            if expected_cols is None:
                expected_cols = len(values)

            # Keep only rows that have the expected number of columns.
            if len(values) == expected_cols:
                rows.append(values)

    # Stop if no usable numeric rows were found.
    if not rows:
        raise ValueError("No usable numeric data rows were found.")

    # The file must contain one trial-number column and at least two
    # related condition columns for a paired Wilcoxon comparison.
    if expected_cols < 3:
        raise ValueError(
            "The input file must contain a trial-number column and "
            "at least two related condition columns."
        )

    # Name column 0 as the trial-number column.
    # Assign generic names to the condition columns because the file has no header.
    column_names = ["trial_number"] + [
        f"col_{i}" for i in range(1, expected_cols)
    ]

    return pd.DataFrame(rows, columns=column_names)


# ──────────────────────────────────────────────────────────────────────────────
# Holm-Bonferroni correction
# ──────────────────────────────────────────────────────────────────────────────

def holm_correct(p_values):
    """
    Return Holm-corrected p-values in their original comparison order.
    """

    # Convert the raw p-values to a NumPy floating-point array.
    p_values = np.asarray(p_values, dtype=float)

    # Holm correction cannot be applied to missing p-values.
    if np.any(np.isnan(p_values)):
        raise ValueError(
            "At least one raw p-value is NaN. Check the input data."
        )

    # Record the number of pairwise comparisons.
    number_of_comparisons = len(p_values)

    # Sort the raw p-values from smallest to largest.
    order = np.argsort(p_values)
    sorted_p_values = p_values[order]

    # Store corrected p-values in sorted order.
    corrected_sorted = np.empty(number_of_comparisons)

    # Track the largest adjusted value encountered to enforce monotonicity.
    running_maximum = 0.0

    # Apply the Holm-Bonferroni adjustment.
    for index, p_value in enumerate(sorted_p_values):
        adjusted_p = (number_of_comparisons - index) * p_value
        running_maximum = max(running_maximum, adjusted_p)
        corrected_sorted[index] = min(running_maximum, 1.0)

    # Restore the corrected p-values to the original comparison order.
    corrected = np.empty(number_of_comparisons)
    corrected[order] = corrected_sorted

    return corrected


# ──────────────────────────────────────────────────────────────────────────────
# Wilcoxon helper
# ──────────────────────────────────────────────────────────────────────────────

def wilcoxon_details(x, y):
    """
    Compute Wilcoxon signed-rank details for one paired comparison.

    Differences are calculated as x - y.

    Therefore:
        W+ is the rank sum for observations where x > y.
        W- is the rank sum for observations where y > x.
    """

    # Convert the paired condition columns to floating-point arrays.
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Calculate the paired differences.
    differences = x - y

    # Remove zero differences, as required by zero_method="wilcox".
    nonzero_differences = differences[differences != 0]

    # The Wilcoxon test is undefined when all paired differences are zero.
    if len(nonzero_differences) == 0:
        return {
            "W": 0.0,
            "W+": 0.0,
            "W-": 0.0,
            "p_raw": np.nan,
            "n_nonzero_pairs": 0
        }

    # Rank the absolute nonzero differences.
    # Average ranks are assigned when tied absolute differences occur.
    ranks = rankdata(
        np.abs(nonzero_differences),
        method="average"
    )

    # Calculate the positive signed-rank sum.
    w_plus = np.sum(ranks[nonzero_differences > 0])

    # Calculate the negative signed-rank sum.
    w_minus = np.sum(ranks[nonzero_differences < 0])

    # For a two-sided test, W is the smaller signed-rank sum.
    w_statistic = min(w_plus, w_minus)

    try:
        # Compute the two-sided Wilcoxon p-value using current SciPy syntax.
        result = wilcoxon(
            x,
            y,
            zero_method="wilcox",
            alternative="two-sided",
            correction=False,
            method="auto"
        )

    except TypeError:
        # Use the older argument name for compatibility with older SciPy versions.
        result = wilcoxon(
            x,
            y,
            zero_method="wilcox",
            alternative="two-sided",
            correction=False,
            mode="auto"
        )

    # Return the test details for this pair of condition columns.
    return {
        "W": w_statistic,
        "W+": w_plus,
        "W-": w_minus,
        "p_raw": result.pvalue,
        "n_nonzero_pairs": len(nonzero_differences)
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main program
# ──────────────────────────────────────────────────────────────────────────────

def main():

    # Read the input file and stop gracefully if a problem occurs.
    try:
        full_df = read_counts_file(input_file)

    except Exception as error:
        print(f"Error: {error}")
        return

    # Store the trial numbers separately.
    # They identify the matched rows but are not analyzed statistically.
    trial_numbers = full_df["trial_number"].copy()

    # Analyze only columns 1 onward.
    condition_df = full_df.drop(columns="trial_number").copy()

    # Use compact numeric formatting when printing pandas tables.
    pd.set_option(
        "display.float_format",
        lambda value: f"{value:.4g}"
    )

    # Show a preview of the data read from the file.
    print()
    print("First few rows read from file:")
    print(full_df.head())

    # Report the structure of the usable dataset.
    print()
    print(f"Number of usable trials: {len(full_df)}")
    print("Trial-number column: trial_number")
    print(
        "Condition columns analyzed: "
        f"{', '.join(condition_df.columns)}"
    )
    print(
        "Number of condition columns analyzed: "
        f"{condition_df.shape[1]}"
    )

    # Report the range of trial numbers.
    print(
        "Trial-number range: "
        f"{trial_numbers.min()} to {trial_numbers.max()}"
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Friedman test across the related condition columns
    # ──────────────────────────────────────────────────────────────────────────

    print()
    print("Friedman test across related condition columns")
    print("The trial-number column is excluded.")
    print("Hypothesis: at least one condition differs from the others.")
    print(f"Alpha: {ALPHA}")
    print()

    # The Friedman test requires at least three related condition columns.
    if condition_df.shape[1] < 3:
        print("Friedman test not performed.")
        print(
            "Reason: The Friedman test requires at least three "
            "related condition columns."
        )

    else:
        try:
            # Pass each condition column as one related condition.
            friedman_result = friedmanchisquare(
                *[
                    condition_df[column].to_numpy(dtype=float)
                    for column in condition_df.columns
                ]
            )

            # Extract the Friedman statistic and p-value.
            friedman_statistic = friedman_result.statistic
            friedman_p = friedman_result.pvalue

            # Degrees of freedom equal the number of conditions minus one.
            friedman_df = condition_df.shape[1] - 1

            # Print the Friedman test results.
            print(
                "Columns tested : "
                f"{', '.join(condition_df.columns)}"
            )
            print(f"χ² statistic   : {friedman_statistic:.4g}")
            print(f"df             : {friedman_df}")
            print(f"p              : {friedman_p:.4g}")
            print(f"Significant    : {friedman_p < ALPHA}")

        except Exception as error:
            print("Friedman test not performed.")
            print(f"Reason: {error}")

    # ──────────────────────────────────────────────────────────────────────────
    # Wilcoxon tests for every unique pair of condition columns
    # ──────────────────────────────────────────────────────────────────────────

    # Store one result dictionary for each pairwise comparison.
    results = []

    # Get the condition-column names in their original order.
    condition_columns = list(condition_df.columns)

    # Compare every unique pair of condition columns.
    for first_index in range(len(condition_columns) - 1):
        for second_index in range(
            first_index + 1,
            len(condition_columns)
        ):
            first_column = condition_columns[first_index]
            second_column = condition_columns[second_index]

            # Compute the Wilcoxon details for this paired comparison.
            statistics = wilcoxon_details(
                condition_df[first_column],
                condition_df[second_column]
            )

            # Store the results for later conversion to a DataFrame.
            results.append({
                "comparison": (
                    f"{first_column} vs {second_column}"
                ),
                "W": statistics["W"],
                "W+": statistics["W+"],
                "W-": statistics["W-"],
                "raw_p": statistics["p_raw"],
                "n_nonzero_pairs": statistics["n_nonzero_pairs"]
            })

    # Convert the pairwise results to a DataFrame.
    results_df = pd.DataFrame(results)

    # ──────────────────────────────────────────────────────────────────────────
    # Holm-Bonferroni correction for the pairwise Wilcoxon p-values
    # ──────────────────────────────────────────────────────────────────────────

    # Initialize the corrected p-value and significance columns.
    results_df["holm_corrected_p"] = np.nan
    results_df["significant_after_holm"] = False

    # Identify comparisons that produced valid raw p-values.
    valid_p_mask = results_df["raw_p"].notna()

    # Correct only valid p-values.
    # Comparisons in which every paired difference is zero remain NaN.
    if valid_p_mask.any():
        results_df.loc[
            valid_p_mask,
            "holm_corrected_p"
        ] = holm_correct(
            results_df.loc[valid_p_mask, "raw_p"]
        )

        # Identify comparisons that remain significant after correction.
        results_df.loc[
            valid_p_mask,
            "significant_after_holm"
        ] = (
            results_df.loc[
                valid_p_mask,
                "holm_corrected_p"
            ] < ALPHA
        )

    # Print the pairwise Wilcoxon results.
    print()
    print(
        "Wilcoxon signed-rank tests for all pairwise "
        "condition comparisons"
    )
    print("The trial-number column is excluded.")
    print(
        "Zero-difference pairs are removed using "
        "zero_method='wilcox'."
    )
    print("Two-sided p-values are reported.")
    print(f"Alpha: {ALPHA}")
    print()

    print(results_df.to_string(index=False))

    # Print notes explaining the output.
    print()
    print("Notes:")
    print(
        "  Column 0 contains trial numbers and is not included "
        "in any statistical test."
    )
    print(
        "  The Friedman test is an omnibus repeated-measures "
        "test across the condition columns."
    )
    print("  The Friedman test is not directional.")
    print("  W is the smaller of W+ and W-.")
    print(
        "  W+ is the rank sum for trials where the first named "
        "condition > the second named condition."
    )
    print(
        "  W- is the rank sum for trials where the second named "
        "condition > the first named condition."
    )
    print(
        "  n_nonzero_pairs is the number of paired differences "
        "included in the Wilcoxon test."
    )
    print(
        "  raw_p is the uncorrected two-sided Wilcoxon p-value."
    )
    print(
        "  holm_corrected_p is the Holm-Bonferroni corrected "
        "p-value."
    )
    print(
        "  significant_after_holm is True when the corrected "
        "p-value is less than alpha."
    )

# ──────────────────────────────────────────────────────────────────────────────
# Run the program
# ──────────────────────────────────────────────────────────────────────────────

# Run main() only when this file is executed directly.
if __name__ == "__main__":
    main()


# In[ ]:




