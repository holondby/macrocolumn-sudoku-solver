# ──────────────────────────────────────────────────────────────────────────────
# Friedman omnibus test and Holm-corrected Wilcoxon pairwise tests
#
# Input file:
#     Headerless numeric table.
#     Rows are matched observations.
#     Columns are related conditions, not row labels or ID numbers.
#     Columns may be separated by tabs, spaces, or mixed whitespace.
#
# Analysis:
#     The Friedman test compares all related columns together.
#     The Wilcoxon tests compare every unique pair of columns.
#     Thus, col_0 is compared with each later column, and later columns
#     are also compared with one another.
#
# Output:
#     Friedman chi-square statistic and p-value.
#     Pairwise Wilcoxon W, W+, W-, raw p, and Holm-corrected p.
# ──────────────────────────────────────────────────────────────────────────────

import re
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


# ──────────────────────────────────────────────────────────────────────────────
# Settings
# ──────────────────────────────────────────────────────────────────────────────

# Ask the user for the input file to analyze.
input_file = input("Enter the TXT filename or full file path: ").strip()

# Stop if the user presses Enter without typing a filename.
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
    Read a headerless whitespace-delimited numeric table.

    Comma-formatted integers, such as 23,085, are accepted.
    Rows with inconsistent column counts are ignored.
    """
    # Convert the filename string to a Path object.
    path = Path(filename)

    # Stop if the requested file does not exist.
    if not path.exists():
        raise FileNotFoundError(f"Could not find file: {filename}")

    # Store usable numeric rows from the file.
    rows = []

    # Store the expected number of columns after reading the first usable row.
    expected_cols = None

    # Open the file as plain text.
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            # Remove leading and trailing spaces from the line.
            line = line.strip()

            # Skip blank lines.
            if not line:
                continue

            # Extract integers, including comma-formatted integers such as 23,085.
            values = re.findall(r"-?\d{1,3}(?:,\d{3})+|-?\d+", line)

            # Skip lines that contain no usable numeric values.
            if not values:
                continue

            # Convert extracted strings to integers after removing commas.
            values = [int(v.replace(",", "")) for v in values]

            # Use the first usable row to set the expected column count.
            if expected_cols is None:
                expected_cols = len(values)

            # Keep only rows with the expected number of columns.
            if len(values) == expected_cols:
                rows.append(values)

    # Stop if no numeric data rows were found.
    if not rows:
        raise ValueError("No usable numeric data rows were found.")

    # At least two related columns are needed for any paired Wilcoxon comparison.
    if expected_cols < 2:
        raise ValueError("At least two columns are required for Wilcoxon comparisons.")

    # Assign generic column names because the input file has no header.
    return pd.DataFrame(rows, columns=[f"col_{i}" for i in range(expected_cols)])


# ──────────────────────────────────────────────────────────────────────────────
# Holm-Bonferroni correction
# ──────────────────────────────────────────────────────────────────────────────

def holm_correct(p_values):
    """
    Return Holm-corrected p-values in the original order.
    """
    # Convert the input p-values to a NumPy array.
    p_values = np.asarray(p_values, dtype=float)

    # Holm correction cannot be applied to missing p-values.
    if np.any(np.isnan(p_values)):
        raise ValueError("At least one raw p-value is NaN. Check the input data.")

    # Number of pairwise comparisons.
    m = len(p_values)

    # Sort p-values from smallest to largest.
    order = np.argsort(p_values)
    sorted_p = p_values[order]

    # Store corrected p-values in sorted order.
    corrected_sorted = np.empty(m)

    # Enforce monotonicity of the adjusted p-values.
    running_max = 0.0

    # Apply the Holm-Bonferroni adjustment.
    for i, p in enumerate(sorted_p):
        adjusted = (m - i) * p
        running_max = max(running_max, adjusted)
        corrected_sorted[i] = min(running_max, 1.0)

    # Restore corrected p-values to the original comparison order.
    corrected = np.empty(m)
    corrected[order] = corrected_sorted

    return corrected


# ──────────────────────────────────────────────────────────────────────────────
# Wilcoxon helper
# ──────────────────────────────────────────────────────────────────────────────

def wilcoxon_details(x, y):
    """
    Compute Wilcoxon signed-rank details for one paired comparison.
    """
    # Convert the paired columns to floating-point arrays.
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Compute paired differences.
    diffs = x - y

    # Remove zero differences, as required by zero_method="wilcox".
    nonzero_diffs = diffs[diffs != 0]

    # If all paired differences are zero, the Wilcoxon test is undefined.
    if len(nonzero_diffs) == 0:
        return {
            "W": 0.0,
            "W+": 0.0,
            "W-": 0.0,
            "p_raw": np.nan,
            "n_nonzero_pairs": 0
        }

    # Rank the absolute nonzero differences.
    ranks = rankdata(np.abs(nonzero_diffs), method="average")

    # W+ is the sum of ranks for positive differences.
    w_plus = np.sum(ranks[nonzero_diffs > 0])

    # W- is the sum of ranks for negative differences.
    w_minus = np.sum(ranks[nonzero_diffs < 0])

    # For a two-sided Wilcoxon test, W is the smaller signed-rank sum.
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
        # Compatibility with older SciPy versions.
        result = wilcoxon(
            x,
            y,
            zero_method="wilcox",
            alternative="two-sided",
            correction=False,
            mode="auto"
        )

    # Return the Wilcoxon statistic details for this column pair.
    return {
        "W": w_statistic,
        "W+": w_plus,
        "W-": w_minus,
        "p_raw": result.pvalue,
        "n_nonzero_pairs": len(nonzero_diffs)
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main program
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # Read the input file and stop gracefully if there is a problem.
    try:
        df = read_counts_file(input_file)
    except Exception as error:
        print(f"Error: {error}")
        return

    # Use compact numeric formatting when printing pandas tables.
    pd.set_option("display.float_format", lambda x: f"{x:.4g}")

    # Show the user a preview of the data that was read.
    print()
    print("First few rows read from file:")
    print(df.head())

    # Report the size of the usable dataset.
    print(f"\nNumber of usable rows: {len(df)}")
    print(f"Number of numeric columns: {df.shape[1]}")

    # ──────────────────────────────────────────────────────────────────────────
    # Friedman test across all related columns
    # ──────────────────────────────────────────────────────────────────────────

    print()
    print("Friedman test across related columns")
    print("Hypothesis: at least one column differs from the others")
    print(f"Alpha: {ALPHA}")
    print()

    # The Friedman test requires at least three related conditions.
    if df.shape[1] < 3:
        print("Friedman test not performed.")
        print("Reason: The Friedman test requires at least three related columns.")
    else:
        try:
            # Pass each DataFrame column as one related condition.
            friedman_result = friedmanchisquare(
                *[df[column].to_numpy(dtype=float) for column in df.columns]
            )

            # Extract the Friedman statistic, p-value, and degrees of freedom.
            friedman_statistic = friedman_result.statistic
            friedman_p = friedman_result.pvalue
            friedman_df = df.shape[1] - 1

            # Print the Friedman test results.
            print(f"Columns tested : {', '.join(df.columns)}")
            print(f"χ² statistic   : {friedman_statistic:.4g}")
            print(f"df             : {friedman_df}")
            print(f"p              : {friedman_p:.4g}")
            print(f"Significant    : {friedman_p < ALPHA}")

        except Exception as error:
            print("Friedman test not performed.")
            print(f"Reason: {error}")

    # ──────────────────────────────────────────────────────────────────────────
    # Wilcoxon tests for every unique pair of columns
    # ──────────────────────────────────────────────────────────────────────────

    # Store one result dictionary for each pairwise comparison.
    results = []

    # Get the column names in their current order.
    columns = list(df.columns)

    # Compare every unique pair: col_0 vs col_1, col_0 vs col_2, etc.
    for i in range(len(columns) - 1):
        for j in range(i + 1, len(columns)):
            col_a = columns[i]
            col_b = columns[j]

            # Compute Wilcoxon details for this pair of related columns.
            stats = wilcoxon_details(df[col_a], df[col_b])

            # Save the results for later conversion to a DataFrame.
            results.append({
                "comparison": f"{col_a} vs {col_b}",
                "W": stats["W"],
                "W+": stats["W+"],
                "W-": stats["W-"],
                "raw_p": stats["p_raw"],
                "n_nonzero_pairs": stats["n_nonzero_pairs"]
            })

    # Convert the list of result dictionaries to a DataFrame.
    results_df = pd.DataFrame(results)

    # ──────────────────────────────────────────────────────────────────────────
    # Holm-Bonferroni correction for Wilcoxon p-values
    # ──────────────────────────────────────────────────────────────────────────

    # Initialize corrected p-values and significance flags.
    results_df["holm_corrected_p"] = np.nan
    results_df["significant_after_holm"] = False

    # Identify comparisons with valid raw p-values.
    valid_p_mask = results_df["raw_p"].notna()

    # Correct only valid p-values; all-zero comparisons remain NaN.
    if valid_p_mask.any():
        results_df.loc[valid_p_mask, "holm_corrected_p"] = holm_correct(
            results_df.loc[valid_p_mask, "raw_p"]
        )

        # Mark comparisons that remain significant after Holm correction.
        results_df.loc[valid_p_mask, "significant_after_holm"] = (
            results_df.loc[valid_p_mask, "holm_corrected_p"] < ALPHA
        )

    # Print the Wilcoxon results table.
    print()
    print("Wilcoxon signed-rank tests for all pairwise column comparisons")
    print("Zero-difference pairs are removed using zero_method='wilcox'.")
    print("Two-sided p-values are reported.")
    print(f"Alpha: {ALPHA}")
    print()

    print(results_df.to_string(index=False))

    # Print explanatory notes for interpreting the output columns.
    print()
    print("Notes:")
    print("  The Friedman test is an omnibus repeated-measures test across all columns.")
    print("  The Friedman test is not directional.")
    print("  W is the smaller of W+ and W-.")
    print("  W+ is the rank sum for cases where the first named column > the second named column.")
    print("  W- is the rank sum for cases where the second named column > the first named column.")
    print("  raw_p is the uncorrected two-sided Wilcoxon p-value.")
    print("  holm_corrected_p is the Holm-Bonferroni corrected p-value.")
    print("  significant_after_holm is True when holm_corrected_p < alpha.")


# ──────────────────────────────────────────────────────────────────────────────
# Run program
# ──────────────────────────────────────────────────────────────────────────────

# Run main() only when this file is executed directly.
if __name__ == "__main__":
    main()
