#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Perform nonparametric comparisons among matched experimental measurements
stored in a whitespace-delimited TXT file.

Example input-file format:

1    337490    29830    163300
2    106364    23764    102948
3    142682    0        91438
4    123378    0        52839
5    358250    841      15257
...
50   191847    0        47

Input-file columns:
    Column 0: trial number
    Column 1: measurements for the first experimental condition
    Column 2: measurements for the second experimental condition
    Column 3: measurements for the third experimental condition

The trial-number column is used only to identify and report the range of
trials. It is excluded from all statistical tests.

The program first performs a Friedman test across the experimental
conditions. If the Friedman test is statistically significant, it performs
paired, two-sided Wilcoxon signed-rank tests comparing the first
experimental condition with each remaining condition. The resulting
p-values are adjusted using the Holm-Bonferroni method.

Calculated p-values below 0.0001 are displayed as "< 0.0001".
"""

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon


ALPHA = 0.05  # Statistical significance threshold
P_VALUE_DISPLAY_LIMIT = 0.0001


def format_p_value(p_value):
    """
    Format a p-value for display.

    Values below 0.0001 are displayed as "< 0.0001".
    Other values are displayed to six decimal places.
    """
    if p_value < P_VALUE_DISPLAY_LIMIT:
        return "< 0.0001"

    return f"{p_value:.6f}"


def load_table(filename):
    """
    Load a whitespace-delimited numeric TXT table.

    The complete input table must contain:
        Column 0: trial number
        Columns 1 onward: matched experimental measurements

    Each row must represent the same trial across all experimental
    conditions.

    Because the Friedman test requires at least three related conditions,
    the complete input table must contain at least four columns:
    one trial-number column and at least three experimental-data columns.
    """
    try:
        data = np.loadtxt(filename)
    except Exception as e:
        raise ValueError(f"Error reading file '{filename}': {e}")

    # Ensure that the loaded data are represented as a two-dimensional table.
    data = np.atleast_2d(data)

    # Require one trial-number column and at least three data columns.
    if data.shape[1] < 4:
        raise ValueError(
            "The input table must contain at least four columns: "
            "one trial-number column and at least three experimental columns. "
            f"Found {data.shape[1]} columns."
        )

    return data


def friedman_from_table(data):
    """
    Perform the Friedman test across all experimental-data columns.

    The supplied array must contain only the experimental measurements.
    The input file's trial-number column must already have been removed.

    Each row is treated as one matched trial, and each column is treated
    as one experimental condition.
    """
    columns = [data[:, i] for i in range(data.shape[1])]
    statistic, p_value = friedmanchisquare(*columns)
    return statistic, p_value


def holm_correction(p_values):
    """
    Apply the Holm-Bonferroni correction to a sequence of raw p-values.

    The correction controls the family-wise error rate when multiple
    post-hoc comparisons are performed.

    Return the adjusted p-values in the same order as the original
    comparisons.
    """
    p_values = np.asarray(p_values, dtype=float)
    number_of_tests = len(p_values)

    # Sort the raw p-values from smallest to largest.
    order = np.argsort(p_values)
    sorted_p_values = p_values[order]

    adjusted_sorted = np.empty(number_of_tests, dtype=float)

    # Multiply each ordered p-value by the number of tests remaining
    # at that step of the Holm procedure.
    for i in range(number_of_tests):
        adjusted_sorted[i] = (
            sorted_p_values[i] * (number_of_tests - i)
        )

    # Ensure that the adjusted p-values do not decrease as the ordered
    # raw p-values increase.
    for i in range(1, number_of_tests):
        adjusted_sorted[i] = max(
            adjusted_sorted[i],
            adjusted_sorted[i - 1]
        )

    # Restrict adjusted p-values to the valid interval from 0 to 1.
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

    # Restore the adjusted p-values to the original comparison order.
    adjusted_p_values = np.empty(number_of_tests, dtype=float)
    adjusted_p_values[order] = adjusted_sorted

    return adjusted_p_values


def posthoc_column0_vs_others(data):
    """
    Compare the first experimental condition with every other condition.

    The supplied array must contain only experimental measurements; it
    must not include the input file's trial-number column.

    Within this experimental-data array:
        Internal column 0 = input-file column 1
        Internal column 1 = input-file column 2
        Internal column 2 = input-file column 3
        and so forth

    Internal column 0 is used as the reference condition. It is compared
    separately with each remaining condition using a paired, two-sided
    Wilcoxon signed-rank test.

    The raw p-values are then adjusted together using the Holm-Bonferroni
    method.
    """
    number_of_columns = data.shape[1]

    # The first experimental-data column is the reference condition.
    reference_column = data[:, 0]

    results = []
    raw_p_values = []

    # Compare the reference condition with each remaining condition.
    for column_index in range(1, number_of_columns):
        comparison_column = data[:, column_index]

        statistic, p_value = wilcoxon(
            reference_column,
            comparison_column,
            zero_method="wilcox",
            alternative="two-sided",
            method="auto"
        )

        results.append({
            "reference_column": 0,
            "comparison_column": column_index,
            "statistic": statistic,
            "p_raw": p_value
        })

        raw_p_values.append(p_value)

    # Adjust the complete family of post-hoc p-values.
    adjusted_p_values = holm_correction(raw_p_values)

    # Add the adjusted p-value and significance decision to each result.
    for result, adjusted_p_value in zip(
        results,
        adjusted_p_values
    ):
        result["p_holm"] = adjusted_p_value
        result["significant"] = adjusted_p_value < ALPHA

    return results


def main():
    # Ask the user to identify the whitespace-delimited input TXT file.
    filename = input("Enter TXT filename: ").strip()

    if not filename:
        print("No filename was entered.")
        return

    # Load and validate the complete table.
    try:
        complete_table = load_table(filename)
    except ValueError as error:
        print(error)
        return

    # Retain input-file column 0 only for reporting the trial-number range.
    # These values are not included in any statistical test.
    trial_numbers = complete_table[:, 0]

    # Remove the trial-number column. All remaining columns contain the
    # matched experimental measurements to be analyzed.
    experimental_data = complete_table[:, 1:]

    print(f"\nLoaded {complete_table.shape[0]} rows.")
    print(
        "Trial-number range: "
        f"{trial_numbers.min():g} to {trial_numbers.max():g}"
    )
    print(
        "Number of experimental columns analyzed: "
        f"{experimental_data.shape[1]}"
    )
    print("Input-file column 0 has been excluded from all statistical tests.")

    # Test for an overall difference among all experimental conditions.
    statistic, p_value = friedman_from_table(experimental_data)

    print("\nFriedman Test Results")
    print("---------------------")
    print(f"Chi-square statistic: {statistic:.6f}")
    print(f"p-value:              {format_p_value(p_value)}")

    # Do not perform post-hoc tests when the omnibus test is not significant.
    if p_value >= ALPHA:
        print(
            f"\nAt α = {ALPHA:.2f}, there is no statistically "
            "significant overall difference among the experimental columns."
        )
        print("Therefore, no post-hoc pairwise tests are performed.")
        return

    print(
        f"\nThe overall difference is significant at α = {ALPHA:.2f}."
    )
    print(
        "Performing paired Wilcoxon signed-rank tests comparing "
        "the first experimental column with each remaining experimental "
        "column, followed by Holm correction."
    )

    # Compare the first experimental condition with each remaining condition.
    results = posthoc_column0_vs_others(experimental_data)

    print("\nPost-hoc Wilcoxon Tests")
    print("-----------------------")
    print(
        "Comparison     | W statistic |   p (raw)  |  p (Holm)  | Significant?"
    )

    for result in results:
        # Convert internal experimental-data column numbers back to their
        # corresponding column numbers in the original input file.
        reference_file_column = result["reference_column"] + 1
        comparison_file_column = result["comparison_column"] + 1

        label = (
            f"Col{reference_file_column} vs "
            f"Col{comparison_file_column}"
        )

        significance = "Yes" if result["significant"] else "No"
        raw_p_text = format_p_value(result["p_raw"])
        holm_p_text = format_p_value(result["p_holm"])

        print(
            f"{label:14s} | "
            f"{result['statistic']:11.4f} | "
            f"{raw_p_text:10s} | "
            f"{holm_p_text:10s} | "
            f"{significance}"
        )

    # Identify comparisons that remain significant after Holm correction.
    significant_results = [
        result for result in results
        if result["significant"]
    ]

    # Print a plain-language summary of the corrected comparisons.
    if not significant_results:
        print(
            "\nIn plain English: after Holm correction for multiple "
            f"comparisons at α = {ALPHA:.2f}, input-file column 1 does "
            "not differ significantly from any of the other experimental "
            "columns."
        )
    else:
        significant_columns = [
            f"input-file column {result['comparison_column'] + 1}"
            for result in significant_results
        ]

        if len(significant_columns) == 1:
            columns_text = significant_columns[0]
        else:
            columns_text = (
                ", ".join(significant_columns[:-1])
                + " and "
                + significant_columns[-1]
            )

        print(
            "\nIn plain English: after Holm correction for multiple "
            f"comparisons at α = {ALPHA:.2f}, input-file column 1 "
            f"differs significantly from {columns_text}."
        )


# Run main() only when this file is executed directly.
if __name__ == "__main__":
    main()


# In[ ]:




