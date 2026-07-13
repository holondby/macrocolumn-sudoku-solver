#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Analyze related count columns with a Friedman omnibus test and
# Wilcoxon signed-rank pairwise tests.
#
# Input file:
#     Headerless, whitespace-separated numeric data.
#     Column 0 is a row identifier and is ignored.
#     Columns 1 onward contain related count measurements.
#
# Wilcoxon comparison choices:
#     1. Compare every count column with every other count column.
#     2. Compare Count 1 only with each later count column.
#
# For each selected pair, differences are calculated as:
#
#     first column - second column
#
# The Wilcoxon output reports:
#     W  = min(W+, W-) for the two-sided test
#     W+ = positive-rank sum for first column > second column
#     W- = negative-rank sum for first column < second column
#
# Holm correction is applied separately to the selected two-sided,
# greater-than, and less-than families of Wilcoxon tests.
#
# The Friedman test always compares all count columns together and is
# unaffected by the selected Wilcoxon comparison method.

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


ALPHA = 0.05


def holm(p_values):
    """Return Holm-corrected p-values in their original order."""
    p_values = np.asarray(p_values, dtype=float)
    corrected = np.full_like(p_values, np.nan)

    valid = np.flatnonzero(
        np.isfinite(p_values)
    )

    if valid.size == 0:
        return corrected

    # Sort valid p-values from smallest to largest.
    order = valid[
        np.argsort(p_values[valid])
    ]

    # Apply Holm multipliers and enforce nondecreasing adjusted values.
    adjusted = np.maximum.accumulate(
        (
            len(order)
            - np.arange(len(order))
        )
        * p_values[order]
    )

    corrected[order] = np.minimum(
        adjusted,
        1.0,
    )

    return corrected


def wilcoxon_results(first, second):
    """Calculate W, W+, W-, and the three raw p-values."""
    differences = (
        np.asarray(first, dtype=float)
        - np.asarray(second, dtype=float)
    )

    # Remove zero differences before ranking and exact p-value calculation.
    differences = differences[
        differences != 0
    ]

    if differences.size == 0:
        return (
            0.0,
            0.0,
            0.0,
            np.nan,
            np.nan,
            np.nan,
        )

    # Rank absolute differences; tied values receive average ranks.
    ranks = rankdata(
        np.abs(differences),
        method="average",
    )

    w_plus = float(
        ranks[differences > 0].sum()
    )

    w_minus = float(
        ranks[differences < 0].sum()
    )

    w = min(
        w_plus,
        w_minus,
    )

    p_two_sided = wilcoxon(
        differences,
        alternative="two-sided",
        method="exact",
    ).pvalue

    p_greater = wilcoxon(
        differences,
        alternative="greater",
        method="exact",
    ).pvalue

    p_less = wilcoxon(
        differences,
        alternative="less",
        method="exact",
    ).pvalue

    return (
        w,
        w_plus,
        w_minus,
        p_two_sided,
        p_greater,
        p_less,
    )


def choose_pairs(columns):
    """Ask whether to test all pairs or Count 1 against later columns."""
    print()
    print(
        "Select the Wilcoxon comparison method:"
    )
    print(
        "1. Compare every count column with every other count column"
    )
    print(
        "2. Compare Count 1 only with each later count column"
    )

    while True:
        choice = input(
            "Enter 1 or 2: "
        ).strip()

        if choice == "1":
            return list(
                combinations(
                    columns,
                    2,
                )
            )

        if choice == "2":
            return [
                (
                    columns[0],
                    column,
                )
                for column in columns[1:]
            ]

        print(
            "Please enter 1 or 2."
        )


def format_p(value):
    """Format a p-value for display."""
    if pd.isna(value):
        return "NaN"

    if value < 0.0001:
        return "< 0.0001"

    return f"{value:.4f}"


def print_table(
    results,
    title,
    statistic,
    raw_p,
    holm_p,
):
    """Print one compact Wilcoxon results table."""
    table = results[
        [
            "Comparison",
            statistic,
            raw_p,
            holm_p,
        ]
    ].copy()

    table[statistic] = table[
        statistic
    ].map(
        lambda value: f"{value:.1f}"
    )

    table[raw_p] = table[
        raw_p
    ].map(
        format_p
    )

    table[holm_p] = table[
        holm_p
    ].map(
        format_p
    )

    table.columns = [
        "Comparison",
        statistic,
        "Raw p-value",
        "Holm-corrected p-value",
    ]

    print()
    print(title)
    print(
        table.to_string(
            index=False
        )
    )


def main():
    filename = input(
        "Enter the TXT filename or full file path: "
    ).strip()

    if not filename:
        print(
            "No filename was entered."
        )
        return

    try:
        data = pd.read_csv(
            filename,
            sep=r"\s+",
            header=None,
            thousands=",",
        )

        if data.shape[1] < 3:
            raise ValueError(
                "The file must contain one identifier column "
                "and at least two count columns."
            )

        # Ignore column 0 and convert all count columns to numeric values.
        counts = data.iloc[
            :,
            1:,
        ].apply(
            pd.to_numeric,
            errors="raise",
        )

    except Exception as error:
        print(
            f"Error: {error}"
        )
        return

    counts.columns = [
        f"Count {number}"
        for number in range(
            1,
            counts.shape[1] + 1,
        )
    ]

    # The Friedman omnibus test requires at least three related conditions.
    print()
    print(
        "Friedman test across all count columns"
    )

    if counts.shape[1] < 3:
        print(
            "Not performed: at least three count columns are required."
        )

    else:
        try:
            friedman_result = friedmanchisquare(
                *[
                    counts[column].to_numpy(
                        dtype=float
                    )
                    for column in counts.columns
                ]
            )

            degrees_of_freedom = (
                counts.shape[1] - 1
            )

            print(
                "Chi-square statistic: "
                f"{friedman_result.statistic:.4f}"
            )

            print(
                "Degrees of freedom: "
                f"{degrees_of_freedom}"
            )

            print(
                "Raw p-value: "
                f"{format_p(friedman_result.pvalue)}"
            )

            print(
                f"Nominally significant at alpha = {ALPHA}: "
                f"{friedman_result.pvalue < ALPHA}"
            )

        except Exception as error:
            print(
                f"Not performed: {error}"
            )

    comparison_pairs = choose_pairs(
        list(counts.columns)
    )

    rows = []

    for first_name, second_name in comparison_pairs:
        (
            w,
            w_plus,
            w_minus,
            p_two_sided,
            p_greater,
            p_less,
        ) = wilcoxon_results(
            counts[first_name],
            counts[second_name],
        )

        rows.append(
            {
                "Comparison": (
                    f"{first_name} vs. "
                    f"{second_name}"
                ),
                "W": w,
                "W+": w_plus,
                "W-": w_minus,
                "Two-sided raw p": p_two_sided,
                "Greater raw p": p_greater,
                "Less raw p": p_less,
            }
        )

    results = pd.DataFrame(
        rows
    )

    # Correct the three selected Wilcoxon test families separately.
    families = [
        (
            "Two-sided raw p",
            "Two-sided Holm p",
        ),
        (
            "Greater raw p",
            "Greater Holm p",
        ),
        (
            "Less raw p",
            "Less Holm p",
        ),
    ]

    for raw_column, corrected_column in families:
        results[corrected_column] = holm(
            results[raw_column]
        )

    print()
    print(
        "Wilcoxon comparisons in each Holm family: "
        f"{len(results)}"
    )

    print_table(
        results,
        "Two-sided tests: first and second columns differ",
        "W",
        "Two-sided raw p",
        "Two-sided Holm p",
    )

    print_table(
        results,
        "One-sided tests: first column > second column",
        "W+",
        "Greater raw p",
        "Greater Holm p",
    )

    print_table(
        results,
        "One-sided tests: first column < second column",
        "W-",
        "Less raw p",
        "Less Holm p",
    )


if __name__ == "__main__":
    main()


# In[ ]:




