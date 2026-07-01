"""
Create an overlay line plot from a whitespace-separated TXT data file.

Expected input-file format:

0    4532     4532     4532
1    40723    103      4545
2    36607    0        17861
3    7277     0        0
...
50   9917     0        0

The file should contain four columns:

Column 1: trial number
Column 2: count values for learning rate 0.01
Column 3: count values for learning rate 0.001
Column 4: count values for learning rate 0.0001

The file should not contain a header row. Columns may be separated by
spaces, tabs, or mixed whitespace.
"""

import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------
# Define column labels for the input file.
# -------------------------------------------------------------------------

# The input file has no header row, so column names are assigned manually.
# The first column is the trial number. The remaining columns correspond to
# the count values obtained under each learning rate.
learning_rates = ["0.01", "0.001", "0.0001"]


# -------------------------------------------------------------------------
# Ask the user to specify the input file.
# -------------------------------------------------------------------------

# The user may enter either a filename in the current working directory
# or a full file path.
input_filename = input("Enter the TXT filename or full file path: ").strip()

# Stop the program if the user presses Enter without typing a filename.
if not input_filename:
    print("No filename was entered.")
    raise SystemExit


# -------------------------------------------------------------------------
# Read the input data.
# -------------------------------------------------------------------------

try:
    # Read the whitespace-separated TXT file into a pandas DataFrame.
    #
    # sep=r"\s+" tells pandas to treat one or more whitespace characters
    # as the separator. This allows the file to contain spaces, tabs,
    # or mixed whitespace between columns.
    #
    # header=None tells pandas that the file does not contain column names.
    #
    # names=[...] supplies the column names that will be used in the DataFrame.
    df = pd.read_csv(
        input_filename,
        sep=r"\s+",
        header=None,
        names=["Trial Number"] + learning_rates
    )

except Exception as e:
    # If the file cannot be opened or parsed, report the error and stop.
    print(f"Error reading the file: {e}")
    raise SystemExit


# -------------------------------------------------------------------------
# Define the visual appearance of the plotted lines.
# -------------------------------------------------------------------------

# Each learning-rate curve is given a different line style so the curves
# remain distinguishable even if the plot is printed in grayscale.
line_styles = ["-", "--", ":"]

# Each learning-rate curve is also given a different color for readability.
line_colors = ["red", "blue", "green"]


# -------------------------------------------------------------------------
# Create the overlay line plot.
# -------------------------------------------------------------------------

# Create a new figure. The figsize values are width and height in inches.
plt.figure(figsize=(12, 7))

# Plot one line for each learning rate.
#
# zip(...) pairs each learning-rate label with its corresponding line style
# and color. For example, learning rate 0.01 is plotted with the first line
# style and first color.
for learning_rate, line_style, line_color in zip(learning_rates, line_styles, line_colors):

    # The x-axis is the trial number.
    # The y-axis is the count column for the current learning rate.
    plt.plot(
        df["Trial Number"],
        df[learning_rate],
        linestyle=line_style,
        color=line_color,
        linewidth=2,
        marker=None,
        label=f"Learning Rate = {learning_rate}"
    )


# -------------------------------------------------------------------------
# Add titles, axis labels, legend, and grid.
# -------------------------------------------------------------------------

# Add the main title for the plot.
plt.title("Learning Rate Counts Across Trials")

# Label the horizontal axis.
plt.xlabel("Trial Number")

# Label the vertical axis.
plt.ylabel("Count")

# Add a legend identifying which curve corresponds to each learning rate.
plt.legend(title="Learning Rate")

# Add a light grid to make the values easier to compare visually.
# alpha=0.3 makes the grid lines partially transparent.
plt.grid(True, alpha=0.3)

# Adjust spacing so labels and titles fit neatly within the figure area.
plt.tight_layout()


# -------------------------------------------------------------------------
# Save and display the plot.
# -------------------------------------------------------------------------

# Save the plot as a high-resolution PNG file in the current working directory.
# dpi=300 is suitable for manuscripts, reports, and presentations.
plt.savefig("overlay_line_plot.png", dpi=300)

# Display the plot on screen.
plt.show()
