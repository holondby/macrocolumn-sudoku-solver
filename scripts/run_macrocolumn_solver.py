"""
Run the macrocolumn reinforcement-learning Sudoku solver.

This wrapper allows the solver to be launched from the repository root with:

    python scripts/run_macrocolumn_solver.py

The main solver implementation is stored in:

    src/macrocolumn_solver.py
"""

from pathlib import Path
import sys


# Ensure that the repository root is on the Python import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.macrocolumn_solver import main


if __name__ == "__main__":
    main()
