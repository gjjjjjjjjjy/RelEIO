"""Where the analysis scripts read the per-run records and write their outputs.

With a records/ directory next to script/ (the public repository), records are read from records/ and figures
and tables are written to output/figures and output/tables. Without it (the authors' working tree, which keeps
the manuscript next to the code), the directories set below are used.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLIC = os.path.isdir(os.path.join(ROOT, "records"))
RECORDS = os.path.join(ROOT, "records") if PUBLIC else os.path.join(ROOT, "revision", "analysis")
FIGURES = os.path.join(ROOT, "output", "figures") if PUBLIC else os.path.join(ROOT, "paper", "figures")
TABLES = os.path.join(ROOT, "output", "tables") if PUBLIC else os.path.join(ROOT, "paper", "tables")


def rel(path):
    """Path relative to the repository root, for messages."""
    return os.path.relpath(path, ROOT)
