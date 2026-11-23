"""Where the analysis scripts read the per-run records and write their outputs.

The same script files are used in two layouts:
  public RelEIO release      records in  records/            figures in  output/figures   tables in  output/tables
  authors' working tree      records in  revision/analysis/  figures in  paper/figures    tables in  paper/tables
The layout is recognised by the presence of records/ next to script/.
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
