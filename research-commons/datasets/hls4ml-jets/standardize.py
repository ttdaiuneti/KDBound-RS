"""Build standardized/v1 from raw/ (see ../_tools/openml_dataset.py::standardize_v1)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "_tools"))
import openml_dataset as od  # noqa: E402

if __name__ == "__main__":
    od.standardize_v1(HERE)
