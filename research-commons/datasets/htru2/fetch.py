"""Download the raw table into raw/ (never overwrites)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "_tools"))
import openml_dataset as od  # noqa: E402

if __name__ == "__main__":
    od.fetch_openml(HERE, 45558)
