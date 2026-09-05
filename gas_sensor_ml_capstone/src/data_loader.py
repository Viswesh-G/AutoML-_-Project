"""Utilities for loading and parsing the UCI gas sensor dataset."""

from pathlib import Path


def list_data_files(data_dir="data"):
    """Return files available in the data directory."""
    data_path = Path(data_dir)
    return sorted(p for p in data_path.iterdir() if p.is_file())


def load_dataset(*args, **kwargs):
    """Placeholder for the final dataset parser.

    The UCI dataset uses its own .dat file format. Implement the parser here
    after inspecting the downloaded raw files.
    """
    raise NotImplementedError("Implement the UCI .dat parser after adding the dataset.")
