"""Access small reference files bundled with PaxPhotometrica."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def data_path(filename: str) -> Path:
    """Return the installed path of one bundled reference-data file."""
    resource = files("paxphotometrica.data").joinpath(filename)
    if not resource.is_file():
        raise FileNotFoundError(f"Missing bundled PaxPhotometrica data file: {filename}")
    # Wheels installed by pip are unpacked, so the Traversable is a filesystem
    # path. Keeping this conversion in one helper makes future resource changes
    # local rather than scattering site-package assumptions through the model.
    return Path(str(resource))

