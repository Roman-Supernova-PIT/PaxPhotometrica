"""Package metadata, resources, and CLI smoke tests."""

from __future__ import annotations

import subprocess
import sys

from paxphotometrica import __version__
from paxphotometrica.resources import data_path


def test_bundled_reference_data_exist() -> None:
    filenames = [
        "passbands.txt",
        "prism_wavelengths.txt",
        "ice_loglam_nodes.txt",
        "bosz_logflux_empca_basis.npz",
    ]
    for filename in filenames:
        assert data_path(filename).is_file()


def test_cli_exposes_only_live_workflows() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "paxphotometrica", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "simulate" in result.stdout
    assert "fit" in result.stdout
    assert "query" in result.stdout
    assert "amp-simulate" not in result.stdout
    assert "amp-fit" not in result.stdout


def test_cli_version() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "paxphotometrica", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == f"paxphotometrica {__version__}"

