"""Unified command-line entry point for PaxPhotometrica."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence

from . import __version__
from . import fit, query, simulate


Command = Callable[..., None]
COMMANDS: dict[str, tuple[Command, str]] = {
    "simulate": (simulate.main, "Generate imaging and prism calibration data."),
    "fit": (fit.main, "Fit the joint sparse calibration model."),
    "query": (query.main, "Query fitted AB zeropoints and passbands."),
}


def format_help() -> str:
    """Return top-level CLI help without importing another CLI framework."""
    command_lines = "\n".join(
        f"  {name:<10} {description}" for name, (_, description) in COMMANDS.items()
    )
    return (
        "usage: paxphot <command> [options]\n\n"
        "Roman-like sparse imaging and prism calibration prototypes.\n\n"
        f"commands:\n{command_lines}\n\n"
        "Run 'paxphot <command> --help' for command-specific options."
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch a PaxPhotometrica subcommand."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(format_help())
        return
    if arguments[0] in {"-V", "--version"}:
        print(f"paxphotometrica {__version__}")
        return

    command_name = arguments.pop(0)
    command = COMMANDS.get(command_name)
    if command is None:
        choices = ", ".join(COMMANDS)
        raise SystemExit(
            f"paxphot: unknown command {command_name!r}; choose from {choices}"
        )
    command_main, _ = command
    command_main(arguments, f"paxphot {command_name}")
