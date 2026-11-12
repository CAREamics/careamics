"""Main CAREamics command-line interface."""

import argparse
from collections.abc import Sequence

from careamics.cli.cli_utils import Subcommand, SubcommandFunc
from careamics.cli.create_config_care import (
    register_command_config_care,
)
from careamics.cli.create_config_n2n import (
    register_command_config_n2n,
)
from careamics.cli.create_config_n2v import (
    register_command_config_n2v,
)
from careamics.cli.predict import register_command_predict
from careamics.cli.train import register_command_train


def main(argv: Sequence[str] | None = None) -> None:
    """Run the CAREamics command-line interface.

    Parameters
    ----------
    argv : Sequence[str] or None, default=None
        Complete command-line argument sequence, including the program name. If `None`,
        `sys.argv` is used.
    """
    parser = argparse.ArgumentParser(prog="careamics")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    func_registry: dict[Subcommand, SubcommandFunc] = {}
    # command function and subparsers are registered at the same time
    register_command_config_n2v(subparsers, func_registry)
    register_command_config_n2n(subparsers, func_registry)
    register_command_config_care(subparsers, func_registry)
    register_command_train(subparsers, func_registry)
    register_command_predict(subparsers, func_registry)

    args = parser.parse_args(argv)

    subcommand_func = func_registry[args.subcommand]
    subcommand_func(args)


if __name__ == "__main__":
    main()
