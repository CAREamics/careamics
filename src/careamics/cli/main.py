"""Main CAREamics command-line interface."""

import argparse
from collections.abc import Callable, Sequence
from typing import Any, Literal

from careamics.cli.create_config_care import (
    add_args_config_care,
    create_config_care,
)
from careamics.cli.create_config_n2n import (
    add_args_config_n2n,
    create_config_n2n,
)
from careamics.cli.create_config_n2v import add_args_config_n2v, create_config_n2v
from careamics.cli.predict import add_args_predict, predict
from careamics.cli.train import add_args_train, train

SubcommandFunc = Callable[[argparse.Namespace], None]
Subcommand = Literal["train", "predict", "config-care", "config-n2v", "config-n2n"]
AddArgsFunc = Callable[[argparse.ArgumentParser], None]


# closure captures the shared registration state
def get_registration_func(
    parser: argparse.ArgumentParser, func_registry: dict[Subcommand, SubcommandFunc]
):
    """Create a function that registers subcommand parsers and functions.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Main CLI parser.
    func_registry : dict[Subcommand, SubcommandFunc]
        Registry populated with each subcommand's execution function.

    Returns
    -------
    Callable
        Function used to register subcommands.
    """
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    def register_command(
        command_name: Subcommand,
        command_func: SubcommandFunc,
        add_parser_args: AddArgsFunc,
        **parser_kwargs: Any,
    ) -> None:
        """Register a subcommand parser and its command function.

        Parameters
        ----------
        command_name : Subcommand
            Name of the subcommand.
        command_func : SubcommandFunc
            Function that runs the subcommand.
        add_parser_args : AddArgsFunc
            Function that adds the subcommand arguments to its parser.
        **parser_kwargs : Any
            Additional arguments passed to `subparsers.add_parser`.
        """
        # create subparser and add arguments
        subparser = subparsers.add_parser(command_name, **parser_kwargs)
        add_parser_args(subparser)

        # register function in dict
        func_registry[command_name] = command_func

    return register_command


def main(argv: Sequence[str] | None = None) -> None:
    """Run the CAREamics command-line interface.

    Parameters
    ----------
    argv : Sequence[str] or None, default=None
        Arguments to parse. If `None`, arguments are read from `sys.argv`.
    """
    # main CLI parser
    parser = argparse.ArgumentParser(prog="careamics")

    # registry to contain functions called by subcommands
    # as subcommands are registered this will be populated
    func_registry: dict[Subcommand, SubcommandFunc] = {}

    # register subcommands
    register_command = get_registration_func(parser, func_registry)
    register_command(
        "train",
        train,
        add_args_train,
        help="Run training.",
    )
    register_command(
        "predict",
        predict,
        add_args_predict,
        help="Run prediction.",
    )
    register_command(
        "config-care",
        create_config_care,
        add_args_config_care,
        help="Create a CARE configuration.",
    )
    register_command(
        "config-n2v",
        create_config_n2v,
        add_args_config_n2v,
        help="Create an N2V configuration.",
    )
    register_command(
        "config-n2n",
        create_config_n2n,
        add_args_config_n2n,
        help="Create an N2N configuration.",
    )

    args = parser.parse_args(argv)

    subcommand_func = func_registry[args.subcommand]
    subcommand_func(args)


if __name__ == "__main__":
    main()
