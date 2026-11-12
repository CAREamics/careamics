"""Main CAREamics command-line interface."""

import argparse
from collections.abc import Callable, Sequence

from careamics.cli.cli_utils import Subcommand
from careamics.cli.create_config_care import (
    create_config_care,
    register_config_care_parser,
)
from careamics.cli.create_config_n2n import (
    create_config_n2n,
    register_config_n2n_parser,
)
from careamics.cli.create_config_n2v import (
    create_config_n2v,
    register_config_n2v_parser,
)
from careamics.cli.predict import predict, register_predict_parser
from careamics.cli.train import register_train_parser, train

SubcommandFunc = Callable[[argparse.Namespace], None]


def get_subcommand_func(subcommand: Subcommand) -> SubcommandFunc:
    """Return the function associated with a subcommand.

    Parameters
    ----------
    subcommand : {"train", "predict", "config-care", "config-n2v", "config-n2n"}
        Name of the subcommand.

    Returns
    -------
    SubcommandFunc
        Function that runs the subcommand.

    Raises
    ------
    ValueError
        If the subcommand is unknown.
    """
    if subcommand == "train":
        return train
    if subcommand == "predict":
        return predict
    if subcommand == "config-care":
        return create_config_care
    if subcommand == "config-n2v":
        return create_config_n2v
    if subcommand == "config-n2n":
        return create_config_n2n

    raise ValueError(f"Unknown subcommand: {subcommand}")


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

    register_train_parser(subparsers)
    register_predict_parser(subparsers)
    register_config_care_parser(subparsers)
    register_config_n2v_parser(subparsers)
    register_config_n2n_parser(subparsers)

    args = parser.parse_args(argv)

    subcommand_func = get_subcommand_func(args.subcommand)
    subcommand_func(args)


if __name__ == "__main__":
    main()
