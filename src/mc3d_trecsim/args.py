"""The arguments for the stream and live fitting algorithms."""
from pathlib import Path

import typed_argparse as tap


class LiveArgs(tap.TypedArgs):
    """The arguments for the live fitting algorithm.

    Args:
        configuration_file (Path): The configuration yaml file.
        create (bool): Whether to create the configuration file. Defaults to False.

    Raises:
        ValueError: Calibration file does not exist
        ValueError: Calibration file is not a file
    """
    configuration_file: Path = tap.arg(positional=True,
                                       help='The path to the configuration yaml file.')
    create: bool = tap.arg('-c', '--create', default=False,
                           help="Whether to create the configuration file.")

    def check(self) -> None:
        """Check the arguments.

        Raises:
            ValueError: Configuration file does not exist
            ValueError: Configuration file is not a file
        """
        if not self.create:
            if not self.configuration_file.exists():
                raise ValueError(f'Configuration file \'{self.configuration_file}\' does not exist.')

            if not self.configuration_file.is_file():
                raise ValueError(f'Configuration file \'{self.configuration_file}\' is not a file.')
