"""Top level module for PyARPES."""
# pylint: disable=unused-import

import warnings

from typing import Union

# Use both version conventions for people's sanity.
VERSION = "3.0.0"
__version__ = VERSION


def check() -> None:
    """Verifies certain aspects of the installation and provides guidance broken installations."""

    def verify_qt_tool() -> Union[str, None]:
        pip_command = "pip install pyqtgraph"
        warning = (
            "Using qt_tool, the PyARPES version of Image Tool, requires "
            "pyqtgraph and Qt5:\n\n\tYou can install with: {}".format(pip_command)
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import pyqtgraph
        except ImportError:
            return warning
        return None

    def verify_igor_pro() -> Union[str, None]:
        pip_command = "pip install https://github.com/chstan/igorpy/tarball/712a4c4#egg=igor-0.3.1"
        warning = "For Igor support, install igorpy with: {}".format(pip_command)
        warning_incompatible = (
            "PyARPES requires a patched copy of igorpy, "
            "available at \n\thttps://github.com/chstan/igorpy/tarball/712a4c4\n\n\tYou can install with: "
            "{}".format(pip_command)
        )
        try:
            import igor

            if igor.__version__ <= "0.3":
                raise ValueError("Not using patched version of igorpy.")

        except ValueError:
            return warning_incompatible
        except (ImportError, AttributeError):
            return warning

        return None

    def verify_bokeh() -> Union[str, None]:
        pip_command = "pip install bokeh>=2.0.0,<3.0.0"

        warning = "For bokeh support, install version 2.3.x\n\t with {}".format(pip_command)
        warning_incompatible = (
            "PyARPES, requires version 2 of bokeh. You can install with \n\t{}".format(pip_command)
        )

        try:
            import bokeh

            if not bokeh.__version__.startswith("2."):
                raise ValueError("Not using the specified version of Bokeh.")

        except ImportError:
            return warning
        except ValueError:
            return warning_incompatible
        return None

    checks = [
        ("Igor Pro Support", verify_igor_pro),
        ("Bokeh Support", verify_bokeh),
        ("qt_tool Support", verify_qt_tool),
    ]

    from colorama import Fore, Style

    print("Checking...")
    for check_name, check_fn in checks:
        initial_str = "[ ] {}".format(check_name)
        print(initial_str, end="", flush=True)

        failure_message = check_fn()

        print("\b" * len(initial_str) + " " * len(initial_str) + "\b" * len(initial_str), end="")

        if failure_message is None:
            print("{}[✔] {}{}".format(Fore.GREEN, check_name, Style.RESET_ALL))
        else:
            print(
                "{}[✘] {}: \n\t{}{}".format(Fore.RED, check_name, failure_message, Style.RESET_ALL)
            )
