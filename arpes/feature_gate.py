"""Implements feature gates so that we can support optional dependencies better.

The way this works is more or less by testing if functionality is available 
at runtime by attempting an import. Sometimes, we may also perform more checks.

Then, we provide a decorator which provides helpful error messages
if a feature gate is not passed.

These gates are also used in import time in the `.all` modules in order to 
control what gets imported when a user requests `arpes.all`.
"""

import importlib
import enum
import functools
import warnings

from dataclasses import dataclass, field
from typing import List, Optional

__all__ = ["gate", "Gates", "failing_feature_gates"]


class Gates(str, enum.Enum):
    """Defines which gates we will check.

    These more or less correspond onto extra requirements groups in
    setup.py but in principle can be used for any functionality which
    requires a certain hardware or software configuration.
    """

    LegacyUI = "legacy_ui"
    ML = "ml"
    Igor = "igor"
    Qt = "qt"


@dataclass
class Gate:
    @property
    def message(self) -> str:
        raise NotImplementedError()

    def check(self) -> bool:
        raise NotImplementedError

    @staticmethod
    def can_import_module(module_name) -> bool:
        try:
            importlib.import_module(module_name)
            return True
        except ImportError:
            return False


@dataclass
class ImportModuleGate(Gate):
    module_name: str
    module_install_name: str

    _already_checked_gate: bool = False
    _gate_did_pass: bool = False

    @property
    def message(self) -> str:
        return f"pip install {self.module_install_name}"

    def check(self) -> bool:
        if not self._already_checked_gate:
            self._already_checked_gate = True
            self._gate_did_pass = self.can_import_module(self.module_name)

        return self._gate_did_pass


@dataclass
class ExtrasGate(Gate):
    name: str
    module_names: List[str] = field(default_factory=list)
    module_install_names: List[str] = field(default_factory=list)

    _already_checked_gate: bool = False
    _gate_did_pass: bool = False

    def __post_init__(self):
        assert len(self.module_install_names) == len(self.module_names)

    @property
    def message(self) -> str:
        return (
            f"pip install arpes[{self.name}] OR pip install {' '.join(self.module_install_names)}"
        )

    def check(self) -> bool:
        if not self._already_checked_gate:
            self._already_checked_gate = True
            self._gate_did_pass = all(self.can_import_module(name) for name in self.module_names)

        return self._gate_did_pass


ALL_GATES = {
    Gates.LegacyUI: [
        ExtrasGate("legacy_ui", ["bokeh"], ["bokeh"]),
    ],
    Gates.ML: [
        ExtrasGate(
            "ml", ["skimage", "sklearn", "cvxpy"], ["scikit-image", "scikit-learn", "cvxpy"]
        ),
    ],
    Gates.Qt: [
        ExtrasGate("qt", ["pyqtgraph"], ["pyqtgraph"]),
    ],
    Gates.Igor: [
        ImportModuleGate("igor", "https://github.com/chstan/igorpy/tarball/712a4c4#egg=igor"),
    ],
}

FAILED_GATE_MESSAGE = """
You need to install some packages before using this PyARPES functionality:

{messages}
"""


def failing_feature_gates(gate_name) -> Optional[List[Gate]]:
    """Determines whether a given feature should be turned on.

    This assessment is made according to whether appropriate modules
    are installed and available. If not, we provide detailed instructions in order to
    instruct the user.
    """
    failing_gates = []
    for element in ALL_GATES[gate_name]:
        if not element.check():
            failing_gates.append(element)

    if failing_gates:
        warnings.warn(
            FAILED_GATE_MESSAGE.format(
                messages=" - " + "\n - ".join([g.message for g in failing_gates])
            )
        )

    return failing_gates


def gate(gate_name: Gates):
    """Runs a feature gate to determine whether we can support optional functionality."""

    def decorate_inner(fn):
        @functools.wraps(fn)
        def wrapped_fn(*args, **kwargs):
            if failing_feature_gates(gate_name):
                raise RuntimeError(
                    "Cannot run function due to missing features. Please read instructions above."
                )

            return fn(*args, **kwargs)

        return wrapped_fn

    return decorate_inner
