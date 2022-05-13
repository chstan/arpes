"""This is a platform generic script for invoking the sphinx-build process."""

import sys
import subprocess
from dataclasses import dataclass
from pathlib import Path
import argparse
import shutil

parser = argparse.ArgumentParser(description="Runs documentation builds for PyARPES.")
parser.add_argument("--clean", action="store_true", default=False)

args = parser.parse_args()


@dataclass
class BuildStep:
    """Provides a platform-agnostic representation of a build step in a multistep build."""

    name: str = "Unnamed build step"

    @property
    def root(self) -> Path:
        """The root of the currently building project."""
        return (Path(__file__).parent / "..").absolute()

    @staticmethod
    def is_windows() -> bool:
        """Whether the platform we are building for is a variant of Windows."""
        return sys.platform == "win32"

    def __call__(self, *args, **kwargs):
        """Runs either call_windows or call_unix accordingly."""
        print(f"Running: {self.name}")
        if self.is_windows():
            self.call_windows(*args, **kwargs)
        else:
            self.call_unix(*args, **kwargs)

    def call_windows(self, *args, **kwargs):
        """Windows specific build variant."""
        raise NotImplementedError

    def call_unix(self, *args, **kwargs):
        """Unix (non-Windows) specific build variant."""
        raise NotImplementedError


@dataclass
class Make(BuildStep):
    """Runs make to build PyARPES documentation.

    This can be parameterized with a build variant, such as
    to run `make clean` or `make html`.
    """

    name: str = "Removing old build files"
    make_step: str = ""

    def call_windows(self):
        """Run make.bat which is the Windows flavored build script."""
        batch_script = str(self.root / "docs" / "make.bat")

        generated_path = (self.root / "docs" / "source" / "generated").resolve().absolute()
        print(f"Removing generated API documentation at {str(generated_path)}")
        shutil.rmtree(str(generated_path))

        subprocess.run(f"{batch_script} {self.make_step}", shell=True)

    def call_unix(self):
        """Use make and the Makefile to build documentation."""
        docs_root = str(self.root / "docs")
        subprocess.run(f"cd {docs_root} && make {self.make_step}", shell=True)


@dataclass
class MakeClean(Make):
    """Run `make clean`."""

    name: str = "Run Sphinx Build (make clean)"
    make_step: str = "clean"


@dataclass
class MakeHtml(Make):
    """Run `make html`."""

    name: str = "Run Sphinx Build (make html)"
    make_step: str = "html"


if args.clean:
    MakeClean()()

MakeHtml()()
