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
    name: str = "Unnamed build step"

    @property
    def root(self) -> Path:
        return (Path(__file__).parent / "..").absolute()

    @staticmethod
    def is_windows():
        return sys.platform == "win32"

    def __call__(self, *args, **kwargs):
        """Runs either call_windows or call_unix accordingy."""

        print(f"Running: {self.name}")
        if self.is_windows():
            self.call_windows(*args, **kwargs)
        else:
            self.call_unix(*args, **kwargs)

    def call_windows(self, *args, **kwargs):
        raise NotImplementedError

    def call_unix(self, *args, **kwargs):
        raise NotImplementedError


@dataclass
class Make(BuildStep):
    name: str = "Removing old build files"
    make_step: str = ""

    def call_windows(self):
        batch_script = str(self.root / "docs" / "make.bat")

        generated_path = (self.root / "docs" / "source" / "generated").resolve().absolute()
        print(f"Removing generated API documentation at {str(generated_path)}")
        shutil.rmtree(str(generated_path))

        subprocess.run(f"{batch_script} {self.make_step}", shell=True)

    def call_unix(self):
        docs_root = str(self.root / "docs")
        subprocess.run(f"cd {docs_root} && make {self.make_step}", shell=True)


@dataclass
class MakeClean(Make):
    name: str = "Run Sphinx Build (make clean)"
    make_step: str = "clean"


@dataclass
class MakeHtml(Make):
    name: str = "Run Sphinx Build (make html)"
    make_step: str = "html"


if args.clean:
    MakeClean()()

MakeHtml()()
