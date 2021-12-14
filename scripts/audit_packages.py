import json
import os

from dataclasses import dataclass
from pathlib import Path

@dataclass
class Package:
    name: str
    size_bytes: int

    @property
    def size_mb(self):
        return self.size_bytes // (1024 * 1024)

    @property
    def size_kb(self):
        return self.size_bytes // 1024

    @classmethod
    def conda_installed(cls):
        prefix = Path(os.getenv("CONDA_PREFIX"))
        meta_files = list((prefix / "conda-meta").glob("*.json"))

        packages = []
        for package_meta in meta_files:
            with open(package_meta, "r") as f:
                packages.append(Package.from_json(json.load(f)))

        return packages
    
    @classmethod 
    def from_json(cls, json_data):
        print(json_data.keys())
        raise ValueError()
        return cls(
            name=json_data["name"],
            size_bytes=json_data["size"],
        )


if __name__ == '__main__':
    packages = Package.conda_installed()
    packages_by_size = sorted(packages, key=lambda p: p.size_bytes, reverse=True)

    print(f"Total size: {sum(p.size_bytes for p in packages) // (1024 * 1024)} (MB)")
    print("\nName" + " " * 20 + "Size (kb)")
    print("-" * 33)

    for p in packages_by_size:
        print(f"{p.name}{' ' * (24 - len(p.name))}{p.size_kb}")

