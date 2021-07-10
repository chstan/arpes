import xarray as xr
import warnings
from typing import Dict
from pathlib import Path
from dataclasses import dataclass, field

import arpes.config
from arpes.io import load_data

__all__ = ["cache_loader"]


TEST_ROOT = (Path(__file__).parent).absolute()


def path_to_datasets() -> Path:
    return TEST_ROOT / "resources" / "datasets"


@dataclass
class CachingDataLoader:
    cache: Dict[str, xr.Dataset] = field(default_factory=dict)

    def load_test_scan(self, example_name, **kwargs):
        if example_name in self.cache:
            return self.cache[example_name].copy(deep=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            path_to_data = path_to_datasets() / example_name
            if not path_to_data.exists():
                raise ValueError(f"{str(path_to_data)} does not exist.")

            data = load_data(str(path_to_data.absolute()), **kwargs)
            self.cache[example_name] = data
            return data.copy(deep=True)


cache_loader = CachingDataLoader()
