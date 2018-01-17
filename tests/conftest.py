import pytest
import arpes.config

@pytest.fixture(scope='class')
def sandbox_configuration(monkeypatch, tmpdir_factory):
    """
    Generates a sandboxed configuration of the ARPES data analysis suite.
    In practice this means modifying sys.path, a few environment variables,
    monkey patching the configuration, and moving test data into place
    """

    # DATA_PATH = '/Users/chstansbury/.../data/'
    # DATASET_CACHE_PATH = '/Users/chstansbury/.../data/cache/'
    monkeypatch.setattr(arpes.config, 'DATA_PATH', TEMP_DATASET_PATH)
    monkeypatch.setattr(arpes.config, 'DATA_PATH', TEMP_DATASET_PATH)
    monkeypatch.setattr(arpes.config, 'DATASET_CACHE_PATH', TEMP_DATASET_CACHE)
