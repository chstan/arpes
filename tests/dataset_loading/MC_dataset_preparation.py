import pytest

@pytest.fixture


def func(x):
    return x + 1

def test_answer():
    assert func(4) == 5