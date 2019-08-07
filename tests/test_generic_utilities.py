import pytest

from arpes.utilities import deep_equals, deep_update


@pytest.mark.parametrize('destination,source,expected_equal', [
    ({}, {}, True),
    ({'a': []}, {'a': []}, True),
    ({'a': [1.1]}, {'a': [1.2]}, False),
    ({'a': [{}, {'b': {'c': [5]}}]}, {'a': [{}, {'b': {'c': [5]}}]}, True),
])
def test_deep_equals(destination, source, expected_equal):
    if expected_equal:
        assert deep_equals(destination, source)
    else:
        assert not deep_equals(destination, source)


@pytest.mark.parametrize('destination,source,expected', [
    ({}, {}, {}),
    ({'a': []}, {'a': []}, {'a': []}),
    ({'a': [1.1]}, {'b': [1.2]}, {'a': [1.1], 'b': [1.2]}),
    ({'a': {'b': 5}}, {'a': {'c': 7}}, {'a': {'b': 5, 'c': 7}}),
])
def test_deep_update(destination, source, expected):
    assert deep_equals(deep_update(destination, source), expected)
