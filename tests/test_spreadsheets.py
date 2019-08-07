import pytest

from arpes.utilities import default_dataset


def test_multiple_spreadsheets_throws_assertion_error(sandbox_configuration):
    sandbox_configuration.with_workspace('spreadsheets')

    with pytest.raises(AssertionError) as e:
        df = default_dataset()

    assert('.xlsx' in str(e.value))
    assert(' == 1' in str(e.value))


def test_whitespace_should_be_trimmed_from_columns_without_errors(sandbox_configuration):
    sandbox_configuration.with_workspace('spreadsheets')
    df = default_dataset(match='whitespace_in_columns', write=False)

    assert(len(df) == 6)
    assert(sorted(list(df.columns)) == ['description', 'id', 'location', 'path'])


def test_loading_cleaned_spreadsheet(sandbox_configuration):
    sandbox_configuration.with_workspace('spreadsheets')

    df = default_dataset(match='already_cleaned', write=False)

    assert(sorted(list(df.columns)) == ['description', 'id', 'location', 'path'])


def test_skip_headers(sandbox_configuration):
    sandbox_configuration.with_workspace('spreadsheets')

    df = default_dataset(match='skip_headers', write=False)

    assert(sorted(list(df.columns)) == ['beta', 'description', 'id', 'location', 'path', 'phi', 'temperature',
                                        'theta'])


def test_user_is_warned_about_required_columns(sandbox_configuration):
    sandbox_configuration.with_workspace('spreadsheets')

    with pytest.warns(Warning) as w:
        with pytest.raises(ValueError) as v:
            df = default_dataset(match='user_warnings', write=False)

    assert('You must supply both a `file` and a `location` column in your '
           'spreadsheet in order to load data.' in str(w[-1].message.args[0]))
    assert('Could not safely read dataset.' in str(v.value))
    assert('Did you supply both a `file` and a `location` column in your spreadsheet?' in str(v.value))


def test_normalize_column_names(sandbox_configuration):
    sandbox_configuration.with_workspace('spreadsheets')
    df = default_dataset(match='normalize_column_names', write=False)

    assert(sorted(list(df.columns)) == ['hv', 'id', 'location', 'path', 'temperature'])


def test_normalize_and_complete_data(sandbox_configuration):
    sandbox_configuration.with_workspace('spreadsheets')
    df = default_dataset(match='normalize_and_complete', write=False)

    assert (list(df['location']) == ['ALG-MC'] * 4 + ['BL7'] * 2)
    assert (list(df['hv']) == [5.93] + [4.2] * 5)

    assert(sorted(list(df.columns)) == ['hv', 'id', 'location', 'path'])
