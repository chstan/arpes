import pytest


def test_experimental_conditions():
    pass


def test_predicates():
    """
    Namely:

    1. is_subtracted
    2. is_spatial
    3. is_kspace
    4. is_slit_vertical
    5. is_synchrotron
    6. is_differentiated
    7. is_multi_region


    :return:
    """
    pass


def test_location_and_endstation():
    pass


def test_spectrometer():
    pass


def test_attribute_normalization():
    """
    1. t0
    2. hv
    3. manipulator/sample location values
    4. beamline settings
    ...

    A full list of these is available at the doc site under
    the description of the data model.

    This is at:
    https://arpes.netlify.com/#/spectra

    :return:
    """
    pass


def test_spectrum_type():
    # TODO DEDUPE THIS
    pass


def test_transposition():
    pass


def test_select_around_data():
    pass


def test_select_around():
    pass


def test_shape():
    pass


def test_id_and_identification_attributes():
    """
    Tests:

    1. id
    2. original_id
    3. scan_name
    4. label
    5. original_parent_scan_name

    :return:
    """
    pass


def test_dataset_attachment():
    """
    Tests:

    1. scan_row
    2. df_index
    3. df_after
    4. df_until_type
    5. referenced_scans

    :return:
    """
    pass


def test_sum_other():
    pass


def test_reference_settings():
    pass


def test_beamline_settings():
    pass


def test_spectrometer_settings():
    pass


def test_sample_position():
    """
    This is also an opportunity to test the dimension/axis
    conventions on each beamline
    :return:
    """
    pass


def test_full_coords():
    pass


def test_cut_nan_coords():
    pass


def test_nan_to_num():
    pass


def test_drop_nan():
    pass


def test_scale_coords():
    pass


def test_transform_coords():
    pass


def test_jupyter_usability_fns():
    """
    1. filter_vars
    2. var_startswith
    3. var_contains
    :return:
    """
    pass


def test_coordinatize():
    pass


def test_raveling():
    """
    Tests ravel and meshgrid
    :return:
    """
    pass


def test_to_arrays():
    pass


# Functional programming utilities
def test_iterate_axis():
    pass


def test_fp_mapping():
    """
    Tests `map_axes` and `map`
    :return:
    """
    pass


def test_enumerate_iter_coords():
    pass


def test_iter_coords():
    pass


def test_dataarray_range():
    pass


def test_stride():
    pass


# shifting
def test_shift_by():
    pass


def test_shift_coords():
    pass


# fit utilities
def test_attribute_accessors():
    """
    Tests

    1. .p
    2. .s

    :return:
    """
    pass


def test_model_evaluation():
    """

    :return:
    """
    pass


def test_param_as_dataset():
    pass


def test_parameter_names():
    pass


def test_spectrum_and_spectra_selection():
    pass


def test_degrees_of_freedom():
    # TODO decide how to de-dedupe axes that have been suffixed with
    # spectrum number for multi-region scans
    pass
