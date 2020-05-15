"""Integration tests on the Python module"""
#########
# Setup #
#########
# Import external modules
import pytest

# Import project modules
import premierconverter as PCon
from .conftest import create_input_data_csv

#######################
# Succeeding examples #
#######################
def test_mod00_default_arguments(tmp_dir_path, input_rows_lst, df_expected_tests):
    """Default arguments"""
    # Setup
    in_filepath = tmp_dir_path / 'tmp_input.csv'
    out_filepath = tmp_dir_path / 't01_output.csv'

    # Given: Input data
    _ = create_input_data_csv(in_filepath, input_rows_lst)

    # When: Apply function
    res_filepath = PCon.convert(in_filepath, out_filepath)

    # Then: Result is as expected
    df_reload_01 = PCon.load_formatted_file(res_filepath)  # Reload resulting data from workbook
    assert PCon.formatted_dfs_are_equal(df_reload_01, df_expected_tests[4])
    print("Correct: The reloaded values are equal, up to floating point tolerance")

@pytest.mark.parametrize("nrows", [None, 2, 4, 100])
def test_mod01_nrows(tmp_dir_path, input_rows_lst, df_expected_tests, nrows):
    """Give the argument to limit the number of rows"""
    # Setup
    in_filepath = tmp_dir_path / 'tmp_input.csv'
    out_filepath = tmp_dir_path / 't02_output.csv'

    # Given: Input data
    _ = create_input_data_csv(in_filepath, input_rows_lst)

    # When: Apply function with limited rows
    res_filepath = PCon.convert(in_filepath, out_filepath, nrows=nrows)

    # Then: Result is as expected
    df_reload_01 = PCon.load_formatted_file(res_filepath)
    assert PCon.formatted_dfs_are_equal(
        df_reload_01,
        df_expected_tests[min(nrows if nrows is not None else 100, 4)]
    )
    print("Correct: The reloaded values are equal, up to floating point tolerance")

################################
# Overwriting an existing file #
################################
def test_mod10_force_overwrite(tmp_dir_path, input_rows_lst, df_expected_tests):
    """
    Check that an existing file will not be overwritten unless force
    is explicitly stated.
    """
    # Setup
    in_filepath = tmp_dir_path / 'tmp_input.csv'
    out_filepath = tmp_dir_path / 't03_output.csv'

    # Given: Input data and a file already exists in the output location
    _ = create_input_data_csv(in_filepath, input_rows_lst)

    out_file_str = 'Some basic file contents'
    _ = out_filepath.write_text(out_file_str)
    assert out_filepath.read_text() == out_file_str  # Check it has worked

    # When: Apply function with default arguments (i.e. not force overwrite)
    # Then: It throws an error and does not change the existing file
    with pytest.raises(FileExistsError) as err:
        PCon.convert(in_filepath, out_filepath)
    assert err is not None  # An error was thrown...
    assert isinstance(err.value, FileExistsError)  # ...of this specific type
    assert 'File already exists' in str(err.value)  # The error message contains is helpful...
    assert str(out_filepath.absolute()) in str(err.value)  # ...and contains the filepath
    assert out_filepath.read_text() == out_file_str  # The file contents are unchanged
    print("Correct: File was not overwritten and helpful error message was thrown")

    # When: Apply function force overwrite
    res_filepath = PCon.convert(in_filepath, out_filepath, force_overwrite=True)

    # Then: Result is as expected
    df_reload_01 = PCon.load_formatted_file(res_filepath)  # Reload resulting data from workbook
    assert PCon.formatted_dfs_are_equal(df_reload_01, df_expected_tests[4])
    print("Correct: The reloaded values are equal, up to floating point tolerance")
