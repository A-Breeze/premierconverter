"""Integration tests on the Python module"""
#########
# Setup #
#########
# Import external modules
import pytest

# Import project modules
import premierconverter as PCon
from .conftest import generate_input_data_csv, add_one_to_index

#######################
# Succeeding examples #
#######################
def test_mod00_default_arguments(tmp_dir_path, input_rows_lst, df_expected_tests):
    """Default arguments"""
    # Setup
    in_filepath = tmp_dir_path / 'tmp_input.csv'
    out_filepath = tmp_dir_path / 't01_output.csv'

    # Given: Input data
    _ = generate_input_data_csv(input_rows_lst, in_filepath)

    # When: Apply function
    res_filepath = PCon.convert(in_filepath, out_filepath)

    # Then: Result is as expected
    df_reload_01 = PCon.load_formatted_file(res_filepath)  # Reload resulting data from workbook
    assert PCon.formatted_dfs_are_equal(df_reload_01, df_expected_tests[5])
    print("Correct: The reloaded values are equal, up to floating point tolerance")

@pytest.mark.parametrize("nrows", [None, 2, 4, 100])
def test_mod01_nrows(tmp_dir_path, input_rows_lst, df_expected_tests, nrows):
    """Give the argument to limit the number of rows"""
    # Setup
    in_filepath = tmp_dir_path / 'tmp_input.csv'
    out_filepath = tmp_dir_path / 't02_output.csv'

    # Given: Input data
    _ = generate_input_data_csv(input_rows_lst, in_filepath)

    # When: Apply function with limited rows
    res_filepath = PCon.convert(in_filepath, out_filepath, nrows=nrows)

    # Then: Result is as expected
    df_reload_01 = PCon.load_formatted_file(res_filepath)
    assert PCon.formatted_dfs_are_equal(
        df_reload_01,
        df_expected_tests[min(nrows if nrows is not None else 100, 5)]
    )
    print("Correct: The reloaded values are equal, up to floating point tolerance")

@pytest.mark.parametrize("idx_ordered, expected_label", [
    ([4, 3, 2, 1, 0], 5),
    ([2] + list(range(5)), 5),
])
def test_mod02_mix_order(
    tmp_dir_path, input_rows_lst, df_expected_tests,
    idx_ordered, expected_label,
):
    """
    Check the results are as expected for various alternative ordering
    and repetitions of the input rows
    """
    # Setup
    in_filepath = tmp_dir_path / 'tmp_input.csv'
    out_filepath = tmp_dir_path / 't11_output.csv'

    # Given: Input data consisting of certain rows in a different order
    _ = generate_input_data_csv([input_rows_lst[i] for i in idx_ordered], in_filepath)

    # When: Apply function
    res_filepath = PCon.convert(in_filepath, out_filepath)

    # Then: Result is as expected
    df_reload_01 = PCon.load_formatted_file(res_filepath)
    assert PCon.formatted_dfs_are_equal(
        df_reload_01, 
        df_expected_tests[expected_label].iloc[idx_ordered, :].reset_index(
            drop=True).pipe(add_one_to_index).rename_axis(index=PCon.ROW_ID_NAME)
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
    _ = generate_input_data_csv(input_rows_lst, in_filepath)

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
    assert PCon.formatted_dfs_are_equal(df_reload_01, df_expected_tests[5])
    print("Correct: The reloaded values are equal, up to floating point tolerance")

###################
# Include factors #
###################
@pytest.mark.parametrize("nrows, include_factors, expected_label", [
    (2, ['NewFact', 'SomeFact'], "2_all_facts"),
    (2, 'NewFact', "2_all_facts"),
    (2, ['SomeFact'], 2),
])
def test_mod20_include_factors(
    tmp_dir_path, input_rows_lst, df_expected_tests,
    nrows, include_factors, expected_label,
):
    """Check that the `include_factors` argument produces the desired outcome"""
    # Setup
    in_filepath = tmp_dir_path / 'tmp_input.csv'
    out_filepath = tmp_dir_path / 't10_output.csv'

    # Given: Input data
    _ = generate_input_data_csv(input_rows_lst, in_filepath)

    # When: Apply function with limited rows and specify factors to include
    res_filepath = PCon.convert(
        in_filepath, out_filepath,
        nrows=nrows, include_factors=include_factors
    )

    # Then: Result is as expected
    df_reload_01 = PCon.load_formatted_file(res_filepath)
    assert PCon.formatted_dfs_are_equal(df_reload_01, df_expected_tests[expected_label])
    print("Correct: The reloaded values are equal, up to floating point tolerance")
