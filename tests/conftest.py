"""
Fixtures which can be used across multiple tests

This is useful because:
- Can define these once in a central location not multiple times
- Can implement setup-teardown strategy for clean environment for each test

Contents:
- Setup
- Utility functions
- Test data
- Temporary locations
"""
# pylint: disable=bad-continuation, invalid-name, redefined-outer-name, dangerous-default-value

#########
# Setup #
#########
# Import built-in modules
from pathlib import Path
import shutil

# Import external modules
import pytest
import numpy as np
import pandas as pd

# Import project modules
import premierconverter as PCon

#####################
# Utility functions #
#####################
def add_one_to_index(df):
    """Add 1 to the index values of a Series of DataFrame"""
    df.index += 1
    return df

def dir_is_empty(dir_path, start_pattern="."):
    """
    Check that the directory at `dir_path` is empty,
    except maybe for some files starting with `start_pattern`
    """
    # Ensure input is a Path type
    dir_path = Path(dir_path)
    dir_files = list(dir_path.rglob(f'[!{start_pattern}]*'))
    if any(dir_files):
        raise FileExistsError(
            f"\n\tDirectory '{dir_path}' is not empty"
            "\n\tPlease delete the following files before proceeding:"
            f"\n\t{[str(dir_file) for dir_file in dir_files]}"
        )
    return True

def get_output_col_names(perils, factors):
    """Column names of the output data frame that contains `perils` and `factors`"""
    return (
        PCon.RAW_STRUCT['stem']['col_names'] +
        [per + PCon.OUTPUT_DEFAULTS['pf_sep'] + fac
         for per, fac in pd.MultiIndex.from_product(
             [perils, [PCon.RAW_STRUCT['bp_name']] + factors]
         )]
    )

def simulate_row_str(row_id, in_row_sers):
    """
    Convert an `in_row_sers` into a string that looks like an input file row.
    row_id: The index column value that you want for the row.
    """
    return(
        str(row_id) + ',"' + in_row_sers[1] + '",' +
        pd.DataFrame([in_row_sers[1:]]).to_csv(
            header=False, index=False, line_terminator="\n"
        )
    )

def generate_input_data_csv(input_rows_lst, in_filepath=None, *, force_overwrite=False):
    """
    Creates the input CSV from a list of input rows at `in_filepath`.
    If in_filepath is not specified, return the string that would have been saved.
    """
    infile_str = ''.join(
        simulate_row_str(row_id + 1, row) for row_id, row in enumerate(input_rows_lst)
    )
    if in_filepath is None:
        return infile_str
    in_filepath = Path(in_filepath)
    if in_filepath.is_file() and not force_overwrite:
        print(
            "\n\tgenerate_input_data_csv: A file already exists at `in_filepath`:"
            f"\n\t{in_filepath.absolute()}"
            "\n\tThis has *not* been overwritten. To overwrite it, re-run"
            "\n\tthis command with `force_overwrite=True`"
        )
        return None
    in_filepath.write_text(infile_str)
    print(f"File created here: {in_filepath.absolute()}")
    return None

#############
# Test data #
#############
@pytest.fixture(scope='session')  # Runs just once for the whole session
def input_rows_lst():
    """
    Get a list of typical raw rows that can be used to form a
    DataFrame of default input data.
    """
    in_row_sers_01 = pd.Series([
    'Ok', 96.95, np.nan, np.nan, 9,
    'Peril1 Base Premium', 0.0, 91.95, 91.95,
    'AnotherPrlBase Premium', 0.0, 5.17, 5.17,
        'Peril1Factor1', 0.99818, -0.17, 91.78,
        'Total Peril Premium', '[some more text]',
    ]).pipe(add_one_to_index)
    in_row_sers_02 = pd.Series([
        'Ok', 170.73, np.nan, np.nan, 11,
        'AnotherPrlBase Premium', 0.0, 101.56, 101.56,
        'AnotherPrlFactor1', 1.064887, 6.59, 108.15,
        'Peril1 Base Premium', 0.0, 100.55, 100.55,
        'AnotherPrlSomeFact', 0.648875, -37.97, 70.18,
        'Total Peril Premium', 2, 'extra text and figures',
    ]).pipe(add_one_to_index)
    in_row_sers_03 = pd.Series([
        'Ok', 161.68, np.nan, np.nan, 5,
        'Peril1NewFact', 0.999998, 0.0, 110.34,
        'Peril1Factor1', 1.2, 18.39, 110.34,
        np.nan, np.nan, np.nan, np.nan,
        'AnotherPrlBase Premium', 0, 51.34, 51.34,
        'Peril1 Base Premium', 0.0, 91.95, 91.95,
        'Total Peril Premium', np.nan,
    ]).pipe(add_one_to_index)

    # An error row
    in_row_sers_error = pd.Series([
        'Error: Some text, that indicates an error.', 0.0, np.nan, np.nan, 4,
    ]).pipe(add_one_to_index)

    # A declined row
    in_row_sers_declined = pd.Series([
        'Declined', np.nan, np.nan, np.nan, 4,
        'Some more text on a declined row', 'even, more. text', np.nan, 0, 0,
    ]).pipe(add_one_to_index)
    return [
        in_row_sers_01, in_row_sers_02, in_row_sers_error,
        in_row_sers_03, in_row_sers_declined]

@pytest.fixture(scope='session')  # Runs just once for the whole session
def df_expected_tests(input_rows_lst):
    """
    Get a dictionary of the expected outputs for the default input data
    and particular number of rows `nrows` converted
    """
    (
        in_row_sers_01, in_row_sers_02, in_row_sers_error,
        in_row_sers_03, in_row_sers_declined
    ) = input_rows_lst
    df_expected_tests = dict()

    # Output from 4 rows
    df_expected_tests[4] = pd.DataFrame(
        columns=get_output_col_names(
            perils=['AnotherPrl', 'Peril1'],
            factors=['Factor1', 'NewFact', 'SomeFact']
        ),
        data=[
            (in_row_sers_01[[1, 2, 5+4*2]].to_list() + [1.] * 3 +
            in_row_sers_01[[5+4*1, 5+4*2+2]].to_list() + [1.] * 2),
            (in_row_sers_02[[1, 2, 5+4*1, 5+4*1+2]].to_list() + [1.] +
            in_row_sers_02[[5+4*3+2, 5+4*3]].to_list() + [1.] * 3),
            in_row_sers_error[[1]].to_list() + [0.] * 9,
            (in_row_sers_03[[1, 2, 5+4*4]].to_list() + [1.] * 3 +
            in_row_sers_03[[5+4*5, 5+4*1+2, 5+2]].to_list() + [1.])
        ],
    ).pipe(add_one_to_index).rename_axis(index=PCon.ROW_ID_NAME)

    # Output including the additional 5th 'declined' row
    df_expected_tests[5] = df_expected_tests[4].append(
        pd.Series({PCon.RAW_STRUCT['stem']['col_names'][0]: in_row_sers_declined[1]}, name=5)
    ).fillna(0.)

    # Output from 2 rows
    df_expected_tests[2] = pd.DataFrame(
        columns=get_output_col_names(
            perils=['AnotherPrl', 'Peril1'],
            factors=['Factor1', 'SomeFact']
        ),
        data=[
            (in_row_sers_01[[1, 2, 5+4*2]].to_list() + [1.] * 2 +
            in_row_sers_01[[5+4*1, 5+4*2+2]].to_list() + [1.]),
            in_row_sers_02[[1, 2, 5+4*1, 5+4*1+2, 5+4*3+2, 5+4*3]].to_list() + [1.] * 2,
        ],
    ).pipe(add_one_to_index).rename_axis(index=PCon.ROW_ID_NAME)

    # Output from 2 rows but including additional factor
    df_expected_tests['2_all_facts'] = pd.DataFrame(
        columns=get_output_col_names(
            perils=['AnotherPrl', 'Peril1'],
            factors=['Factor1', 'NewFact', 'SomeFact']
        ),
        data=[
            (in_row_sers_01[[1, 2, 5+4*2]].to_list() + [1.] * 3 +
            in_row_sers_01[[5+4*1, 5+4*2+2]].to_list() + [1.] * 2),
            (in_row_sers_02[[1, 2, 5+4*1, 5+4*1+2]].to_list() + [1.] +
            in_row_sers_02[[5+4*3+2, 5+4*3]].to_list() + [1.] * 3),
        ],
    ).pipe(add_one_to_index).rename_axis(index=PCon.ROW_ID_NAME)

    return df_expected_tests

####################
# Temporary folder #
####################
@pytest.fixture(scope='function')  # Runs once per test function
def tmp_dir_path(tmp_path_factory):
    """Get a clean folder for use in a given test function"""
    # Setup
    tmp_dir_path = tmp_path_factory.mktemp("tmp_for_tests")
    # Output
    yield tmp_dir_path
    # Teardown
    shutil.rmtree(str(tmp_dir_path))
    # assert not tmp_dir_path.is_dir()  # For debugging only
    # print("Correct: Workspace restored")
