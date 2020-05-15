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

def create_input_data_csv(in_filepath, input_rows_lst):
    """Creates the input DataFrame and saves it as a CSV at `in_filepath`"""
    df_raw_01 = pd.DataFrame(input_rows_lst).pipe(add_one_to_index)
    df_raw_01.to_csv(in_filepath, index=True, header=None)
    return df_raw_01

#############
# Test data #
#############
@pytest.fixture(scope='session')  # Runs just once for the whole session
def input_rows_lst():
    """
    Get a list of typical raw rows that can be used to form a
    DataFrame of default input data.
    """
    df_raw_row01 = pd.Series([
        'Ok', 96.95, np.nan, np.nan, 9,
        'Peril1 Base Premium', 0.0, 91.95, 91.95,
        'AnotherPrlBase Premium', 0.0, 5.17, 5.17,
        'Peril1Factor1', 0.99818, -0.17, 91.78,
        'Total Peril Premium', '[some more text]',
    ]).pipe(add_one_to_index)
    df_raw_row02 = pd.Series([
        'Ok', 170.73, np.nan, np.nan, 11,
        'AnotherPrlBase Premium', 0.0, 101.56, 101.56,
        'AnotherPrlFactor1', 1.064887, 6.59, 108.15,
        'Peril1 Base Premium', 0.0, 100.55, 100.55,
        'AnotherPrlSomeFact', 0.648875, -37.97, 70.18,
        'Total Peril Premium', 2, 'extra text and figures',
    ]).pipe(add_one_to_index)
    df_raw_row03 = pd.Series([
        'Ok', 161.68, np.nan, np.nan, 5,
        'Peril1NewFact', 0.999998, 0.0, 110.34,
        'Peril1Factor1', 1.2, 18.39, 110.34,
        np.nan, np.nan, np.nan, np.nan,
        'AnotherPrlBase Premium', 0, 51.34, 51.34,
        'Peril1 Base Premium', 0.0, 91.95, 91.95,
        'Total Peril Premium', np.nan,
    ]).pipe(add_one_to_index)
    df_raw_row_error = pd.Series([
        'Some text that indicates an error', 0.0, np.nan, np.nan, 4,
    ]).pipe(add_one_to_index)
    return [df_raw_row01, df_raw_row02, df_raw_row_error, df_raw_row03]

@pytest.fixture(scope='session')  # Runs just once for the whole session
def df_expected_tests(input_rows_lst):
    """
    Get a dictionary of the expected outputs for the default input data
    and particular number of rows `nrows` converted
    """
    df_raw_row01, df_raw_row02, df_raw_row_error, df_raw_row03 = input_rows_lst
    df_expected_tests = dict()

    # Full output
    df_expected_tests[4] = pd.DataFrame(
        columns=get_output_col_names(
            perils=['AnotherPrl', 'Peril1'],
            factors=['Factor1', 'NewFact', 'SomeFact']
        ),
        data=[
            (df_raw_row01[[1, 2, 5+4*2]].to_list() + [1.] * 3 +
            df_raw_row01[[5+4*1, 5+4*2+2]].to_list() + [1.] * 2),
            (df_raw_row02[[1, 2, 5+4*1, 5+4*1+2]].to_list() + [1.] +
            df_raw_row02[[5+4*3+2, 5+4*3]].to_list() + [1.] * 3),
            df_raw_row_error[[1]].to_list() + [0.] * 9,
            (df_raw_row03[[1, 2, 5+4*4]].to_list() + [1.] * 3 +
            df_raw_row03[[5+4*5, 5+4*1+2, 5+2]].to_list() + [1.])
        ],
    ).pipe(add_one_to_index).rename_axis(index=PCon.ROW_ID_NAME)

    # Output from 2 rows
    df_expected_tests[2] = pd.DataFrame(
        columns=get_output_col_names(
            perils=['AnotherPrl', 'Peril1'],
            factors=['Factor1', 'SomeFact']
        ),
        data=[
            (df_raw_row01[[1, 2, 5+4*2]].to_list() + [1.] * 2 +
            df_raw_row01[[5+4*1, 5+4*2+2]].to_list() + [1.]),
            df_raw_row02[[1, 2, 5+4*1, 5+4*1+2, 5+4*3+2, 5+4*3]].to_list() + [1.] * 2,
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
