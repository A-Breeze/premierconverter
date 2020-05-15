"""Unit tests on the package's utility functions"""
# pylint: disable=invalid-name, bad-continuation

#########
# Setup #
#########
# Import external modules
import pytest
import pandas as pd
import numpy as np

# Import project modules
import premierconverter as PCon

####################
# set_na_after_val #
####################
args_to_try = [
    {'row_sers': pd.Series([1, 'Total premium', 7]), 'match_val': 'Total premium'},
    {'row_sers': pd.Series(['Total premium', 'foo', 7]), 'match_val': 'Total premium'},
    {'row_sers': pd.Series([np.nan, 'Total premium', np.nan]), 'match_val': 'Total premium'},
    {'row_sers': pd.Series([np.nan, 5, 'Total premium', np.nan]), 'match_val': 'Total premium'},
]
@pytest.mark.parametrize("args", args_to_try)
def test_util00_set_na_after_val_match_occurs(args):
    """Check that the function works when the match_val exists in the Series"""
    # Given: `match_val` occurs in `row_sers`
    row_sers, match_val = args['row_sers'], args['match_val']
    # When: Apply the function
    actual_result = PCon.set_na_after_val(row_sers, match_val)
    # Then: Expect values match up before the first instance of `match_val`
    # and are np.nan after that
    index_before_first_match = row_sers.index[row_sers == match_val].min() - 1
    expected_result = pd.Series(
        row_sers.loc[:index_before_first_match].to_list() +
        [np.nan] * (row_sers.shape[0] - index_before_first_match - 1)
    )
    assert (row_sers.index == actual_result.index).all()
    assert pd.concat([
        actual_result.to_frame('actual'), expected_result.to_frame('expected')
    ], axis=1).assign(compare=lambda df: (
        df['actual'] == df['expected']) | (
        df['actual'].isna() & df['expected'].isna()
    ))['compare'].all()
    print("Correct: Example passes the test")

args_to_try = [
    {'row_sers': pd.Series([1]), 'match_val': 2},
    {'row_sers': pd.Series([]), 'match_val': ''},
    {'row_sers': pd.Series(['Total premium ', 'total premium', 't0tal premium']),
     'match_val': 'Total premium'},  # Close, but not quite
    {'row_sers': pd.Series([np.nan]), 'match_val': 1.},
]
@pytest.mark.parametrize("args", args_to_try)
def test_util01_set_na_after_val_no_match(args):
    """Check that the function works when the match_val does not exist in the Series"""
    # Given: `match_val` does *not* occur in `row_sers`
    # When: Apply the function
    # Then: Expect that the `row_sers` is returned unchanged
    assert args['row_sers'].equals(PCon.set_na_after_val(**args))
    print("Correct: Example passes the test")

################
# trim_na_cols #
################
dfs_to_try = [
    pd.DataFrame.from_dict({0: [1, np.nan]}, orient='index'),
    pd.DataFrame.from_dict({0: [np.nan]}, orient='index'),
    pd.DataFrame.from_dict({'foo': [np.nan, 'hi']}, orient='index'),
    pd.DataFrame.from_dict({0: ['test', np.nan, np.nan, -3, np.nan]}, orient='index'),
]
@pytest.mark.parametrize("df", dfs_to_try)
def test_util10_trim_na_cols_1_row_df(df):
    """Check that the function works in the simplified case of a one-row DataFrame"""
    # Given: A DataFrame with one row
    # When: Apply the function
    # Then: Columns with NaN value and only NaNs to the right are removed
    # and the rest stays the same.
    non_na_cols = df.columns[df.notna().iloc[0, :]]
    if non_na_cols.shape[0] > 0:
        assert df.loc[:, :non_na_cols[-1]].equals(PCon.trim_na_cols(df))
    else:
        assert df.loc[:, [False] * df.shape[1]].equals(PCon.trim_na_cols(df))
    print("Correct: Example passes the test")

dfs_to_try = [
    pd.DataFrame([
        [1, 2., np.nan],
        [7, np.nan, -.5]
    ]),
    pd.DataFrame([
        [np.nan],
        [np.nan, 'foo', np.nan]
    ]),
    pd.DataFrame([[np.nan] * 3, [np.nan]])
]
@pytest.mark.parametrize("df", dfs_to_try)
def test_util11_trim_na_cols_num_non_missing_const(df):
    """
    Check that the number of non-missing values stays constant on application of the function.
    Simple check to give evidence that no non-missing values are removed by the function.
    """
    # Given: A DataFrame with some missing values
    # When: Apply the function
    # Then: The number of non-missing values remains constant
    assert df.notna().sum().sum() == PCon.trim_na_cols(df).notna().sum().sum()
    print("Correct: Example passes the test")
