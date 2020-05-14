---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" -->
# Notes for development of automated tests
Notebook for considering what automated tests the project should implement. The actual tests that are implemented can be found in the `tests/` project folder.

**Important note**: No spreadsheet data is used in the automated tests. Any "data" used in this repo is entirely dummy data, i.e. it has been randomised and all names have been masked so they can be used for training purposes. This notebook is for training purposes only.
<!-- #endregion -->

<!-- #region _cell_guid="79c7e3d0-c299-4dcb-8224-4455121ee9b0" _uuid="d629ff2d2480ee46fbb7e2d37f6b5fab8052498a" -->
<!-- This table of contents is updated *manually* -->
# Contents
1. [Setup](#Setup): Import packages, Utility functions, Configuration variables
1. [Typical workflow](#Typical-workflow)
    - [Data for tests](#Data-for-tests)
    - [Test steps](#Test-steps): Setup, Test, Teardown
1. [Integration tests](#Integration-tests):
    - [Succeeding examples](#Succeeding-examples)
    - [Overwriting an existing file](#Overwriting-an-existing-file)
    - [Exceptions raised](#Exceptions-raised)
1. [Unit tests](#Unit-tests):
    - [Utility functions](#Utility-functions)
<!-- #endregion -->

<div align="right" style="text-align: right"><a href="#Contents">Back to Contents</a></div>

# Setup

```python
# Set warning messages
import warnings
# Show all warnings in IPython
warnings.filterwarnings('always')
# Ignore specific numpy warnings (as per <https://github.com/numpy/numpy/issues/11788#issuecomment-422846396>)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
```

```python
# Import built-in modules
import sys
import platform
import os
from pathlib import Path

# Import external modules
from IPython import __version__ as IPy_version
import numpy as np
import pandas as pd
from click import __version__ as click_version

# Import project modules
from pyprojroot import here
root_dir_path = here()
# Allow modules to be imported relative to the project root directory
if not sys.path[0] == root_dir_path:
    sys.path.insert(0, str(root_dir_path))
import premierconverter as PCon

# Re-load the project module that we are working on
%load_ext autoreload
%aimport premierconverter
%autoreload 1

# Check they have loaded and the versions are as expected
assert platform.python_version_tuple() == ('3', '6', '6')
print(f"Python version:\t\t\t{sys.version}")
assert IPy_version == '7.13.0'
print(f'IPython version:\t\t{IPy_version}')
assert np.__version__ == '1.18.2'
print(f'numpy version:\t\t\t{np.__version__}')
assert pd.__version__ == '0.25.3'
print(f'pandas version:\t\t\t{pd.__version__}')
assert click_version == '7.1.1'
print(f'click version:\t\t\t{click_version}')
print(f'premierconverter version:\t{PCon.__version__}')
```

```python
# Utility functions
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
```

```python
# Configuration variables
tmp_dir_path = Path("tmp_for_tests")
assert tmp_dir_path.is_dir()
assert dir_is_empty(tmp_dir_path)
print("Correct: Location for temporary storage of test files is available and empty.")
```

<div align="right" style="text-align: right"><a href="#Contents">Back to Contents</a></div>

# Typical workflow
## Data for tests
Define some minimal data that can be used for unit tests. Consists of `raw` data for input and `expected` data to compare against results. We'll define individual rows and select which we want the input data for each test. The raw data can also be saved to a file to test the whole pipeline.

```python
# Typical raw rows
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
df_raw_row_error = pd.Series([
    'Some text that indicates an error', 0.0, np.nan, np.nan, 4,
]).pipe(add_one_to_index)

# Typical raw DataFrame input
df_raw_01 = pd.DataFrame([
    df_raw_row01, df_raw_row02, df_raw_row_error
]).pipe(add_one_to_index)
df_raw_01.head()
```

```python
# Expected result
df_expected_01 = pd.DataFrame(
    columns=[
        'Premier_Test_Status', 'Total_Premium',
        'AnotherPrl Base Premium', 'AnotherPrl Factor1', 'AnotherPrl SomeFact',
        'Peril1 Base Premium', 'Peril1 Factor1', 'Peril1 SomeFact'
    ],
    data=[
        df_raw_row01[[1, 2, 5+4*2]].to_list() + [1., 1.] + df_raw_row01[[5+4*1, 5+4*2+2]].to_list() + [1.],
        df_raw_row02[[1, 2, 5+4*1, 5+4*1+2, 5+4*3+2, 5+4*3]].to_list() + [1., 1.],
        df_raw_row_error[[1]].to_list() + [0.] * 7,
    ],
).pipe(add_one_to_index).rename_axis(index='Ref_num')
df_expected_01
```

## Test steps
### Setup
Create the input data file.

```python
in_filepath = tmp_dir_path / 't01_typical_input.csv'
df_raw_01 = pd.DataFrame([
    df_raw_row01, df_raw_row02, df_raw_row_error
]).pipe(add_one_to_index)
df_raw_01.to_csv(in_filepath, index=True, header=None)
```

### Test

```python
# Check data can be loaded
df_raw_01_from_csv = PCon.read_raw_data(in_filepath)
assert (df_raw_01_from_csv.index == df_raw_01.index).all()
assert (df_raw_01_from_csv.dtypes == df_raw_01.dtypes).all()
assert (df_raw_01_from_csv.isna() == df_raw_01.isna()).all().all()
assert (np.abs(
    df_raw_01_from_csv.select_dtypes(['int', 'float']).fillna(0) - 
    df_raw_01.select_dtypes(['int', 'float']).fillna(0)
) < 1e-10).all().all()
assert (
    df_raw_01_from_csv.select_dtypes(exclude=['int', 'float']).astype(str) == 
    df_raw_01.select_dtypes(exclude=['int', 'float']).astype(str)
).all().all()
print("Correct: The CSV that has been created can be loaded and matches")
```

```python
# Run with default arguments
out_filepath = tmp_dir_path / 't01_output_01.csv'
res_filepath = PCon.convert(in_filepath, out_filepath)
```

```python
# Run the pipeline manually to check
# Get converted DataFrame
df_formatted_01 = PCon.convert_df(df_raw_01_from_csv)

df_formatted_01.head()
```

```python
# Reload resulting data from workbook
df_reload_01 = PCon.load_formatted_file(res_filepath)

# Check it matches re-loaded results
if PCon.formatted_dfs_are_equal(df_formatted_01, df_reload_01):
    print("Correct: The reloaded values are equal, up to floating point tolerance")
```

```python
# Check it matches expected output
if PCon.formatted_dfs_are_equal(df_formatted_01, df_expected_01):
    print("Correct: The reloaded values are equal, up to floating point tolerance")
```

### Teardown
Delete the files that have been created

```python
[filepath.unlink() 
 for filepath in [in_filepath, res_filepath] 
 if filepath.is_file()]
assert dir_is_empty(tmp_dir_path)
print("Correct: Workspace restored")
```

<div align="right" style="text-align: right"><a href="#Contents">Back to Contents</a></div>

# Integration tests
## Succeeding examples

```python

```

## Overwriting an existing file

```python

```

## Exceptions raised

```python

```

<div align="right" style="text-align: right"><a href="#Contents">Back to Contents</a></div>

# Unit tests
## Utility functions
### `set_na_after_val`

```python
# Given: `match_val` occurs in `row_sers` 
args_to_try = [
    {'row_sers': pd.Series([1, 'Total premium', 7]), 'match_val': 'Total premium'},
    {'row_sers': pd.Series(['Total premium', 'foo', 7]), 'match_val': 'Total premium'},
    {'row_sers': pd.Series([np.nan, 'Total premium', np.nan]), 'match_val': 'Total premium'},
    {'row_sers': pd.Series([np.nan, 5, 'Total premium', np.nan]), 'match_val': 'Total premium'},
]

for args in args_to_try:
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
print("Correct: All examples pass the test")
```

```python
# Given: `match_val` does *not* occur in `row_sers` 
args_to_try = [
    {'row_sers': pd.Series([1]), 'match_val': 2},
    {'row_sers': pd.Series([]), 'match_val': ''},
    {'row_sers': pd.Series(['Total premium ', 'total premium', 't0tal premium']),
     'match_val': 'Total premium'},  # Close, but not quite
    {'row_sers': pd.Series([np.nan]), 'match_val': 1.},
]
# When: Apply the function
# Then: Expect that the `row_sers` is returned unchanged
for args in args_to_try:
    assert args['row_sers'].equals(PCon.set_na_after_val(**args))
print("Correct: All examples pass the test")
```

### `trim_na_cols`

```python
# Given: A DataFrame with one row
dfs_to_try = [
    pd.DataFrame.from_dict({0: [1, np.nan]}, orient='index'),
    pd.DataFrame.from_dict({0: [np.nan]}, orient='index'),
    pd.DataFrame.from_dict({'foo': [np.nan, 'hi']}, orient='index'),
    pd.DataFrame.from_dict({0: ['test', np.nan, np.nan, -3, np.nan]}, orient='index'),
]
# When: Apply the function
# Then: Columns with NaN value and only NaNs to the right are removed 
# and the rest stays the same.
for df in dfs_to_try:
    non_na_cols = df.columns[df.notna().iloc[0,:]]
    if non_na_cols.shape[0] > 0:
        assert df.loc[:,:non_na_cols[-1]].equals(PCon.trim_na_cols(df))
    else:
        assert df.loc[:,[False] * df.shape[1]].equals(PCon.trim_na_cols(df))
print("Correct: All examples pass the test")
```

```python
# Given: A DataFrame with some missing values
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
# When: Apply the function
# Then: The number of non-missing values remains constant
for df in dfs_to_try:
    assert df.notna().sum().sum() == PCon.trim_na_cols(df).notna().sum().sum()
print("Correct: All examples pass the test")
```

<div align="right" style="text-align: right"><a href="#Contents">Back to Contents</a></div>
