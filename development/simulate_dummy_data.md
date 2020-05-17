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
# Simulate dummy data
Code to generate some dummy data that can be used for development and tests.

**Important note**: No data files are committed to this repo. Any "data" used in this repo is entirely dummy data, i.e. it has been randomised and all names have been masked so they can be used for training purposes. This notebook is for training purposes only.
<!-- #endregion -->

<!-- #region _cell_guid="79c7e3d0-c299-4dcb-8224-4455121ee9b0" _uuid="d629ff2d2480ee46fbb7e2d37f6b5fab8052498a" -->
<!-- This table of contents is updated *manually* -->
# Contents
1. [Setup](#Setup): Import packages, Configuration variables
1. [Typical workflow](#Typical-workflow)
1. [Manual specifications](#Manual-specifications)
1. [Automated generate function](#Automated-generate-function)
1. [Generated dummy data](#Generated-dummy-data)
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
import io

# Import external modules
from IPython import __version__ as IPy_version
import numpy as np
import pandas as pd

# Import project modules
from pyprojroot import here
root_dir_path = here()
# Allow modules to be imported relative to the project root directory
if not sys.path[0] == root_dir_path:
    sys.path.insert(0, str(root_dir_path))
import proj_config

# Check they have loaded and the versions are as expected
assert platform.python_version_tuple() == ('3', '6', '6')
print(f"Python version:\t\t\t{sys.version}")
assert IPy_version == '7.13.0'
print(f'IPython version:\t\t{IPy_version}')
assert np.__version__ == '1.18.2'
print(f'numpy version:\t\t\t{np.__version__}')
assert pd.__version__ == '0.25.3'
print(f'pandas version:\t\t\t{pd.__version__}')
```

```python
# Configuration variables
raw_data_folder_path = proj_config.example_data_dir_path
assert raw_data_folder_path.is_dir()
print("Correct: All locations are available as expected")
```

<div align="right" style="text-align: right"><a href="#Contents">Back to Contents</a></div>

# Typical workflow
Define some minimal, dummy data that can be used for development and unit tests. Specifically, we define some `in_row_sers`, each of which is a Series representation of the row in the input data, i.e. it includes the *whole* row (not just the truncated version that will be read in). From this we can create:
- The input data CSV, saved at a specified location.
- The DataFrame `df_raw` that we expect will be the result of loading the input CSV (i.e. after truncation).
- The expected DataFrame result of the conversion. We specify this in an ad hoc way, and use it to check the actual result of the conversion is as expected. There may be various outputs, depending on which of the `in_row_sers` are used and in what order.


<div align="right" style="text-align: right"><a href="#Contents">Back to Contents</a></div>

# Manual specifications
## Variables and utility functions
We put these in a simple class, so they can be accessed with the same syntax that will be used once they are incorporated into the package.

```python
class premierconverter:
    """Class to hold constants that will actually form part of the package module"""
    def __init__(self):
        """Define constants that are needed for this script"""
        self.RAW_STRUCT = {
            'stop_row_at': 'Total Peril Premium',
            'stem': {
                'ncols': 5,
                'chosen_cols': [0, 1],
                'col_names': ['Premier_Test_Status', 'Total_Premium'],
                'col_types': [np.dtype('object'), np.dtype('float')],
            },
            'f_set': {
                'ncols': 4,
                'col_names': ['Peril_Factor', 'Relativity', 'Premium_increment', 'Premium_cumulative'],
                'col_types': [np.dtype('object')] + [np.dtype('float')] * 3,
            },
            'bp_name': 'Base Premium',
        }
        self.TRUNC_AFTER_REGEX = r",\s*{}.*".format(self.RAW_STRUCT['stop_row_at'])
        self.ROW_ID_NAME = "Ref_num"
        self.OUTPUT_DEFAULTS = {
            'pf_sep': ' ',
            'file_delimiter': ','
        }

# Get an object for use in the rest of the script
PCon = premierconverter()
```

```python
# Utility functions
def add_one_to_index(df):
    """Add 1 to the index values of a Series of DataFrame"""
    df.index += 1
    return df
```

## `in_row_sers`
Typical raw rows as Series

```python
# Usual data rows
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
```

## Expected results from conversion

```python
# Set up and utilty function
df_expected_tests = dict()

def get_output_col_names(perils, factors):
    """Column names of the output data frame that contains `perils` and `factors`"""
    return (
        PCon.RAW_STRUCT['stem']['col_names'] +
        [per + PCon.OUTPUT_DEFAULTS['pf_sep'] + fac 
         for per, fac in pd.MultiIndex.from_product(
             [perils, [PCon.RAW_STRUCT['bp_name']] + factors]
         )]
    )
```

```python
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
```

<div align="right" style="text-align: right"><a href="#Contents">Back to Contents</a></div>

# Automated generate function
## Generate input CSVs

```python
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
```

```python
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
    return None
```

```python
# Try it out
input_rows_lst = [in_row_sers_declined, in_row_sers_03]  # Choose in_row_sers to use
print(generate_input_data_csv(input_rows_lst))
```

## Read the input CSV

```python
# Choose in_row_sers to use in this example
input_rows_lst = [in_row_sers_declined, in_row_sers_03]
```

```python
# Read the input CSV truncated at the specified regex
# and correctly load it as a DataFrame
with io.StringIO(generate_input_data_csv(input_rows_lst)) as in_filepath:
    in_lines_trunc_df = pd.read_csv(
        in_filepath, header=None, index_col=False,
        sep=PCon.TRUNC_AFTER_REGEX,
        engine='python'
    )
assert ((
    in_lines_trunc_df.shape[1] == 1
) or (
    in_lines_trunc_df.iloc[:, 1].isna().sum() == in_lines_trunc_df.shape[0]
))
print(f"Correct: Every row has zero or 1 match to the regex pattern \"{PCon.TRUNC_AFTER_REGEX}\"")

with io.StringIO('\n'.join(in_lines_trunc_df[0])) as in_lines_trunc_stream:
    df_raw = pd.read_csv(
        in_lines_trunc_stream, header=None, index_col=0,
        names=range(in_lines_trunc_df[0].str.count(",").max() + 1)
    ).rename_axis(index=PCon.ROW_ID_NAME)
df_raw.head()
```

## Check it is as expected

```python
def set_na_after_val(row_sers, match_val):
    """
    Return a copy of `row_sers` with values on or after the 
    first instance of `match_val` set to NaN (i.e. missing).
    
    row_sers: Series to look through
    match_val: Scalar to find. If no occurrences are found, 
        return a copy of the original Serires.
    """
    res = row_sers.to_frame('val').assign(
        keep=lambda df: pd.Series(np.select(
            # All matching values are set to 1.0
            # Others are set to NaN
            [df['val'] == match_val],
            [1.0],
            default=np.nan,
        ), index=df.index).ffill(
            # Forward fill so that all entries on or after the first
            # match are set to 1.0, not NaN
        ).isna(),  # Convert NaN/1.0 to True/False
        # Take the original value, except where 'keep' is False,
        # where the value is replaced with NaN.
        new_val=lambda df: df['val'].where(df['keep'], np.nan)
    )['new_val']
    return(res)
```

```python
def trim_na_cols(df):
    """
    Remove any columns on the right of a DataFrame `df` which have all missing 
    values up to the first column with at least one non-missing value.
    """
    keep_col = df.isna().mean(
        # Get proportion of each column that is missing.
        # Columns with all missing values will have 1.0 proportion missing.
    ).to_frame('prop_missing').assign(
        keep=lambda df: pd.Series(np.select(
            # All columns with at least one non-missing value are set to 1.0
            # Others are set to NaN
            [df['prop_missing'] < 1.],
            [1.0],
            default=np.nan,
        ), index=df.index).bfill(
            # Backward fill so that all columns on or before the last
            # column with at least one non-missing value are set to 1.0
        ).notna()  # Convert 1.0/NaN to True/False
    )['keep']
    return(df.loc[:, keep_col])
```

```python
# Set unwanted values to NaN
# and remove surplus columns (with all missing values) from the right.
# Re-cast columns to numeric if possible.
df_raw_expected = pd.DataFrame(input_rows_lst).pipe(add_one_to_index).apply(
    set_na_after_val, match_val=PCon.RAW_STRUCT['stop_row_at'], axis=1
).pipe(trim_na_cols).apply(pd.to_numeric, errors='ignore')

df_raw_expected.head()
```

```python
assert (df_raw_expected.dtypes == df_raw.dtypes).all()
assert df_raw_expected.shape == df_raw.shape
assert (df_raw_expected.isna() == df_raw.isna()).all().all()
assert np.max(np.max(np.abs(
    df_raw_expected.select_dtypes(exclude='object') - df_raw.select_dtypes(exclude='object')
))) < 1e-15
print("Correct: Expected df_raw matches loaded df_raw")
```

<div align="right" style="text-align: right"><a href="#Contents">Back to Contents</a></div>

# Generated dummy data
For use during package development

```python
# All dummy rows
input_rows_lst = [in_row_sers_01, in_row_sers_02, in_row_sers_error, in_row_sers_03, in_row_sers_declined]
in_filepath = raw_data_folder_path / 'minimal_input_adj.csv'
generate_input_data_csv(input_rows_lst, in_filepath)
```

```python
# Look at the output string
print(in_filepath.read_text())
```

```python
# Check it worked
in_lines_trunc_df = pd.read_csv(
    in_filepath, header=None, index_col=False,
    sep=PCon.TRUNC_AFTER_REGEX,
    engine='python'
)
assert ((
    in_lines_trunc_df.shape[1] == 1
) or (
    in_lines_trunc_df.iloc[:, 1].isna().sum() == in_lines_trunc_df.shape[0]
))
print(f"Correct: Every row has zero or 1 match to the regex pattern \"{PCon.TRUNC_AFTER_REGEX}\"")

with io.StringIO('\n'.join(in_lines_trunc_df[0])) as in_lines_trunc_stream:
    df_raw = pd.read_csv(
        in_lines_trunc_stream, header=None, index_col=0,
        names=range(in_lines_trunc_df[0].str.count(",").max() + 1)
    ).rename_axis(index=PCon.ROW_ID_NAME)

df_raw_expected = pd.DataFrame(input_rows_lst).pipe(add_one_to_index).apply(
    set_na_after_val, match_val=PCon.RAW_STRUCT['stop_row_at'], axis=1
).pipe(trim_na_cols).apply(pd.to_numeric, errors='ignore')

assert (df_raw_expected.dtypes == df_raw.dtypes).all()
assert df_raw_expected.shape == df_raw.shape
assert (df_raw_expected.isna() == df_raw.isna()).all().all()
assert np.max(np.max(np.abs(
    df_raw_expected.select_dtypes(exclude='object') - df_raw.select_dtypes(exclude='object')
))) < 1e-15
print("Correct: Expected df_raw matches loaded df_raw")
```

## Save expected conversion results
This is only needed for package development

```python
force_overwrite = False
for nrows, df_expected in df_expected_tests.items():
    expected_filepath = raw_data_folder_path / f'minimal_expected_output_{nrows}.csv'
    if expected_filepath.is_file() and not force_overwrite:
        print(
            "\n\tA file already exists at that location:"
            f"\n\t{expected_filepath.absolute()}"
            "\n\tThis has *not* been overwritten. To overwrite it, re-run"
            "\n\tthis command with `force_overwrite = True`"
        )
        continue
    df_expected.to_csv(expected_filepath)
    print(f"File saved here: {expected_filepath}")
```

## Check it is as expected
See the main development notebook.


<div align="right" style="text-align: right"><a href="#Contents">Back to Contents</a></div>
