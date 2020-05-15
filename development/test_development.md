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
1. [Default data for tests](#Default-data-for-tests)
1. [Integration tests](#Integration-tests):
    - [Succeeding examples](#Succeeding-examples)
    - [Overwriting an existing file](#Overwriting-an-existing-file)
1. [Testing the CLI](#Testing-the-CLI)
    - [CLI succeeding examples](#CLI-succeeding-examples)
    - [CLI force overwrite](#CLI-force-overwrite)
    - [CLI invalid arguments](#CLI-invalid-arguments)
1. [Unit tests](#Unit-tests):
    - [Utility functions](#Utility-functions)
    - [Filepath validation](#Filepath-validation)
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
# Other warnings that come up sometimes
warnings.filterwarnings("ignore", message="The usage of `cmp` is deprecated")
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
import pytest
from click.testing import CliRunner

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
assert pytest.__version__ == '5.0.1'
print(f'pytest version:\t\t\t{pytest.__version__}')
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

# Typical raw DataFrame input
input_rows_lst = [df_raw_row01, df_raw_row02, df_raw_row_error, df_raw_row03]
df_raw_01 = pd.DataFrame(input_rows_lst).pipe(add_one_to_index)
df_raw_01.head()
```

```python
# Expected result
perils = ['AnotherPrl', 'Peril1']
factors = ['Factor1', 'NewFact', 'SomeFact']
df_expected_01 = pd.DataFrame(
    columns=(
        ['Premier_Test_Status', 'Total_Premium'] +
        [per + PCon.OUTPUT_DEFAULTS['pf_sep'] + fac 
         for per, fac in
         pd.MultiIndex.from_product([perils, ['Base Premium'] + factors])]
    ),
    data=[
        (df_raw_row01[[1, 2, 5+4*2]].to_list() + [1.] * 3 + 
         df_raw_row01[[5+4*1, 5+4*2+2]].to_list() + [1.] * 2),
        df_raw_row02[[1, 2, 5+4*1, 5+4*1+2]].to_list() + [1.] + df_raw_row02[[5+4*3+2, 5+4*3]].to_list() + [1.] * 3,
        df_raw_row_error[[1]].to_list() + [0.] * 9,
        (df_raw_row03[[1, 2, 5+4*4]].to_list() + [1.] * 3 + 
         df_raw_row03[[5+4*5, 5+4*1+2, 5+2]].to_list() + [1.])
    ],
).pipe(add_one_to_index).rename_axis(index='Ref_num')
df_expected_01
```

## Test steps
### Setup
Create the input data file.

```python
in_filepath = tmp_dir_path / 'tmp_input.csv'
df_raw_01 = pd.DataFrame(input_rows_lst).pipe(add_one_to_index)
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
out_filepath = tmp_dir_path / 'tmp_output.csv'
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
assert PCon.formatted_dfs_are_equal(df_formatted_01, df_reload_01)
print("Correct: The reloaded values are equal, up to floating point tolerance")
```

```python
# Check it matches expected output
assert PCon.formatted_dfs_are_equal(df_formatted_01, df_expected_01)
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

# Default data for tests
Where tests are looking for specific aspects of the input or output data, the data can be created on an ad hoc basis. However, many tests just need some reasonable, minimal data - which is specified here, so it can be easily recreated / used for each such test.

```python
def create_input_data_csv(in_filepath, input_rows_lst):
    """Creates the input DataFrame and saves it as a CSV at `in_filepath`"""
    df_raw_01 = pd.DataFrame(input_rows_lst).pipe(add_one_to_index)
    df_raw_01.to_csv(in_filepath, index=True, header=None)
    return(df_raw_01)
```

Default expected output for each nrows, so we don't have to specify it multiple times

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
```

<div align="right" style="text-align: right"><a href="#Contents">Back to Contents</a></div>

# Integration tests
## Succeeding examples
### Default arguments

```python
# Setup
in_filepath = tmp_dir_path / 'tmp_input.csv'
out_filepath = tmp_dir_path / 't01_output.csv'

# Given: Input data
_ = create_input_data_csv(in_filepath, input_rows_lst)

# When: Apply function
res_filepath = PCon.convert(in_filepath, out_filepath)
df_reload_01 = PCon.load_formatted_file(res_filepath) # Reload resulting data from workbook

# Then: Result is as expected
assert PCon.formatted_dfs_are_equal(df_reload_01, df_expected_tests[4])
print("Correct: The reloaded values are equal, up to floating point tolerance")

# Teardown
[filepath.unlink() 
 for filepath in [in_filepath, res_filepath] 
 if filepath.is_file()]
assert dir_is_empty(tmp_dir_path)
print("Correct: Workspace restored")
```

### Limit number of rows

```python
# Setup
in_filepath = tmp_dir_path / 'tmp_input.csv'
out_filepath = tmp_dir_path / 't02_output.csv'

# Given: Input data
_ = create_input_data_csv(in_filepath, input_rows_lst)

# When: Apply function with limited rows
nrows = 2  # Possible values: [2, 4, 5, None]
res_filepath = PCon.convert(in_filepath, out_filepath, nrows=nrows)
df_reload_01 = PCon.load_formatted_file(res_filepath) # Reload resulting data from workbook

# Then: Result is as expected
assert PCon.formatted_dfs_are_equal(
    df_reload_01,
    df_expected_tests[min(nrows if nrows is not None else 100, 4)]
)
print("Correct: The reloaded values are equal, up to floating point tolerance")

# Teardown
[filepath.unlink() 
 for filepath in [in_filepath, res_filepath] 
 if filepath.is_file()]
assert dir_is_empty(tmp_dir_path)
print("Correct: Workspace restored")
```

## Overwriting an existing file

```python
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
err = None
try:
    PCon.convert(in_filepath, out_filepath)
except Exception as e:
    err = e
assert err is not None  # An error was thrown...
assert isinstance(err, FileExistsError)  # ...of this specific type
assert 'File already exists' in str(err)  # The error message contains is helpful...
assert str(out_filepath.absolute()) in str(err)  # ...and contains the filepath
assert out_filepath.read_text() == out_file_str  # The file contents are unchanged
print("Correct: File was not overwritten and helpful error message was thrown")

# When: Apply function force overwrite
res_filepath = PCon.convert(in_filepath, out_filepath, force_overwrite=True)

# Then: Result is as expected
df_reload_01 = PCon.load_formatted_file(res_filepath)  # Reload resulting data from workbook
assert PCon.formatted_dfs_are_equal(df_reload_01, df_expected_tests[4])
print("Correct: The reloaded values are equal, up to floating point tolerance")

# Teardown
[filepath.unlink() 
 for filepath in [in_filepath, res_filepath] 
 if filepath.is_file()]
assert dir_is_empty(tmp_dir_path)
print("Correct: Workspace restored")
```

<div align="right" style="text-align: right"><a href="#Contents">Back to Contents</a></div>

# Testing the CLI
## CLI succeeding examples
### Version and help

```python
# When: Submit the 'version' option to the CLI
runner = CliRunner()
result = runner.invoke(PCon.cli, ['--version'])

# Then: Returns the correct version
assert result.exit_code == 0
assert result.output == f"cli, version {PCon.__version__}\n"
print("Correct: Version is available from the CLI")
```

```python
# When: Submit the 'help' option to the CLI
runner = CliRunner()
result = runner.invoke(PCon.cli, ['-h'])

# Then: Print the help, which includes some useful phrases
assert result.exit_code == 0
assert 'Usage: cli [OPTIONS] <input filepath> <output filepath>\n' in result.output
assert '--force' in result.output
assert '-r, --nrows INTEGER' in result.output
assert '-s, --sep TEXT' in result.output
assert '-n, --no_checks' in result.output
print("Correct: Help is available and useful from the CLI")
```

### Default arguments

```python
# Setup
in_filepath = tmp_dir_path / 'tmp_input.csv'
out_filepath = tmp_dir_path / 't04_output.csv'

# Given: Input data
_ = create_input_data_csv(in_filepath, input_rows_lst)

# When: We run the CLI with default arguments
runner = CliRunner()
result = runner.invoke(
    PCon.cli,
    [str(in_filepath), str(out_filepath)]  # Default arguments
)

# Then: The CLI command completes successfully
# and the resulting output data is as expected
assert result.exit_code == 0
assert result.output == f"Output saved here:\t{out_filepath.absolute()}\n"
print("Correct: CLI has completed without error and with correct message")

df_reload_01 = PCon.load_formatted_file(out_filepath)
assert PCon.formatted_dfs_are_equal(df_reload_01, df_expected_tests[4])
print("Correct: The reloaded values are equal, up to floating point tolerance")

# Teardown
[filepath.unlink() 
 for filepath in [in_filepath, out_filepath] 
 if filepath.is_file()]
assert dir_is_empty(tmp_dir_path)
print("Correct: Workspace restored")
```

### Limit number of rows

```python
# Setup
in_filepath = tmp_dir_path / 'tmp_input.csv'
out_filepath = tmp_dir_path / 't05_output.csv'

# Given: Input data
_ = create_input_data_csv(in_filepath, input_rows_lst)

# When: We run the CLI with option for limited number of rows
nrows = 2  # Possible values: [2, 4, 5, None]
runner = CliRunner()
result = runner.invoke(PCon.cli, [
    str(in_filepath), str(out_filepath),
    '-r', nrows,
])

# Then: The CLI command completes successfully
# and the resulting output data is as expected
assert result.exit_code == 0
assert result.output == f"Output saved here:\t{out_filepath.absolute()}\n"
print("Correct: CLI has completed without error and with correct message")

df_reload_01 = PCon.load_formatted_file(out_filepath)
assert PCon.formatted_dfs_are_equal(
    df_reload_01,
    df_expected_tests[min(nrows if nrows is not None else 100, 4)]
)
print("Correct: The reloaded values are equal, up to floating point tolerance")

# Teardown
[filepath.unlink() 
 for filepath in [in_filepath, out_filepath] 
 if filepath.is_file()]
assert dir_is_empty(tmp_dir_path)
print("Correct: Workspace restored")
```

### Custom separator

```python
# Setup
in_filepath = tmp_dir_path / 'tmp_input.txt'
out_filepath = tmp_dir_path / 't06_output'

# Given: Input data with custom separator character
custom_sep = '|'  # Can parametrise this value with various length-one strings, e.g. ['|', '_', '\t']
df_raw_01 = pd.DataFrame(input_rows_lst).pipe(add_one_to_index)
df_raw_01.to_csv(in_filepath, index=True, header=None, sep=custom_sep)

# When: We run the CLI with option for limited number of rows
runner = CliRunner()
result = runner.invoke(PCon.cli, [
    str(in_filepath), str(out_filepath),
    '-s', custom_sep,
])

# Then: The CLI command completes successfully
# and the resulting output data is as expected
assert result.exit_code == 0
assert result.output == f"Output saved here:\t{out_filepath.absolute()}\n"
print("Correct: CLI has completed without error and with correct message")

df_reload_01 = PCon.load_formatted_file(out_filepath, file_delimiter=custom_sep)
assert PCon.formatted_dfs_are_equal(df_reload_01, df_expected_tests[4])
print("Correct: The reloaded values are equal, up to floating point tolerance")

# Teardown
[filepath.unlink() 
 for filepath in [in_filepath, out_filepath] 
 if filepath.is_file()]
assert dir_is_empty(tmp_dir_path)
print("Correct: Workspace restored")
```

## CLI force overwrite

```python
# Setup
in_filepath = tmp_dir_path / 'tmp_input.csv'
out_filepath = tmp_dir_path / 't07_output.csv'

# Given: Input data and a file already exists in the output location
_ = create_input_data_csv(in_filepath, input_rows_lst)

out_file_str = 'Some basic file contents'
_ = out_filepath.write_text(out_file_str)
assert out_filepath.read_text() == out_file_str  # Check it has worked

# When: We run the CLI with default arguments (i.e. not force overwrite)
runner = CliRunner()
result = runner.invoke(PCon.cli, [str(in_filepath), str(out_filepath)])

# Then: The CLI command exits with an error
assert result.exit_code == 1
assert result.output == ''
assert result.exception  # An error was thrown...
assert isinstance(result.exception, FileExistsError)  # ...of this specific type
assert 'File already exists' in str(result.exception)  # The error message contains is helpful...
assert str(out_filepath.absolute()) in str(result.exception)  # ...and contains the filepath
assert out_filepath.read_text() == out_file_str  # The file contents are unchanged
print("Correct: File was not overwritten and helpful error message was thrown")

# When: Run the CLI with force overwrite
result = runner.invoke(
    PCon.cli, [str(in_filepath), str(out_filepath), '--force']
)

# Then: Result is as expected
df_reload_01 = PCon.load_formatted_file(out_filepath)  # Reload resulting data from workbook
assert PCon.formatted_dfs_are_equal(df_reload_01, df_expected_tests[4])
print("Correct: The reloaded values are equal, up to floating point tolerance")

# Teardown
[filepath.unlink() 
 for filepath in [in_filepath, out_filepath] 
 if filepath.is_file()]
assert dir_is_empty(tmp_dir_path)
print("Correct: Workspace restored")
```

## CLI invalid arguments
### Missing filepaths

```python
# Setup
out_filepath = tmp_dir_path / 't08_output.csv'

# When: We run the CLI with no arguments
runner = CliRunner()
result = runner.invoke(PCon.cli, [])

# Then: The CLI command exits with an error
assert result.exit_code == 2
assert "Error: Missing argument '<input filepath>'." in result.output
assert not out_filepath.is_file()
print("Correct: CLI shows an error message and output file is not written")

# Teardown
assert dir_is_empty(tmp_dir_path)
print("Correct: Workspace restored")
```

```python
# Setup
in_filepath = tmp_dir_path / 'tmp_input.csv'
out_filepath = tmp_dir_path / 't09_output.csv'

# Given: A file exists in the input location
assert not in_filepath.is_file()
in_filepath.write_text('Some text for the file')
assert in_filepath.is_file()

# When: We run the CLI with an input file that exists but no output file
runner = CliRunner()
result = runner.invoke(PCon.cli, [str(in_filepath)])

# Then: The CLI command exits with an error
assert result.exit_code == 2
assert "Error: Missing argument '<output filepath>'." in result.output
assert not out_filepath.is_file()
print("Correct: CLI shows an error message and output file is not written")

# Teardown
in_filepath.unlink() if in_filepath.is_file() else None
assert dir_is_empty(tmp_dir_path)
print("Correct: Workspace restored")
```

```python
# Setup
in_filepath = tmp_dir_path / 'tmp_input.csv'
out_filepath = tmp_dir_path / 't10_output.csv'

# When: We run the CLI with an input file that does not exist
runner = CliRunner()
result = runner.invoke(PCon.cli, [str(in_filepath), str(out_filepath)])

# Then: The CLI command exits with an error
assert result.exit_code == 2
assert f"Path '{str(in_filepath)}' does not exist." in result.output
assert not out_filepath.is_file()
print("Correct: CLI shows an error message and output file is not written")

# Teardown
assert dir_is_empty(tmp_dir_path)
print("Correct: Workspace restored")
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

## Filepath validation
### `in_filepath`

```python
# Given: Input file does not exist
in_filepath = tmp_dir_path / 'foo.csv'
# When: Attempt to validate the filepath
# Then: Error thrown
with pytest.raises(FileNotFoundError) as err:
    PCon.validate_input_options(in_filepath)
assert err is not None  # An error was thrown...
assert isinstance(err.value, FileNotFoundError)  # ...of this specific type
assert 'There is no file at the input location' in str(err.value)  # The error message contains is helpful...
assert str(in_filepath.absolute()) in str(err.value)  # ...and contains the filepath
print("Correct: Helpful error message was thrown")
```

```python
# Given: Input file exists but does not have a recognised extension
in_filepath = tmp_dir_path / 'foo.foo'
in_filepath.write_text("Some text")
assert in_filepath.is_file()
# When: Attempt to validate the filepath
# Then: Warning is thrown (not an exception)
with pytest.warns(UserWarning) as wrns:
    PCon.validate_input_options(in_filepath)
assert len(wrns) == 1  # Exactly 1 warning message was thrown
assert (
    f"file extension '{in_filepath.suffix}' "
    "is not one of the recognised file extensions" 
    in wrns[0].message.args[0]
)
print("Correct: Helpful warning was thrown")

# Teardown
in_filepath.unlink() if in_filepath.is_file() else None
assert dir_is_empty(tmp_dir_path)
print("Correct: Workspace restored")
```

```python
# Given: Input file exists and has a recognised extension
in_filepath = tmp_dir_path / 'fo o'  # Parametrise this, e.g. ['foo.csv', 'fo o', 'foo.CsV', '01_f.tXt']
in_filepath.write_text("Some text")
assert in_filepath.is_file()
# When: Attempt to validate the filepath
# Then: No warnings or errors are thrown
rtn_val = 1
with pytest.warns(None) as wrns:
    rtn_val = PCon.validate_input_options(in_filepath)
assert len(wrns) == 0  # No warnings are produced
assert rtn_val is None  # Validation function completed
print("Correct: `in_filepath` was validated")

# Teardown
in_filepath.unlink() if in_filepath.is_file() else None
assert dir_is_empty(tmp_dir_path)
print("Correct: Workspace restored")
```

### `out_filepath`

```python
# Given: Output file already exists
out_filepath = tmp_dir_path / '01_f.tXt'  # Parametrise this, e.g. ['foo.csv', 'fo o', 'foo.CsV', '01_f.tXt']
out_filepath.write_text("Some text")
assert out_filepath.is_file()
# When: Attempt to validate the filepath (and no force_overwrite passed)
# Then: Error thrown
with pytest.raises(FileExistsError) as err:
    PCon.validate_output_options(out_filepath)
assert err is not None  # An error was thrown...
assert isinstance(err.value, FileExistsError)  # ...of this specific type
assert 'File already exists at the output location' in str(err.value)  # The error message contains is helpful...
assert str(out_filepath.absolute()) in str(err.value)  # ...and contains the filepath
assert 'If you want to overwrite it, re-run with `force_overwrite = True`' in str(err.value)
print("Correct: Helpful error message was thrown")

# When: Attempt to validate the filepath with force_overwrite
# Then: No warnings or errors are thrown
rtn_val = 1
with pytest.warns(None) as wrns:
    rtn_val = PCon.validate_output_options(out_filepath, force_overwrite=True)
assert len(wrns) == 0  # No warnings are produced
assert rtn_val is None  # Validation function completed
print("Correct: `out_filepath` was validated")

# Teardown
out_filepath.unlink() if out_filepath.is_file() else None
assert dir_is_empty(tmp_dir_path)
print("Correct: Workspace restored")
```

```python
# Given: Output file location is in a folder that does not exist
# (so certainly the output file does not exist)
out_dir = tmp_dir_path / 'another folder'
out_filepath = out_dir / 'foo.csv'
# When: Attempt to validate the filepath
# Then: Error thrown
with pytest.raises(FileNotFoundError) as err:
    PCon.validate_output_options(out_filepath)
assert err is not None  # An error was thrown...
assert isinstance(err.value, FileNotFoundError)  # ...of this specific type
assert 'The folder of the output file does not exist' in str(err.value)  # The error message contains is helpful...
assert str(out_filepath.parent.absolute()) in str(err.value)  # ...and contains the filepath
print("Correct: Helpful error message was thrown")
```

```python
# Given: Output file deos not have a recognised extension
out_filepath = tmp_dir_path / 'fo o.xlsx'  # Parametrise this
# When: Attempt to validate the filepath
# Then: Warning is thrown (not an exception)
with pytest.warns(UserWarning) as wrns:
    PCon.validate_output_options(out_filepath)
assert len(wrns) == 1  # Exactly 1 warning message was thrown
assert (
    f"file extension '{out_filepath.suffix}' "
    "is not one of the recognised file extensions" 
    in wrns[0].message.args[0]
)
print("Correct: Helpful warning was thrown")
```

```python
# Given: Output file has a recognised extension and does not exist
out_filepath = tmp_dir_path / 'foo.CsV'  # Parametrise this, e.g. ['foo.csv', 'fo o', 'foo.CsV', '01_f.tXt']
# When: Attempt to validate the filepath
# Then: No warnings or errors are thrown
rtn_val = 1
with pytest.warns(None) as wrns:
    rtn_val = PCon.validate_output_options(out_filepath)
assert len(wrns) == 0  # No warnings are produced
assert rtn_val is None  # Validation function completed
print("Correct: `out_filepath` was validated")
```

<div align="right" style="text-align: right"><a href="#Contents">Back to Contents</a></div>
