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
1. [Setup](#Setup)
1. [Data for tests](#Data-for-tests)
1. [Typical workflow](#Typical-workflow)
1. [Unit tests](#Unit-tests):
    - [Utility functions](#Utility-functions)
    - [TBA](#TBA)
1. [Integration tests](#Integration-tests):
    - [`convert` pipeline](#convert-pipeline)
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
# Configuration variables
if on_kaggle:
    raw_data_folder_path = Path('/kaggle/input') / 'dummy-premier-data-raw'
else:
    import proj_config
    raw_data_folder_path = proj_config.example_data_dir_path
assert raw_data_folder_path.is_dir()
print("Correct: All locations are available as expected")
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

# Data for tests
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
    'Total Peril Premium',
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

<div align="right" style="text-align: right"><a href="#Contents">Back to Contents</a></div>

# Typical workflow

```python
# Setup
in_filepath = tmp_dir_path / 't01_typical_input.csv'
df_raw_01 = pd.DataFrame([
    df_raw_row01, df_raw_row02, df_raw_row_error
]).pipe(add_one_to_index)
df_raw_01.to_csv(in_filepath, index=True, header=None)

# Check it has been created correctly
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
print("Correct: The CSV that has been created can be loaded from CSV and matches")
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

```python
res_filepath.is_file()
```

```python
# Teardown: Delete the files that have been created
[filepath.unlink() 
 for filepath in [in_filepath, res_filepath] 
 if filepath.is_file()]
assert dir_is_empty(tmp_dir_path)
print("Correct: Workspace restored")
```

<div align="right" style="text-align: right"><a href="#Contents">Back to Contents</a></div>

# Unit tests
**TODO**: Write this section

```python

```

```python

```

<div align="right" style="text-align: right"><a href="#Contents">Back to Contents</a></div>

# Integration tests
**TODO**: Write this section

```python

```

<div align="right" style="text-align: right"><a href="#Contents">Back to Contents</a></div>
