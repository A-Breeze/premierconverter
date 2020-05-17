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

**Important note**: No data files are committed to this repo, including for use in the automated tests. Any "data" used in this repo is entirely dummy data, i.e. it has been randomised and all names have been masked so they can be used for training purposes. This notebook is for training purposes only.
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
    - [Include additional factors](#Include-additional-factors)
    - [Alternative row order](#Alternative-row-order)
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

# Silently import another notebook. This *runs the code* of the notebook.
import contextlib, io
f = io.StringIO()
with contextlib.redirect_stdout(f):
    import simulate_dummy_data as conf
add_one_to_index = conf.add_one_to_index
generate_input_data_csv = conf.generate_input_data_csv
    
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
# Get some typical input rows, represented as Series
input_rows_lst = [
    conf.in_row_sers_01,
    conf.in_row_sers_02,
    conf.in_row_sers_error,
    conf.in_row_sers_03,
    conf.in_row_sers_declined,
]
input_rows_lst[0].to_frame().T  # Example
```

```python
# Generate the CSV of typical input
print(generate_input_data_csv(input_rows_lst))
```

```python
# Expected result
df_expected_tests = conf.df_expected_tests
df_expected_tests[5]
```

## Test steps
### Setup

```python
# Create the input data file
in_filepath = tmp_dir_path / 'tmp_input.csv'
generate_input_data_csv(input_rows_lst, in_filepath)
```

```python
# Create the expected output as a DataFrame
conf.df_expected_tests[5]
```

### Test

```python
# Check data can be loaded
in_lines_trunc_df = PCon.read_input_lines(in_filepath)
PCon.validate_input_lines_trunc(in_lines_trunc_df)
df_trimmed_from_csv = PCon.split_lines_to_df(in_lines_trunc_df)

# Compare against what we'd expect
df_trimmed_expected = pd.DataFrame(input_rows_lst).pipe(conf.add_one_to_index).apply(
    conf.set_na_after_val, match_val=PCon.RAW_STRUCT['stop_row_at'], axis=1
).pipe(conf.trim_na_cols).apply(pd.to_numeric, errors='ignore')

assert (df_trimmed_from_csv.index == df_trimmed_expected.index).all()
assert (df_trimmed_from_csv.dtypes == df_trimmed_expected.dtypes).all()
assert (df_trimmed_from_csv.isna() == df_trimmed_expected.isna()).all().all()
assert (np.abs(
    df_trimmed_from_csv.select_dtypes(['int', 'float']).fillna(0) - 
    df_trimmed_expected.select_dtypes(['int', 'float']).fillna(0)
) < 1e-10).all().all()
assert (
    df_trimmed_from_csv.select_dtypes(exclude=['int', 'float']).astype(str) == 
    df_trimmed_expected.select_dtypes(exclude=['int', 'float']).astype(str)
).all().all()
print("Correct: The CSV that has been created can be loaded and matches")
```

```python
# Run with default arguments
out_filepath = tmp_dir_path / 'tmp_output.csv'
res_filepath = PCon.convert(in_filepath, out_filepath)
```

```python
# Check against expected output
df_reload = PCon.load_formatted_file(out_filepath)
assert PCon.formatted_dfs_are_equal(df_reload, conf.df_expected_tests[5])
print("Correct: The reloaded values are equal, up to floating point tolerance")
```

### Teardown
Delete the files that have been created

```python
[filepath.unlink() 
 for filepath in [in_filepath, out_filepath] 
 if filepath.is_file()]
assert dir_is_empty(tmp_dir_path)
print("Correct: Workspace restored")
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
_ = generate_input_data_csv(input_rows_lst, in_filepath)

# When: Apply function
res_filepath = PCon.convert(in_filepath, out_filepath)

# Then: Result is as expected
df_reload_01 = PCon.load_formatted_file(res_filepath)  # Reload resulting data from workbook
assert PCon.formatted_dfs_are_equal(df_reload_01, df_expected_tests[5])
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
_ = generate_input_data_csv(input_rows_lst, in_filepath)

# When: Apply function with limited rows
nrows = 2  # Possible values: [2, 4, 5, None]
res_filepath = PCon.convert(in_filepath, out_filepath, nrows=nrows)

# Then: Result is as expected
df_reload_01 = PCon.load_formatted_file(res_filepath)
assert PCon.formatted_dfs_are_equal(
    df_reload_01,
    df_expected_tests[min(nrows if nrows is not None else 100, 5)]
)
print("Correct: The reloaded values are equal, up to floating point tolerance")

# Teardown
[filepath.unlink() 
 for filepath in [in_filepath, res_filepath] 
 if filepath.is_file()]
assert dir_is_empty(tmp_dir_path)
print("Correct: Workspace restored")
```

### Alternative row order

```python
# Setup
in_filepath = tmp_dir_path / 'tmp_input.csv'
out_filepath = tmp_dir_path / 't11_output.csv'

# Given: Input data consisting of certain rows in a different order
idx_ordered, expected_label = [4, 3, 2, 1, 0], 5 # Can parametrise
# idx_ordered, expected_label = [2] + list(range(5)), 5
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

# Teardown
[filepath.unlink() 
 for filepath in [in_filepath, res_filepath] 
 if filepath.is_file()]
assert dir_is_empty(tmp_dir_path)
print("Correct: Workspace restored")
```

## Include factors

```python
# Setup
in_filepath = tmp_dir_path / 'tmp_input.csv'
out_filepath = tmp_dir_path / 't10_output.csv'

# Given: Input data
_ = generate_input_data_csv(input_rows_lst, in_filepath)

# When: Apply function with limited rows and specify factors to include
nrows = 2
expected_label, include_factors = '2_all_facts', ['NewFact', 'SomeFact']  # Can parametrise
res_filepath = PCon.convert(
    in_filepath, out_filepath,
    nrows=nrows, include_factors=include_factors
)

# Then: Result is as expected
df_reload_01 = PCon.load_formatted_file(res_filepath)
assert PCon.formatted_dfs_are_equal(df_reload_01, df_expected_tests[expected_label])
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
assert '-n, --no_checks' in result.output
assert '-p, --hide_progress' in result.output
print("Correct: Help is available and useful from the CLI")
```

### Default arguments

```python
# Setup
in_filepath = tmp_dir_path / 'tmp_input.csv'
out_filepath = tmp_dir_path / 't04_output.csv'

# Given: Input data
_ = generate_input_data_csv(input_rows_lst, in_filepath)

# When: We run the CLI with default arguments
runner = CliRunner()
result = runner.invoke(
    PCon.cli,
    [str(in_filepath), str(out_filepath)]  # Default arguments
)

# Then: The CLI command completes successfully
# and the resulting output data is as expected
assert result.exit_code == 0
assert f"Output saved here: {out_filepath.absolute()}\n" in result.output
for step_num in range(1, 6):
    assert f"Step {step_num}" in result.output
print("Correct: CLI has completed without error and with correct message")

df_reload_01 = PCon.load_formatted_file(out_filepath)
assert PCon.formatted_dfs_are_equal(df_reload_01, df_expected_tests[5])
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
_ = generate_input_data_csv(input_rows_lst, in_filepath)

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
assert f"Output saved here: {out_filepath.absolute()}\n" in result.output
print("Correct: CLI has completed without error and with correct message")

df_reload_01 = PCon.load_formatted_file(out_filepath)
assert PCon.formatted_dfs_are_equal(
    df_reload_01,
    df_expected_tests[min(nrows if nrows is not None else 100, 5)]
)
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
_ = generate_input_data_csv(input_rows_lst, in_filepath)

out_file_str = 'Some basic file contents'
_ = out_filepath.write_text(out_file_str)
assert out_filepath.read_text() == out_file_str  # Check it has worked

# When: We run the CLI with default arguments (i.e. not force overwrite)
runner = CliRunner()
result = runner.invoke(PCon.cli, [str(in_filepath), str(out_filepath)])

# Then: The CLI command exits with an error
assert result.exit_code == 1
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
assert PCon.formatted_dfs_are_equal(df_reload_01, df_expected_tests[5])
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
