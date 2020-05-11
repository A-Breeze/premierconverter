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
# Data Conversion Challenge
Challenge to automate the conversion of raw data into a specified format of data to make it more usable.

**Important note**: The data used in this notebook has been randomised and all names have been masked so they can be used for training purposes. This notebook is for training purposes only.

This notebook is available in the following locations. These versions are kept in sync *manually* - there should not be discrepancies, but it is possible.
- On Kaggle: <https://www.kaggle.com/btw78jt/data-conversion-challenge-202004>
- In the GitHub project repo: <https://github.com/A-Breeze/premierconverter>. See the `README.md` for further instructions.
<!-- #endregion -->

<!-- #region _cell_guid="79c7e3d0-c299-4dcb-8224-4455121ee9b0" _uuid="d629ff2d2480ee46fbb7e2d37f6b5fab8052498a" -->
<!-- This table of contents is updated *manually* -->
# Contents
1. [Setup](#Setup): Import packages, Config variables
1. [Variables](#Variables): Raw data structure, Inputs
1. [Workflow](#Workflow): Load raw data, Remove unwanted extra values, Stem section, Factor sets, Output to CSV, Load expected output to check it is as expected
1. [Using the functions](#Using-the-functions)
1. [Unused rough work](#Unused-rough-work): Replace multiple string terms, Chained drop a column MultiIndex level
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
# Other warnings that sometimes occur
warnings.filterwarnings("ignore", message="unclosed file <_io.Buffered")
```

```python
# Determine whether this notebook is running on Kaggle
from pathlib import Path

on_kaggle = False
print("Current working directory: " + str(Path('.').absolute()))
if str(Path('.').absolute()) == '/kaggle/working':
    on_kaggle = True
```

```python
# Import built-in modules
import sys
import platform
import os

# Import external modules
from IPython import __version__ as IPy_version
import numpy as np
import pandas as pd
from click import __version__ as click_version

# Import project modules
if not on_kaggle:
    from pyprojroot import here
    root_dir_path = here()
    # Allow modules to be imported relative to the project root directory
    if not sys.path[0] == root_dir_path:
        sys.path.insert(0, str(root_dir_path))
import premierconverter as PCon

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
# Output exact environment specification, in case it is needed later
print("Capturing full package environment spec")
print("(But note that not all these packages are required)")
!pip freeze > requirements_snapshot.txt
!jupyter --version > jupyter_versions_snapshot.txt
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

<div align="right" style="text-align: right"><a href="#Contents">Back to Contents</a></div>

# Variables


## Raw data structure

```python
# Configuration variables for the expected format and structure of the data
accepted_file_extensions = ['.csv', '', '.txt']
input_file_encodings = ['utf-8', 'latin-1', 'ISO-8859-1']
file_delimiter = ','

raw_struct = {
    'stop_row_at': 'Total Peril Premium',
    'stem': {
        'ncols': 5,
        'chosen_cols': [0,1],
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

# Output variables, considered to be constants
# Column name of the row IDs
row_id_name = "Ref_num"
```

## Parameters

```python
# Include Factors which are not found in the data
include_factors = None
if include_factors is None:
    include_factors = []

# Maximum number of rows to read in
nrows = None

# Seperator for Peril_Factor column names in output
pf_sep = '_'
```

```python
# Input file location
in_filepath = raw_data_folder_path / 'minimal01_input.csv'

# Checks the file exists and is an Excel file
in_filepath = Path(in_filepath)
if not in_filepath.is_file():
    raise FileNotFoundError(
        "\n\tin_filepath: There is no file at the input location:"
        f"\n\t'{in_filepath.absolute()}'"
        "\n\tCannot read the input data"
    )
if not in_filepath.suffix.lower() in accepted_file_extensions:
    warnings.warn(
        f"in_filepath: The input file extension '{in_filepath.suffix}' "
        f"is not one of the recognised file extensions {accepted_file_extensions}"
    )
print("Correct: Input file exists and has a recognised extension")
```

```python
# Output file location
out_filepath = 'formatted_dummy_data1.csv'
force_overwrite = False

# Checks
out_filepath = Path(out_filepath)

if not out_filepath.parent.is_dir():
    raise FileNotFoundError(
        f"\n\tout_filepath: The folder of the output file does not exist"
        f"Folder path: '{out_filepath.parent}'"
        "\n\tCreate the output folder before running this command"
    )

if out_filepath.is_file() and not force_overwrite:
    raise FileExistsError(
        "\n\tOutput options: File already exists at the output location:"
        f"\n\t'{out_filepath.absolute()}'"
        "\n\tIf you want to overwrite it, re-run with `force_overwrite = True`"
    )
else:
    if not out_filepath.suffix in accepted_file_extensions:
        warnings.warn(
            f"out_filepath: The output file extension '{out_filepath.suffix}' "
            f"is not one of the recognised file extensions {accepted_file_extensions}",
        )

print("Correct: A suitable location for output has been chosen")
```

<div align="right" style="text-align: right"><a href="#Contents">Back to Contents</a></div>

# Workflow


## Load raw data

```python
df_raw = None
for encoding in input_file_encodings:
    try:
        df_raw = pd.read_csv(
            in_filepath,
            header=None, index_col=0, nrows=nrows,
            sep=file_delimiter, encoding=encoding
        ).rename_axis(index=row_id_name)
        # print(f"'{encoding}': Success")  # Used for debugging only
        break
    except UnicodeDecodeError:
        # print(f"'{encoding}': Fail")  # Used for debugging only
        pass
if df_raw is None:
    raise IOError(
        "\n\tread_raw_data: pandas.read_csv() failed."
        f"\n\tFile cannot be read with any of the encodings: {input_file_encodings}"
    )

df_raw.head()
```

```python
# Check it is not malformed
if df_raw.shape[1] == 0:
    warnings.warn(
        "Raw data: No columns of data have been read. "
        "Are you sure you have specified the correct file? "
        f"Are values seperated by the character '{file_delimiter}'?"
    )
if df_raw.shape[0] <= 1:
    warnings.warn(
        "Raw data: Only one row of data has been read. "
        "Are you sure you have specified the correct file? "
        "Are rows of data split into lines of the file?"
    )
if not df_raw.index.is_unique:
    warnings.warn(
        f"Raw data: Row identifiers '{row_id_name}' are not unique. "
        "This may lead to unexpected results."
    )
```

## Remove unwanted extra values

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
df_trimmed = df_raw.apply(
    set_na_after_val, match_val=raw_struct['stop_row_at'], axis=1
).pipe(trim_na_cols).apply(pd.to_numeric, errors='ignore')

df_trimmed.head()
```

```python
# Check it is as expected
if not (
    # At least the stem columns and one factor set column
    df_trimmed.shape[1] >= 
    raw_struct['stem']['ncols'] + 1 * raw_struct['f_set']['ncols']
) or not (
    # Stem columns plus a multiple of factor set columns
    (df_trimmed.shape[1] - raw_struct['stem']['ncols']) 
    % raw_struct['f_set']['ncols'] == 0
):
    warnings.warn(
        f"Trimmed data: Incorrect number of columns with relevant data: {df_trimmed.shape[1] + 1}"
        "\n\tThere should be: 1 for index, "
        f"{raw_struct['stem']['ncols']} for stem section, "
        f"and by a multiple of {raw_struct['f_set']['ncols']} for factor sets"
    )
```

## Stem section

```python
# Get the stem section of columns
df_stem = df_trimmed.iloc[
    :, raw_struct['stem']['chosen_cols']
].pipe(  # Rename the columns
    lambda df: df.rename(columns=dict(zip(
        df.columns, 
        raw_struct['stem']['col_names']
    )))
)

df_stem.head()
```

```python
# Checks
if not (
    df_stem.dtypes == raw_struct['stem']['col_types']
).all():
    warnings.warn(
        "Stem columns: Unexpected column data types"
        f"\n\tExepcted: {raw_struct['stem']['col_types']}"
        f"\n\tActual:   {df_stem.dtypes.tolist()}"
    )
```

## Factor sets

```python
# Combine the rest of the DataFrame into one
df_fsets = pd.concat([
    # For each of the factor sets of columns
    df_trimmed.iloc[  # Select the columns
        :, fset_start_col:(fset_start_col + raw_struct['f_set']['ncols'])
    ].dropna(  # Remove rows that have all missing values
        how="all"
    ).pipe(lambda df: df.rename(columns=dict(zip(  # Rename columns
        df.columns, raw_struct['f_set']['col_names']
    )))).reset_index()  # Get row_ID as a column

    for fset_start_col in range(
        raw_struct['stem']['ncols'], df_trimmed.shape[1], raw_struct['f_set']['ncols']
    )
], sort=False).reset_index(drop=True)  # Best practice to ensure a unique index

df_fsets.head()
```

```python
# Checks
if not (
    df_fsets[raw_struct['f_set']['col_names']].dtypes == 
    raw_struct['f_set']['col_types']
).all():
    warnings.warn(
        "Factor sets columns: Unexpected column data types"
        f"\n\tExpected: {raw_struct['f_set']['col_types']}"
        f"\n\tActual:   {df_fsets[raw_struct['f_set']['col_names']].dtypes.tolist()}"
    )
```

```python
perils_implied = df_fsets.Peril_Factor.drop_duplicates(  # Get only unique 'Peril_Factor' combinations
).to_frame().pipe(lambda df: df.loc[  # Filter to leave only 'Base Premium' occurences
    df.Peril_Factor.str.contains(raw_struct['bp_name']), :
]).assign(
    # Get the 'Peril' part of 'Peril_Factor'
    Peril=lambda df: df.Peril_Factor.str.replace(raw_struct['bp_name'], "").str.strip()
).Peril.sort_values().to_list()

perils_implied
```

```python
# Check that every 'Peril_Factor' starts with a Peril
if not df_fsets.Peril_Factor.str.startswith(
    tuple(perils_implied)
).all():
    warnings.warn(
        "Implied perils: Not every Peril_Factor starts with a Peril. "
        "Suggests the raw data format is not as expected."
    )
if '' in perils_implied:
    warnings.warn(
        "Implied perils: Empty string has been implied. "
        "Suggests the raw data format is not as expected."
    )
```

```python
# Split out Peril_Factor
df_fsets_split = df_fsets.assign(
    # Split the Peril_Factor column into two
    Factor=lambda df: df.Peril_Factor.str.replace(
            '|'.join(perils_implied), ""
    ).str.strip(),
    Peril=lambda df: df.apply(
        lambda row: row.Peril_Factor.replace(row.Factor, "").strip()
        , axis=1
    )
).drop(columns='Peril_Factor')

df_fsets_split.head()
```

```python
# Get the Base Premiums for all row_IDs and Perils
df_base_prems = df_fsets_split.query(
    # Get only the Base Preimum rows
    f"Factor == '{raw_struct['bp_name']}'"
).assign(
    # Create Peril_Factor combination for column names
    Peril_Factor=lambda df: df.Peril + pf_sep + df.Factor,
    Custom_order=0,  # Will be used later to ensure desired column order
).pivot_table(
    # Pivot to 'Peril_Factor' columns and one row per row_ID
    index=row_id_name,
    columns=['Peril', 'Custom_order', 'Peril_Factor'],
    values='Premium_cumulative'
)

df_base_prems.head()
```

```python
# Warning if the data set is not complete
if df_base_prems.isna().sum().sum() > 0:
    warnings.warn(
        "Base Premiums: Base Premium is missing for some rows and Perils."
        "Suggests the raw data format is not as expected."
    )
```

```python
# Ensure every row_ID has a row for every Peril, Factor combination
# Get the Relativity for all row_ID, Perils and Factors
df_factors = df_fsets_split.query(
    # Get only the Factor rows
    f"Factor != '{raw_struct['bp_name']}'"
).drop(
    columns=['Premium_increment', 'Premium_cumulative']
).set_index(
    # Ensure there is one row for every combination of row_ID, Peril, Factor
    [row_id_name, 'Peril', 'Factor']
).pipe(lambda df: df.reindex(index=pd.MultiIndex.from_product([
    df.index.get_level_values(row_id_name).unique(),
    df.index.get_level_values('Peril').unique(),
    # Include additional factors if desired from the inputs
    set(df.index.get_level_values('Factor').tolist() + include_factors),
], names = df.index.names
))).sort_index().fillna({  # Any new rows need to have Relativity of 1
    'Relativity': 1.,
}).reset_index().assign(
    # Create Peril_Factor combination for column names
    Peril_Factor=lambda df: df.Peril + pf_sep + df.Factor,
    Custom_order=1
).pivot_table(
    # Pivot to 'Peril_Factor' columns and one row per row_ID
    index=row_id_name,
    columns=['Peril', 'Custom_order', 'Peril_Factor'],
    values='Relativity'
)

df_factors.head()
```

```python
# Checks
if not df_factors.apply(lambda col: (col > 0)).all().all():
    warnings.warn(
        "Factor relativities: At least one relativity is below zero."
    )
```

```python
# Combine Base Premium and Factors columns
df_base_factors = df_base_prems.merge(
    df_factors,
    how='inner', left_index=True, right_index=True
).pipe(
    # Sort columns (uses 'Custom_order')
    lambda df: df[df.columns.sort_values()]
)

# Drop unwanted levels of the column MultiIndex
# Possible to do this following in a chain, but much to complicated
# See 'Chained drop a column MultiIndex level' in 'Unused rough work'
df_base_factors.columns = df_base_factors.columns.get_level_values('Peril_Factor')

df_base_factors.head()
```

```python
# Join back on to stem section
df_formatted = df_stem.merge(
    df_base_factors,
    how='left', left_index=True, right_index=True
).fillna(0.)  # The only mising values are from 'error' rows

df_formatted.iloc[:10,:20]
```

## Output to CSV

```python
# Save it
df_formatted.to_csv(
    out_filepath,
    sep=file_delimiter, index=True
)
print("Output saved")
```

### Reload the spreadsheet to check it worked

```python
# Check it worked
df_reload = pd.read_csv(
    out_filepath,
    index_col=0, sep=file_delimiter
)

df_reload.head()
```

```python
assert (df_formatted.dtypes == df_reload.dtypes).all()
assert df_reload.shape == df_formatted.shape
assert (df_formatted.index == df_reload.index).all()
assert df_formatted.iloc[:,1:].apply(
    lambda col: np.abs(col - df_reload[col.name]) < 1e-10
).all().all()
print("Correct: The reloaded values are equal, up to floating point tolerance")
```

## Load expected output to check it is as expected

```python
# Location of sheet of expected results
expected_filepath = raw_data_folder_path / 'minimal01_expected_output.csv'
```

```python
df_expected = None
for encoding in input_file_encodings:
    try:
        df_expected = pd.read_csv(
            expected_filepath,
            index_col=0, sep=file_delimiter,
            encoding=encoding
        ).apply(lambda col: (
            col.astype('float') 
            if np.issubdtype(col.dtype, np.number)
            else col
        ))
        # print(f"'{encoding}': Success")  # Used for debugging only
        break
    except UnicodeDecodeError:
        # print(f"'{encoding}': Fail")  # Used for debugging only
        pass
if df_expected is None:
    raise IOError(
        "\n\tload_formatted_file: pandas.read_csv() failed."
        f"\n\tFile cannot be read with any of the encodings: {input_file_encodings}"
    )

df_expected.head()
```

```python
assert (df_formatted.dtypes == df_expected.dtypes).all()
assert df_expected.shape == df_formatted.shape
assert (df_formatted.index == df_expected.index).all()
assert df_formatted.iloc[:,1:].apply(
    lambda col: np.abs(col - df_expected[col.name]) < 1e-10
).all().all()
print("Correct: The reloaded values are equal, up to floating point tolerance")
```

<div align="right" style="text-align: right"><a href="#Contents">Back to Contents</a></div>

# Using the functions

```python
help(PCon.convert)
```

```python
# Run with default arguments
in_filepath = raw_data_folder_path / 'minimal01_input.csv'
out_filepath = 'formatted_data.csv'
res_filepath = PCon.convert(in_filepath, out_filepath)
```

```python
# Run the pipeline manually to check
# Load raw data
df_raw = PCon.read_raw_data(in_filepath)
# Get converted DataFrame
df_formatted = PCon.convert_df(df_raw)

df_formatted.head()
```

```python
# Reload resulting data from workbook
df_reload = PCon.load_formatted_file(res_filepath)

# Check it matches expectations
if PCon.formatted_dfs_are_equal(df_formatted, df_reload):
    print("Correct: The reloaded values are equal, up to floating point tolerance")
```

```python
# Check against expected output from manually created worksheet
expected_filepath = raw_data_folder_path / 'minimal01_expected_output.csv'
df_expected = PCon.load_formatted_file(expected_filepath)

# Check it matches expectations
if PCon.formatted_dfs_are_equal(df_formatted, df_expected):
    print("Correct: The reloaded values are equal, up to floating point tolerance")
```

```python
# Delete the results file
res_filepath.unlink()
print("Workspace restored")
```

<div align="right" style="text-align: right"><a href="#Contents">Back to Contents</a></div>
