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
<!-- #endregion -->

<!-- #region _cell_guid="79c7e3d0-c299-4dcb-8224-4455121ee9b0" _uuid="d629ff2d2480ee46fbb7e2d37f6b5fab8052498a" -->
<!-- This table of contents is updated *manually* -->
# Contents
1. [Setup](#Setup): Import packages, Config variables
1. [Variables](#Variables): Raw data structure, Inputs
1. [Workflow](#Workflow): Load raw data, Stem section, Factor sets
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
from openpyxl import __version__ as opyxl_version
from openpyxl import load_workbook

# Import project modules
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
assert opyxl_version == '3.0.3'
print(f'openpyxl version:\t\t{opyxl_version}')
assert PCon.__version__ == '0.1.2'
print(f'premierconverter version:\t{PCon.__version__}')
```

```python
# Output exact environment specification, in case it is needed later
print("Capturing full package environment spec")
print("(But note that not all these packages are required)")
!pip freeze > requirements_Kaggle.txt
!jupyter --version > jupyter_versions.txt
```

```python
# Configuration variables
input_folder_path = Path('/kaggle/input')
raw_data_folder_path = input_folder_path / 'dummy-premier-data-raw'
assert raw_data_folder_path.is_dir()
print("Correct: All locations are available as expected")
```

<div align="right" style="text-align: right"><a href="#Contents">Back to Contents</a></div>

# Variables


## Raw data structure

```python
# Configuration variables for the expected format and structure of the data
excel_extensions = ['.xlsx', '.xls', '.xlsm']

raw_struct = {
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
    'bp_name': 'Base Premium'
}
```

## Parameters

```python
# Include Factors which are not found in the data
include_factors = None
if include_factors is None:
    include_factors = []
    
# Seperator for Peril_Factor column names in output
pf_sep = '_'
```

```python
# Input file location
input_filepath = raw_data_folder_path / 'Premier_Test_dummy_data.xlsx'
input_sheet = None
if input_sheet is None:
    input_sheet = 0

# Checks
input_filepath = Path(input_filepath)
if not input_filepath.is_file():
    raise FileNotFoundError(
        "\n\tinput_filepath: There is no file at the input location:"
        f"\n\t'{input_filepath}'"
        "\n\tCannot read the input data"
    )
if not input_filepath.suffix in excel_extensions:
    warnings.warn(
        f"input_filepath: The input file extension '{input_filepath.suffix}' "
        "is not a recognised Excel extension",
    )
print("Correct: Input file exists and has an Excel extension")
```

```python
# Output file location
out_filepath = 'formatted_dummy_data1.xlsx'
out_sheet_name = 'Sheet2'
force_overwrite = False

# Checks
out_filepath = Path(out_filepath)
xl_writer = pd.ExcelWriter(out_filepath, engine = 'openpyxl')

if not out_filepath.parent.is_dir():
    raise FileNotFoundError(
        f"\n\tout_filepath: The folder of the output file does not exist"
        f"Folder path: '{out_filepath.parent}'"
        "\n\tCreate the output folder before running this command"
    )

if out_filepath.is_file():
    out_workbook = load_workbook(out_filepath)   
    if out_sheet_name in out_workbook.sheetnames and not force_overwrite:
        raise FileExistsError(
            "\n\tOutput options: Sheet already exists at the output location:"
            f"\n\tLocation: '{out_filepath}'"
            f"\n\tSheet name: '{out_sheet_name}'"
            "\n\tIf you want to overwrite it, re-run with `force_overwrite = True`"
        )
    # Set the pandas ExcelWriter to point at this workbook
    xl_writer.book = out_workbook
else:
    if not out_filepath.suffix in excel_extensions:
        warnings.warn(
            f"out_filepath: The output file extension '{out_filepath.suffix}' "
            "is not a recognised Excel extension",
        )

print("Correct: A suitable location for output has been chosen")
```

<div align="right" style="text-align: right"><a href="#Contents">Back to Contents</a></div>

# Workflow


## Load raw data

```python
df_raw = pd.read_excel(
    input_filepath, sheet_name=input_sheet,
    engine="openpyxl",  # As per: https://stackoverflow.com/a/60709194
    header=None, index_col=0,
).rename_axis(index="Ref_num")

df_raw.head()
```

```python
# Check it is as expected
if not (
    (df_raw.shape[1] - raw_struct['stem']['ncols']) 
    % raw_struct['f_set']['ncols'] == 0
):
    warnings.warn(
        f"Raw data: Incorrect number of columns: {df_raw.shape[1]}"
        "\n\tThere should be: 1 for index, "
        f"{raw_struct['stem']['ncols']} for stem section, "
        f"and by a multiple of {raw_struct['f_set']['ncols']} for factor sets"
    )
```

## Stem section

```python
# Get the stem section of columns
df_stem = df_raw.iloc[
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
    df_raw.iloc[  # Select the columns
        :, fset_start_col:(fset_start_col + raw_struct['f_set']['ncols'])
    ].dropna(  # Remove rows that have all missing values
        how="all"
    ).pipe(lambda df: df.rename(columns=dict(zip(  # Rename columns
        df.columns, raw_struct['f_set']['col_names']
    )))).reset_index()  # Get 'Ref_num' as a column

    for fset_start_col in range(
        raw_struct['stem']['ncols'], df_raw.shape[1], raw_struct['f_set']['ncols']
    )
], sort=False)

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
        f"\n\tExepcted: {raw_struct['f_set']['col_types']}"
        f"\n\tActual:   {df_fsets[raw_struct['f_set']['col_names']].dtypes.tolist()}"
    )
```

```python
# Get all perils in data set by looking at occurences of 'Base Premium'
perils_implied = df_fsets.Peril_Factor.drop_duplicates(  # Get only unique 'Peril_Factor' combinations
).to_frame().assign(  # Filter to leave only 'Base Premium' occurences
    row_to_keep=lambda df: df.Peril_Factor.str.contains(raw_struct['bp_name'])
).query('row_to_keep').drop(columns='row_to_keep').assign(
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
# Get the Base Premiums for all Ref_nums and Perils
df_base_prems = df_fsets_split.query(
    # Get only the Base Preimum rows
    f"Factor == '{raw_struct['bp_name']}'"
).assign(
    # Create Peril_Factor combination for column names
    Peril_Factor=lambda df: df.Peril + pf_sep + df.Factor,
    Custom_order=0,  # Will be used later to ensure desired column order
).pivot_table(
    # Pivot to 'Peril_Factor' columns and 'Ref_num' rows
    index='Ref_num',
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
# Ensure every Ref_num has a row for every Peril, Factor combination
# Get the Relativity for all Ref_nums, Perils and Factors
df_factors = df_fsets_split.query(
    # Get only the Factor rows
    f"Factor != '{raw_struct['bp_name']}'"
).drop(
    columns=['Premium_increment', 'Premium_cumulative']
).set_index(
    # Ensure there is one row for every combination of Ref_num, Peril, Factor
    ['Ref_num', 'Peril', 'Factor']
).pipe(lambda df: df.reindex(index=pd.MultiIndex.from_product([
    df.index.get_level_values('Ref_num').unique(),
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
    # Pivot to 'Peril_Factor' columns and 'Ref_num' rows
    index='Ref_num',
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

## Output to Excel

```python
# Save it
df_formatted.to_excel(xl_writer, sheet_name=out_sheet_name)
xl_writer.save()
xl_writer.close()
print("Output saved")
```

### Reload the spreadsheet to check it worked

```python
# Check it worked
df_reload = pd.read_excel(
    out_filepath, sheet_name=out_sheet_name,
    engine="openpyxl",  # As per: https://stackoverflow.com/a/60709194
    index_col=[0],
).apply(lambda col: (
    col if col.name in raw_struct['stem']['col_names'][0]
    else col.astype('float')
))

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

<div align="right" style="text-align: right"><a href="#Contents">Back to Contents</a></div>

# Using the functions

```python
help(PCon.convert)
```

```python
# Run with default arguments
input_filepath = raw_data_folder_path / 'Premier_Test_dummy_data.xlsx'
out_filepath, out_sheet_name = PCon.convert(input_filepath)
```

```python
# Run the pipeline manually to check
# Load raw data
df_raw = PCon.read_raw_data(input_filepath)
# Get converted DataFrame
df_formatted = PCon.convert_df(df_raw)

df_formatted.head()
```

```python
# Reload resulting data from workbook
df_reload = PCon.load_formatted_spreadsheet(out_filepath, out_sheet_name)

# Check it matches expectations
assert (df_formatted.dtypes == df_reload.dtypes).all()
assert df_reload.shape == df_formatted.shape
assert (df_formatted.index == df_reload.index).all()
assert df_formatted.iloc[:,1:].apply(
    lambda col: np.abs(col - df_reload[col.name]) < 1e-10
).all().all()
print("Correct: The reloaded values are equal, up to floating point tolerance")
```

```python
# Delete the results workbook
out_filepath.unlink()
print("Workspace restored")
```

<div align="right" style="text-align: right"><a href="#Contents">Back to Contents</a></div>

# Unused rough work


## Replace multiple string terms

```python
import functools

def multi_replace(base_str, replacement_dict):
    """
    Run str.replace() multiple times to replace multiple terms.
    
    base_str: Starting string from which you want to replace substrings
    replacement_dict: Each item of the dictionary is {string_to_replace: replacement_string}
    """
    return(functools.reduce(
        lambda current_str, replace_pair: current_str.replace(*replace_pair),
        {key: str(val) for key, val in replacement_dict.items()}.items(),
        base_str
    ))
```

## Chained drop a column MultiIndex level

```python
# df_base_factors = df_base_prems.merge(
#     df_factors,
#     how='inner', left_index=True, right_index=True
# ).pipe(
#     lambda df: df[df.columns.sort_values()]
# ).rename(
#     columns=lambda x: '', level = 'Peril'
# ).rename(
#     columns=lambda x: '', level = 'Custom_order'
# ).stack([0,1]).reset_index(level=[1,2], drop=True)

# df_base_factors.head()
```

<div align="right" style="text-align: right"><a href="#Contents">Back to Contents</a></div>
