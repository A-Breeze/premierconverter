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

**Important note**: The data used in this notebook has been randomised and all names have been masked so they can be used for training purposes. No data is committed to the project repo. This notebook is for development purposes only.

This notebook is available in the following locations. These versions are kept in sync *manually* - there should not be discrepancies, but it is possible.
- On Kaggle: <https://www.kaggle.com/btw78jt/data-conversion-challenge-202004>
- In the GitHub project repo: <https://github.com/A-Breeze/premierconverter>. See the `README.md` for further instructions, and the associated `simulate_dummy_data` notebook to generate the dummy data that is used for this notebook.
<!-- #endregion -->

<!-- #region _cell_guid="79c7e3d0-c299-4dcb-8224-4455121ee9b0" _uuid="d629ff2d2480ee46fbb7e2d37f6b5fab8052498a" -->
<!-- This table of contents is updated *manually* -->
# Contents
1. [Setup](#Setup): Import packages, Config variables
1. [Variables](#Variables): Raw data structure, Inputs
1. [Workflow](#Workflow): Load raw data, Remove unwanted extra values, Stem section, Factor sets, Output to CSV, Load expected output to check it is as expected
1. [Using the functions](#Using-the-functions): Default arguments, Limited rows
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

ON_KAGGLE = False
print("Current working directory: " + str(Path('.').absolute()))
if str(Path('.').absolute()) == '/kaggle/working':
    ON_KAGGLE = True
```

```python
# Import built-in modules
import sys
import platform
import os
import io

# Import external modules
from IPython import __version__ as IPy_version
import numpy as np
import pandas as pd
from click import __version__ as click_version

# Import project modules
if not ON_KAGGLE:
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
# Output exact environment specification, in case it is needed later
if ON_KAGGLE:
    print("Capturing full package environment spec")
    print("(But note that not all these packages are required)")
    !pip freeze > requirements_snapshot.txt
    !jupyter --version > jupyter_versions_snapshot.txt
```

```python
# Configuration variables
if ON_KAGGLE:
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
ACCEPTED_FILE_EXTENSIONS = ['.csv', '', '.txt']
INPUT_FILE_ENCODINGS = ['utf-8', 'latin-1', 'ISO-8859-1']
INPUT_SEPARATOR = ","

RAW_STRUCT = {
    'stop_row_at': 'Total Peril Premium',
    'stem': {
        'ncols': 5,
        'chosen_cols': [0,1],
        'col_names': ['Premier_Test_Status', 'Total_Premium'],
        'col_types': [np.dtype('object'), np.dtype('float')],
    },
    'f_set': {
        'include_Test_Status': ['Ok'],
        'ncols': 4,
        'col_names': ['Peril_Factor', 'Relativity', 'Premium_increment', 'Premium_cumulative'],
        'col_types': [np.dtype('object')] + [np.dtype('float')] * 3,
    },
    'bp_name': 'Base Premium',
}
TRUNC_AFTER_REGEX = r",\s*{}.*".format(RAW_STRUCT['stop_row_at'])

# Output variables, considered to be constants
# Column name of the row IDs
ROW_ID_NAME = "Ref_num"

OUTPUT_DEFAULTS = {
    'pf_sep': ' ',
    'file_delimiter': ','
}
```

## Parameters

```python
# Include Factors which are not found in the data
include_factors = None
if include_factors is None:
    include_factors = []

# Maximum number of rows to read in
nrows = None
```

```python
# Input file location
in_filepath = raw_data_folder_path / 'minimal_input_adj.csv'

# Checks the file exists and has a recognised extension
in_filepath = Path(in_filepath)
if not in_filepath.is_file():
    raise FileNotFoundError(
        "\n\tin_filepath: There is no file at the input location:"
        f"\n\t'{in_filepath.absolute()}'"
        "\n\tCannot read the input data"
    )
if not in_filepath.suffix.lower() in ACCEPTED_FILE_EXTENSIONS:
    warnings.warn(
        f"in_filepath: The input file extension '{in_filepath.suffix}' "
        f"is not one of the recognised file extensions {ACCEPTED_FILE_EXTENSIONS}"
    )
print("Correct: Input file exists and has a recognised extension")
```

```python
# View the first n raw CSV lines (without loading into a DataFrame)
nlines = 2
lines = []
with in_filepath.open() as f: 
    for line_num in range(nlines):
        lines.append(f.readline())
print(''.join(lines))
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
    if not out_filepath.suffix in ACCEPTED_FILE_EXTENSIONS:
        warnings.warn(
            f"out_filepath: The output file extension '{out_filepath.suffix}' "
            f"is not one of the recognised file extensions {ACCEPTED_FILE_EXTENSIONS}",
        )

print("Correct: A suitable location for output has been chosen")
```

<div align="right" style="text-align: right"><a href="#Contents">Back to Contents</a></div>

# Workflow


## Load raw data

```python
# Load the CSV lines truncated as required
in_lines_trunc_df = None
for encoding in INPUT_FILE_ENCODINGS:
    try:
        in_lines_trunc_df = pd.read_csv(
            in_filepath, header=None, index_col=False,
            nrows=nrows, sep=TRUNC_AFTER_REGEX,
            engine='python', encoding=encoding,
        )
        # print(f"'{encoding}': Success")  # Used for debugging only
        break
    except UnicodeDecodeError:
        # print(f"'{encoding}': Fail")  # Used for debugging only
        pass
if in_lines_trunc_df is None:
    raise IOError(
        "\n\tread_input_lines: pandas.read_csv() failed."
        f"\n\tFile cannot be read with any of the encodings: {INPUT_FILE_ENCODINGS}"
    )

in_lines_trunc_df.head()
```

```python
# Check it worked and is not malformed
if in_lines_trunc_df.shape[0] <= 1:
    warnings.warn(
        "Raw data lines: Only one row of data has been read. "
        "Are you sure you have specified the correct file? "
        "Are rows of data split into lines of the file?"
    )
if not ((
    in_lines_trunc_df.shape[1] == 1
) or (
    in_lines_trunc_df.iloc[:, 1].isna().sum() == in_lines_trunc_df.shape[0]
)):
    warnings.warn(
        "Raw data lines: A line in the input data has more than one match "
        f"to the regex pattern \"{TRUNC_AFTER_REGEX}\". "
        "Are you sure you have specified the correct file?"
    )
```

```python
# Convert to DataFrame
with warnings.catch_warnings():
    # Ignore dtype warnings at this point, because we check them later on (after casting)
    warnings.filterwarnings(
        "ignore", message='.*Specify dtype option on import or set low_memory=False',
        category=pd.errors.DtypeWarning,
    )
    with io.StringIO('\n'.join(in_lines_trunc_df[0])) as in_lines_trunc_stream:
        df_trimmed = pd.read_csv(
            in_lines_trunc_stream, header=None, index_col=0, sep=INPUT_SEPARATOR,
            names=range(in_lines_trunc_df[0].str.count(INPUT_SEPARATOR).max() + 1),
        ).rename_axis(index=PCon.ROW_ID_NAME)

df_trimmed.head()
```

```python
# Check it is as expected and not malformed
if not df_trimmed.index.is_unique:
    warnings.warn(
        f"Trimmed data: Row identifiers '{ROW_ID_NAME}' are not unique. "
        "This may lead to unexpected results."
    )
if not (
    # At least the stem columns and one factor set column
    df_trimmed.shape[1] >= 
    RAW_STRUCT['stem']['ncols'] + 1 * RAW_STRUCT['f_set']['ncols']
) or not (
    # Stem columns plus a multiple of factor set columns
    (df_trimmed.shape[1] - RAW_STRUCT['stem']['ncols']) 
    % RAW_STRUCT['f_set']['ncols'] == 0
):
    warnings.warn(
        "Trimmed data: Incorrect number of columns with relevant data: "
        f"{df_trimmed.shape[1] + 1}"
        "\n\tThere should be: 1 for index, "
        f"{RAW_STRUCT['stem']['ncols']} for stem section, "
        f"and by a multiple of {RAW_STRUCT['f_set']['ncols']} for factor sets"
    )
```

## Stem section

```python
# Get the stem section of columns
df_stem = df_trimmed.iloc[
    :, RAW_STRUCT['stem']['chosen_cols']
].pipe(  # Rename the columns
    lambda df: df.rename(columns=dict(zip(
        df.columns, 
        RAW_STRUCT['stem']['col_names']
    )))
)

df_stem.head()
```

```python
# Checks
if not (
    df_stem.dtypes == RAW_STRUCT['stem']['col_types']
).all():
    warnings.warn(
        "Stem columns: Unexpected column data types"
        f"\n\tExepcted: {RAW_STRUCT['stem']['col_types']}"
        f"\n\tActual:   {df_stem.dtypes.tolist()}"
    )
```

## Factor sets

```python
# Combine the rest of the DataFrame into one
df_fsets = pd.concat([
    # For each of the factor sets of columns
    df_trimmed.loc[  # Filter to only the valid rows
        df_trimmed[1].isin(RAW_STRUCT['f_set']['include_Test_Status'])
    ].iloc[  # Select the columns
        :, fset_start_col:(fset_start_col + RAW_STRUCT['f_set']['ncols'])
    ].dropna(  # Remove rows that have all missing values
        how="all"
    ).pipe(lambda df: df.rename(columns=dict(zip(  # Rename columns
        df.columns, RAW_STRUCT['f_set']['col_names']
    )))).reset_index()  # Get row_ID as a column

    for fset_start_col in range(
        RAW_STRUCT['stem']['ncols'], df_trimmed.shape[1], RAW_STRUCT['f_set']['ncols']
    )
], sort=False).apply(  # Where possible, convert object columns to numeric dtype
    pd.to_numeric, errors='ignore'
).reset_index(drop=True)  # Best practice to ensure a unique index

df_fsets.head()
```

```python
# Checks
if not (
    df_fsets[RAW_STRUCT['f_set']['col_names']].dtypes == 
    RAW_STRUCT['f_set']['col_types']
).all():
    warnings.warn(
        "Factor sets columns: Unexpected column data types"
        f"\n\tExpected: {RAW_STRUCT['f_set']['col_types']}"
        f"\n\tActual:   {df_fsets[RAW_STRUCT['f_set']['col_names']].dtypes.tolist()}"
    )
```

```python
perils_implied = df_fsets.Peril_Factor.drop_duplicates(  # Get only unique 'Peril_Factor' combinations
).to_frame().pipe(lambda df: df.loc[  # Filter to leave only 'Base Premium' occurences
    df.Peril_Factor.str.contains(RAW_STRUCT['bp_name']), :
]).assign(
    # Get the 'Peril' part of 'Peril_Factor'
    Peril=lambda df: df.Peril_Factor.str.replace(RAW_STRUCT['bp_name'], "").str.strip()
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
    f"Factor == '{RAW_STRUCT['bp_name']}'"
).assign(
    # Create Peril_Factor combination for column names
    Peril_Factor=lambda df: df.Peril + OUTPUT_DEFAULTS['pf_sep'] + df.Factor,
    Custom_order=0,  # Will be used later to ensure desired column order
).pivot_table(
    # Pivot to 'Peril_Factor' columns and one row per row_ID
    index=ROW_ID_NAME,
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
    f"Factor != '{RAW_STRUCT['bp_name']}'"
).drop(
    columns=['Premium_increment', 'Premium_cumulative']
).set_index(
    # Ensure there is one row for every combination of row_ID, Peril, Factor
    [ROW_ID_NAME, 'Peril', 'Factor']
).pipe(lambda df: df.reindex(index=pd.MultiIndex.from_product([
    df.index.get_level_values(ROW_ID_NAME).unique(),
    df.index.get_level_values('Peril').unique(),
    # Include additional factors if desired from the inputs
    set(df.index.get_level_values('Factor').tolist() + include_factors),
], names = df.index.names
))).sort_index().fillna({  # Any new rows need to have Relativity of 1
    'Relativity': 1.,
}).reset_index().assign(
    # Create Peril_Factor combination for column names
    Peril_Factor=lambda df: df.Peril + OUTPUT_DEFAULTS['pf_sep'] + df.Factor,
    Custom_order=1
).pivot_table(
    # Pivot to 'Peril_Factor' columns and one row per row_ID
    index=ROW_ID_NAME,
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
    out_filepath, sep=OUTPUT_DEFAULTS['file_delimiter'], index=True
)
print("Output saved")
```

### Reload the spreadsheet to check it worked

```python
# Check it worked
df_reload = pd.read_csv(
    out_filepath, index_col=0, sep=OUTPUT_DEFAULTS['file_delimiter'],
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
expected_filepath = raw_data_folder_path / 'minimal_expected_output_5.csv'
```

```python
df_expected = None
for encoding in INPUT_FILE_ENCODINGS:
    try:
        df_expected = pd.read_csv(
            expected_filepath,
            index_col=0, sep=OUTPUT_DEFAULTS['file_delimiter'],
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
        f"\n\tFile cannot be read with any of the encodings: {INPUT_FILE_ENCODINGS}"
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
## Default arguments

```python
help(PCon.convert)
```

```python
#in_filepath = raw_data_folder_path / 'minimal_input_adj.csv'
out_filepath = 'formatted_data.csv'
res_filepath = PCon.convert(in_filepath, out_filepath)
```

```python
# Run the pipeline manually to check
# Load raw data
in_lines_trunc_df = PCon.read_input_lines(in_filepath)
PCon.validate_input_lines_trunc(in_lines_trunc_df)
df_trimmed = PCon.split_lines_to_df(in_lines_trunc_df)
# Get converted DataFrame
df_formatted = PCon.convert_df(df_trimmed)

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
expected_filepath = raw_data_folder_path / 'minimal_expected_output_5.csv'
df_expected = PCon.load_formatted_file(expected_filepath)

# Check it matches expectations
if PCon.formatted_dfs_are_equal(df_reload, df_expected):
    print("Correct: The reloaded values are equal, up to floating point tolerance")
```

```python
# Delete the results file
res_filepath.unlink()
print("Workspace restored")
```

## Limited rows

```python
nrows = 2  # Choose a specific number for which the expected results have been created: 2, 4 or 5
in_filepath = raw_data_folder_path / 'minimal_input_adj.csv'
out_filepath = f'formatted_data_{nrows}.csv'
res_filepath = PCon.convert(in_filepath, out_filepath, nrows = nrows)

# Check against expected output from manually created worksheet
expected_filepath = raw_data_folder_path / f'minimal_expected_output_{nrows}.csv'
df_expected = PCon.load_formatted_file(expected_filepath)
df_reload = PCon.load_formatted_file(res_filepath)

# Check it matches expectations
if PCon.formatted_dfs_are_equal(df_reload, df_expected):
    print("Correct: The reloaded values are equal, up to floating point tolerance")

# Delete the results file
res_filepath.unlink()
print("Workspace restored")
```

## Limited rows with included factors

```python
nrows = 2
include_factors = ['NewFact', 'SomeFact']
in_filepath = raw_data_folder_path / 'minimal_input_adj.csv'
out_filepath = f'formatted_data_2_all_facts.csv'
res_filepath = PCon.convert(in_filepath, out_filepath, nrows=nrows, include_factors=include_factors)

# Check against expected output from manually created worksheet
expected_filepath = raw_data_folder_path / 'minimal_expected_output_2_all_facts.csv'  # Specifically created for this test
df_expected = PCon.load_formatted_file(expected_filepath)
df_reload = PCon.load_formatted_file(res_filepath)

# Check it matches expectations
if PCon.formatted_dfs_are_equal(df_reload, df_expected):
    print("Correct: The reloaded values are equal, up to floating point tolerance")

# Delete the results file
res_filepath.unlink()
print("Workspace restored")
```

Further connotations are tested in the package's automated test suite.


<div align="right" style="text-align: right"><a href="#Contents">Back to Contents</a></div>
