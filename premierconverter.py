"""
Automate the conversion of raw data into a specified format of data to make it more usable
"""
# pylint: disable=bad-continuation, useless-return

__version__ = '0.3.1'   # Ensure this is kept in-sync with VERSION in the SETUP.PY

#########
# Setup #
#########
# Import built-in modules
from pathlib import Path
import warnings

# Import external modules
import numpy as np
import pandas as pd
import click

# Configuration variables for the expected format and structure of the data
ACCEPTED_FILE_EXTENSIONS = ['.csv', '', '.txt']
INPUT_FILE_ENCODINGS = ['utf-8', 'latin-1', 'ISO-8859-1']

RAW_STRUCT = {
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

# Output variables, considered to be constants
# Column name of the row IDs
ROW_ID_NAME = "Ref_num"

######################
# Workflow functions #
######################
def validate_input_options(in_filepath):
    """Checks on in_filepath"""
    # Ensure inputs are correct format
    in_filepath = Path(in_filepath)

    # Checks the file exists and is an Excel file
    if not in_filepath.is_file():
        raise FileNotFoundError(
            "\n\tin_filepath: There is no file at the input location:"
            f"\n\t'{in_filepath.absolute()}'"
            "\n\tCannot read the input data"
        )
    if in_filepath.suffix.lower() not in ACCEPTED_FILE_EXTENSIONS:
        warnings.warn(
            f"in_filepath: The input file extension '{in_filepath.suffix}' "
            f"is not one of the recognised file extensions {ACCEPTED_FILE_EXTENSIONS}"
        )
    return None


def validate_output_options(out_filepath, force_overwrite=False):
    """Checks on out_filepath"""
    # Ensure inputs are correct format
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
    if out_filepath.suffix not in ACCEPTED_FILE_EXTENSIONS:
        warnings.warn(
            f"out_filepath: The output file extension '{out_filepath.suffix}' "
            f"is not one of the recognised file extensions {ACCEPTED_FILE_EXTENSIONS}",
        )
    return None


def read_raw_data(in_filepath, nrows=None, file_delimiter=','):
    """
    Load data from file

    in_filepath: Location of the file to read
    nrows: Maximum number of rows to read
    file_delimiter: Character that separates input values in lines
    Returns: The loaded DataFrame, if it is successful
    """
    df_raw = None
    for encoding in INPUT_FILE_ENCODINGS:
        try:
            df_raw = pd.read_csv(
                in_filepath,
                header=None, index_col=0, nrows=nrows,
                sep=file_delimiter, encoding=encoding
            ).rename_axis(index=ROW_ID_NAME)
            # print(f"'{encoding}': Success")  # Used for debugging only
            break
        except UnicodeDecodeError:
            # print(f"'{encoding}': Fail")  # Used for debugging only
            pass
    if df_raw is None:
        raise IOError(
            "\n\tread_raw_data: pandas.read_csv() failed."
            f"\n\tFile cannot be read with any of the encodings: {INPUT_FILE_ENCODINGS}"
        )
    return df_raw


def validate_raw_data(df_raw, file_delimiter=','):
    """Checks on the loaded raw data"""
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
            f"Raw data: Row identifiers '{ROW_ID_NAME}' are not unique. "
            "This may lead to unexpected results."
        )
    return None


# Helper functions to remove unwanted values
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
    return res


def trim_na_cols(df):  # pylint: disable=invalid-name
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


def remove_unwanted_values(df_raw):
    """
    Set unwanted values to NaN, remove surplus columns
    (with all missing values) from the right, and re-cast
    columns to numeric if possible.

    Returns: Adjusted DataFrame
    """
    df_trimmed = df_raw.apply(
        set_na_after_val, match_val=RAW_STRUCT['stop_row_at'], axis=1
    ).pipe(trim_na_cols).apply(pd.to_numeric, errors='ignore')
    return df_trimmed


def validate_trimmed_data(df_trimmed):
    """Checks on the trimmed data"""
    # Check it is as expected
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
    return None


def get_stem_columns(df_trimmed):
    """Select and format the stem columns from the raw data"""
    df_stem = df_trimmed.iloc[
        :, RAW_STRUCT['stem']['chosen_cols']
    ].pipe(  # Rename the columns
        lambda df: df.rename(columns=dict(zip(
            df.columns,
            RAW_STRUCT['stem']['col_names']
        )))
    )
    return df_stem


def validate_stem_columns(df_stem):
    """Checks on the selected stem columns"""
    if not (
        df_stem.dtypes == RAW_STRUCT['stem']['col_types']
    ).all():
        warnings.warn(
            "Stem columns: Unexpected column data types"
            f"\n\tExepcted: {RAW_STRUCT['stem']['col_types']}"
            f"\n\tActual:   {df_stem.dtypes.tolist()}"
        )
    return None


def get_factor_sets(df_trimmed):
    """Concatenate the columns in the raw data that consist of the factor sets"""
    df_fsets = pd.concat([
        # For each of the factor sets of columns
        df_trimmed.iloc[  # Select the columns
            :, fset_start_col:(fset_start_col + RAW_STRUCT['f_set']['ncols'])
        ].dropna(  # Remove rows that have all missing values
            how="all"
        ).pipe(lambda df: df.rename(columns=dict(zip(  # Rename columns
            df.columns, RAW_STRUCT['f_set']['col_names']
        )))).reset_index()  # Get row_ID as a column

        for fset_start_col in range(
            RAW_STRUCT['stem']['ncols'], df_trimmed.shape[1], RAW_STRUCT['f_set']['ncols']
        )
    ], sort=False).reset_index(drop=True)  # Best practice to ensure a unique index
    return df_fsets


def validate_factor_sets(df_fsets):
    """Checks on concatenated factor sets columns"""
    if not (
        df_fsets[RAW_STRUCT['f_set']['col_names']].dtypes ==
        RAW_STRUCT['f_set']['col_types']
    ).all():
        warnings.warn(
            "Factor sets columns: Unexpected column data types"
            f"\n\tExepcted: {RAW_STRUCT['f_set']['col_types']}"
            f"\n\tActual:   {df_fsets[RAW_STRUCT['f_set']['col_names']].dtypes.tolist()}"
        )
    return None


def get_implied_perils(df_fsets):
    """Get all perils in data set by looking at occurences of 'Base Premium'"""
    perils_implied = df_fsets.Peril_Factor.drop_duplicates(
        # Get only unique 'Peril_Factor' combinations
    ).to_frame().pipe(lambda df: df.loc[  # Filter to leave only 'Base Premium' occurences
        df.Peril_Factor.str.contains(RAW_STRUCT['bp_name']), :
    ]).assign(
        # Get the 'Peril' part of 'Peril_Factor'
        Peril=lambda df: df.Peril_Factor.str.replace(RAW_STRUCT['bp_name'], "").str.strip()
    ).Peril.sort_values().to_list()
    return perils_implied


def validate_peril_factors(df_fsets, perils_implied):
    """Checks on implied perils deduced from the factor sets"""
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
    return None


def split_peril_factor(df_fsets, perils_implied):
    """Split the Peril_Factor column into two"""
    df_fsets_split = df_fsets.assign(
        Factor=lambda df: df.Peril_Factor.str.replace(
            '|'.join(perils_implied), ""
        ).str.strip(),
        Peril=lambda df: df.apply(
            lambda row: row.Peril_Factor.replace(row.Factor, "").strip()
            , axis=1
        )
    ).drop(columns='Peril_Factor')
    return df_fsets_split


def get_base_prems(df_fsets_split, pf_sep="_"):
    """
    Get the Base Premiums for all row_IDs and Perils
    pf_sep: Seperator for Peril_Factor column names in output
    """
    df_base_prems = df_fsets_split.query(
        # Get only the Base Preimum rows
        f"Factor == '{RAW_STRUCT['bp_name']}'"
    ).assign(
        # Create Peril_Factor combination for column names
        Peril_Factor=lambda df: df.Peril + pf_sep + df.Factor,
        Custom_order=0,  # Will be used later to ensure desired column order
    ).pivot_table(
        # Pivot to 'Peril_Factor' columns and one row per row_ID
        index=ROW_ID_NAME,
        columns=['Peril', 'Custom_order', 'Peril_Factor'],
        values='Premium_cumulative'
    )
    return df_base_prems


def validate_base_prems(df_base_prems):
    """Checks on formatted Base Premiums"""
    if df_base_prems.isna().sum().sum() > 0:
        warnings.warn(
            "Base Premiums: Base Premium is missing for some rows and Perils. "
            "Suggests the raw data format is not as expected."
        )
    return None


def get_all_factor_relativities(
    df_fsets_split,
    include_factors=None,
    pf_sep='_'
):
    """
    Ensure every row_ID has a row for every Peril, Factor combination
    Get the Relativity for all row_ID, Perils and Factors

    include_factors: If any of the factors in this list are not implied
        in the data, then such factors are also returned in the output.
    pf_sep: Seperator for Peril_Factor column names in output
    """
    # Set defaults
    include_factors = None
    if include_factors is None:
        include_factors = []

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
    ], names=df.index.names)
    )).sort_index().fillna({  # Any new rows need to have Relativity of 1
        'Relativity': 1.,
    }).reset_index().assign(
        # Create Peril_Factor combination for column names
        Peril_Factor=lambda df: df.Peril + pf_sep + df.Factor,
        Custom_order=1
    ).pivot_table(
        # Pivot to 'Peril_Factor' columns and one row per row_ID
        index=ROW_ID_NAME,
        columns=['Peril', 'Custom_order', 'Peril_Factor'],
        values='Relativity'
    )
    return df_factors


def validate_relativities(df_factors):
    """Checks on formatted Relativities"""
    if not df_factors.apply(lambda col: (col > 0)).all().all():
        warnings.warn(
            "Factor relativities: At least one relativity is below zero."
        )
    return None


def get_base_and_factors(df_base_prems, df_factors):
    """Combine Base Premium and Factors columns"""
    df_base_factors = df_base_prems.merge(
        df_factors,
        how='inner', left_index=True, right_index=True
    ).pipe(
        # Sort columns (uses 'Custom_order')
        lambda df: df[df.columns.sort_values()]
    )

    # Drop unwanted levels of the column MultiIndex
    df_base_factors.columns = df_base_factors.columns.get_level_values('Peril_Factor')
    return df_base_factors


def join_stem_to_base_factors(df_stem, df_base_factors):
    """Join formatted stem section to combined Base Premium and Factors columns"""
    df_formatted = df_stem.merge(
        df_base_factors,
        how='left', left_index=True, right_index=True
    ).fillna(0.)  # The only mising values are from 'error' rows
    return df_formatted


def save_to_csv(df_formatted, out_filepath, file_delimiter=","):
    """Save DataFrame to specified output location"""
    df_formatted.to_csv(
        out_filepath,
        sep=file_delimiter, index=True
    )
    return True


######################
# Pipeline functions #
######################
def convert_df(
    df_raw,
    include_factors=None,
    pf_sep="_",
    with_validation=True,
):
    """
    Convert DataFrame of raw data into a specified format

    include_factors: If any of the factors in this list are not implied
        in the data, then such factors are also returned in the output.
    pf_sep: Seperator for Peril_Factor column names in output.
    with_validation: Set to False to stop optional validation checks from
        running (which might make this function run a little faster).
    """
    # Validate raw data
    if with_validation:
        validate_raw_data(df_raw)

    # Remove unwanted values and resulting empty columns
    df_trimmed = remove_unwanted_values(df_raw)
    if with_validation:
        validate_trimmed_data(df_trimmed)

    # Select and format the stem columns
    df_stem = get_stem_columns(df_trimmed)
    if with_validation:
        validate_stem_columns(df_stem)

    # Select and format the factor set columns
    df_fsets = get_factor_sets(df_trimmed)
    if with_validation:
        validate_factor_sets(df_fsets)
    perils_implied = get_implied_perils(df_fsets)
    if with_validation:
        validate_peril_factors(df_fsets, perils_implied)
    df_fsets_split = split_peril_factor(df_fsets, perils_implied)
    df_base_prems = get_base_prems(df_fsets_split, pf_sep)
    if with_validation:
        validate_base_prems(df_base_prems)
    df_factors = get_all_factor_relativities(
        df_fsets_split, include_factors, pf_sep
    )
    if with_validation:
        validate_relativities(df_factors)
    df_base_factors = get_base_and_factors(df_base_prems, df_factors)

    # Join stem and base and factor columns
    df_formatted = join_stem_to_base_factors(df_stem, df_base_factors)

    return df_formatted


def convert(
    in_filepath,
    out_filepath,
    force_overwrite=False,
    nrows=None,
    file_delimiter=',',
    **kwargs,
):
    """
    Load raw data, convert to specified format, and save result

    in_filepath: Path to file containing a sheet with the raw data
    out_filepath: Path of a file to save the formatted data
        If it does not exist, a new workbook will be created.
        The directory must already exist.

    file_delimiter: Seperator for values in the input and output files
    force_overwrite: Set to True if you want to overwrite an existing file

    nrows: Maximum number of rows to read
    **kwargs: Other arguments to pass to convert_df

    Returns: out_filepath, if it completes successfully
    """
    # Set defaults
    in_filepath = Path(in_filepath)
    out_filepath = Path(out_filepath)

    # Validate function inputs
    validate_input_options(in_filepath)
    validate_output_options(out_filepath, force_overwrite)

    # Load raw data
    df_raw = read_raw_data(in_filepath, nrows, file_delimiter)

    # Get converted DataFrame
    df_formatted = convert_df(df_raw, **kwargs)

    # Save results to a workbook
    if save_to_csv(df_formatted, out_filepath, file_delimiter):
        print(
            f"Output saved here:\t{out_filepath.absolute()}"
        )

    return out_filepath


#######################
# Reloading functions #
#######################
def load_formatted_file(out_filepath, file_delimiter=','):
    """
    Utility function to load data from output file

    *Not* designed to check if there have been any changes since
    the output sheet was created.
    """
    df_reload = None
    for encoding in INPUT_FILE_ENCODINGS:
        try:
            df_reload = pd.read_csv(
                out_filepath,
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
    if df_reload is None:
        raise IOError(
            "\n\tload_formatted_file: pandas.read_csv() failed."
            f"\n\tFile cannot be read with any of the encodings: {INPUT_FILE_ENCODINGS}"
        )
    return df_reload


def formatted_dfs_are_equal(df1, df2, tol=1e-10):
    """Reasonableness checks between two DataFrames of output format"""
    assert df1.shape == df2.shape
    assert (df1.index == df2.index).all()
    assert (df1.dtypes == df2.dtypes).all()
    assert df1.iloc[:, 1:].apply(
        lambda col: np.abs(col - df2[col.name]) < tol
    ).all().all()
    return True


######################
# Make CLI available #
######################
@click.command(context_settings=dict(
    help_option_names=['-h', '--help']
))
@click.version_option(__version__)
@click.argument(
    'in_filepath',
    type=click.Path(exists=True),
    required=True,
    metavar='<input filepath>',
)
@click.argument(
    'out_filepath',
    type=click.Path(),
    required=True,
    metavar='<output filepath>',
)
@click.option(
    '--force', 'force_overwrite',
    is_flag=True,
    help='Overwrite an existing output worksheet.',
)
@click.option(
    '--nrows', '-r', 'nrows',
    type=int, default=None, show_default=True,
    help='Maximum number of rows to read.',
)
@click.option(
    '--sep', '-s', 'file_delimiter',
    type=str, default=",", show_default=True,
    help='Separator for in and out files.',
)
@click.option(
    '--no_checks', '-n', 'no_checks',
    is_flag=True,
    help='Stop optional validation checks from running.',
)
def cli(  # pylint: disable=too-many-arguments
    in_filepath,
    out_filepath,
    # Default values for these arguments are given above
    force_overwrite,
    nrows,
    file_delimiter,
    no_checks,
):
    """
    Load raw data, convert to specified format, and save result

    <input filepath>: Path to file containing the raw data

    <output filepath>: Path where the resulting file should go.
    If it does not exist, a new workbook will be created.
    The directory must already exist.
    """
    # Pass parameters to convert()
    convert(
        in_filepath=in_filepath,
        out_filepath=out_filepath,
        force_overwrite=force_overwrite,
        nrows=nrows,
        file_delimiter=file_delimiter,
        with_validation=not no_checks,
    )
    return None

# Identifying whether this script is being run on Kaggle
# allows a Full Version of the script to be saved
# (i.e. without an error occurring).
ON_KAGGLE = False
if str(Path().absolute()) == '/kaggle/working':
    ON_KAGGLE = True

if __name__ == '__main__':
    if ON_KAGGLE:
        print("Script run complete")
    else:
        cli()  # pylint: disable=no-value-for-parameter
