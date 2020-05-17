"""Tests on the Command Line Interface (CLI)"""
# pylint: disable=invalid-name

#########
# Setup #
#########
# Import external modules
import pytest
from click.testing import CliRunner

# Import project modules
import premierconverter as PCon
from .conftest import generate_input_data_csv

####################
# Version and help #
####################
def test_CLI00_version():
    """Check that the version option is available and correct"""
    # When: Submit the 'version' option to the CLI
    runner = CliRunner()
    result = runner.invoke(PCon.cli, ['--version'])

    # Then: Returns the correct version
    assert result.exit_code == 0
    assert result.output == f"cli, version {PCon.__version__}\n"
    print("Correct: Version is available from the CLI")

def test_CLI01_help():
    """Check that the help available and covers the CLI options"""
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

###########################
# CLI succeeding examples #
###########################
def test_CLI10_default_arguments(tmp_dir_path, input_rows_lst, df_expected_tests):
    """Default arguments"""
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

@pytest.mark.parametrize("nrows", [None, 2, 4, 100])
def test_CLI11_nrows(tmp_dir_path, input_rows_lst, df_expected_tests, nrows):
    """Give the argument to limit the number of rows"""
    # Setup
    in_filepath = tmp_dir_path / 'tmp_input.csv'
    out_filepath = tmp_dir_path / 't05_output.csv'

    # Given: Input data
    _ = generate_input_data_csv(input_rows_lst, in_filepath)

    # When: We run the CLI with option for limited number of rows
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

#######################
# CLI force overwrite #
#######################
def test_CLI20_force_overwrite(tmp_dir_path, input_rows_lst, df_expected_tests):
    """
    Check that an existing file will not be overwritten unless force
    is explicitly stated.
    """
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
    assert 'File already exists' in str(result.exception)  # The error message contains is helpful..
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

#########################
# CLI invalid arguments #
#########################
def test_CLI30_missing_all_args(tmp_dir_path):
    """Error thrown when no arguments are passed to the CLI"""
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

def test_CLI31_missing_out_filepath_arg(tmp_dir_path):
    """Error thrown when giving a correct input file but no output file"""
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

def test_CLI32_input_file_does_not_exist(tmp_dir_path):
    """Error thrown when there is no file at the `in_filepath` location"""
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
