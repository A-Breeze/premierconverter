"""Tests on the Command Line Interface (CLI)"""
#########
# Setup #
#########
# Import external modules
from click.testing import CliRunner

# Import project modules
import premierconverter as PCon
from . import conftest

###########################
# CLI succeeding examples #
###########################
def test_CLI_default_arguments(tmp_dir_path, df_expected_tests):  # pylint: disable=invalid-name
    """Default arguments"""
    # Given: Input data and output location
    in_filepath = tmp_dir_path / 'tmp_input.csv'
    out_filepath = tmp_dir_path / 't04_output.csv'
    _ = conftest.create_input_data_csv(in_filepath)

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
