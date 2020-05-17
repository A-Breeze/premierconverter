"""Unit tests on input/output file functions"""
# pylint: disable=assignment-from-none

#########
# Setup #
#########
# Import external modules
import pytest

# Import project modules
import premierconverter as PCon

##########################
# in_filepath validation #
##########################
def test_io00_in_filepath_does_not_exist(tmp_dir_path):
    """Check an error is raised if there is no file at the in_filepath"""
    # Given: Input file does not exist
    in_filepath = tmp_dir_path / 'foo.csv'
    # When: Attempt to validate the filepath
    # Then: Error thrown
    with pytest.raises(FileNotFoundError) as err:
        PCon.validate_input_options(in_filepath)
    assert err is not None  # An error was thrown...
    assert isinstance(err.value, FileNotFoundError)  # ...of this specific type
    assert (  # The error message is helpful...
        'There is no file at the input location' in str(err.value)
    )
    assert str(in_filepath.absolute()) in str(err.value)  # ...and contains the filepath
    print("Correct: Helpful error message was thrown")

def test_io01_in_filepath_unrecognised_extension(tmp_dir_path):
    """Check an error is raised if there is no file at the in_filepath"""
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

@pytest.mark.parametrize("filename", ['foo.csv', 'fo o', 'foo.CsV', '01_f.tXt'])
def test_io02_in_filepath_valid(tmp_dir_path, filename):
    """Check the function correctly accepts a valid in_filepath"""
    # Given: Input file exists and has a recognised extension
    in_filepath = tmp_dir_path / filename
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

##########################
# out_filepath validation #
##########################
@pytest.mark.parametrize("filename", ['foo.csv', 'fo o', 'foo.CsV', '01_f.tXt'])
def test_io10_out_filepath_already_exists(tmp_dir_path, filename):
    """
    Check an error is raised if there is already a file at out_filepath...
    ...but you can stop this error by explicitly passing force_overwrite.
    """
    # Given: Output file already exists
    out_filepath = tmp_dir_path / filename
    out_filepath.write_text("Some text")
    assert out_filepath.is_file()
    # When: Attempt to validate the filepath (and no force_overwrite passed)
    # Then: Error thrown
    with pytest.raises(FileExistsError) as err:
        PCon.validate_output_options(out_filepath)
    assert err is not None  # An error was thrown...
    assert isinstance(err.value, FileExistsError)  # ...of this specific type
    assert (  # The error message contains is helpful...
        'File already exists at the output location' in str(err.value)
    )
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

def test_io11_out_filepath_no_folder(tmp_dir_path):
    """Check an error is thrown if the folder of out_filepath does not exist"""
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
    assert (  # The error message contains is helpful...
        'The folder of the output file does not exist' in str(err.value)
    )
    assert str(out_filepath.parent.absolute()) in str(err.value)  # ...and contains the filepath
    print("Correct: Helpful error message was thrown")

@pytest.mark.parametrize("filename", ['foo.xlsx', 'fo .o', 'foo.gzip', '01_f.zip'])
def test_io12_out_filepath_unrecognised_extension(tmp_dir_path, filename):
    """Check an error is thrown if the folder of out_filepath does not exist"""
    # Given: Output file deos not have a recognised extension
    out_filepath = tmp_dir_path / filename
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

@pytest.mark.parametrize("filename", ['foo.csv', 'fo o', 'foo.CsV', '01_f.tXt'])
def test_io13_out_filepath_valid(tmp_dir_path, filename):
    """Check the function correctly accepts a valid in_filepath"""
    # Given: Output file has a recognised extension and does not exist
    out_filepath = tmp_dir_path / filename
    # When: Attempt to validate the filepath
    # Then: No warnings or errors are thrown
    rtn_val = 1
    with pytest.warns(None) as wrns:
        rtn_val = PCon.validate_output_options(out_filepath)
    assert len(wrns) == 0  # No warnings are produced
    assert rtn_val is None  # Validation function completed
    print("Correct: `out_filepath` was validated")
