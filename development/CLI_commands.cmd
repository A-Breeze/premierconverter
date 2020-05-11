@echo off
:: Script to record CLI commands
:: For development ONLY. Suitable to run on Windows ONLY.

:: ############
:: How to run #
:: ############
:: Ensure the premierconverter package is installed in the current environment
:: (See README.md for installing the development or built versions)
:: Ensure pyprojroot is installed:
:: pip install pyprojroot==0.2.0  & :: Will install if not already available

:: To ensure we're testing the *installed* module premierconverter, 
:: rather than the original script premierconverter.py, 
:: move to the subdirectory before running the script:
:: > cd development
:: > CLI_commands.cmd

:: #######
:: Setup #
:: #######
:: Get the example data location from the project config file
set ORIG_WDIR=%cd%
python -c "from pyprojroot import here; print(here())" > tmpFile 
set /p PROJ_ROOT_DIR= < tmpFile 
del tmpFile
cd %PROJ_ROOT_DIR%
python -c "import proj_config; print(proj_config.example_data_dir_path)" > tmpFile 
set /p EX_DATA_DIR= < tmpFile 
del tmpFile
cd %ORIG_WDIR%
:: Tidy up
set ORIG_WDIR=
set PROJ_ROOT_DIR=

:: Specify that all commands are printed first, before their output
@echo on
:: To unset this option: @echo off

:: ##################
:: Example commands #
:: ##################
:: All commands call the module by "python -m premierconverter [...]".
:: They could also be run on the script by "python premierconverter.py [...]" 
:: *only if* the script is in the current directory.

:: python -m premierconverter --version  & :: Shows the version
:: python -m premierconverter --help  & :: Shows the help

:: python -m premierconverter  & :: Error because in_filename is required
:: python -m premierconverter "%EX_DATA_DIR%/there_is_no_file_here"  & :: Error because file does not exist
:: python -m premierconverter "%EX_DATA_DIR%/minimal01_input.csv"  & :: Error because out_filename is required

:: python -m premierconverter "%EX_DATA_DIR%/minimal01_input.csv" out_data.csv  & :: Creates output
:: python -m premierconverter "%EX_DATA_DIR%/minimal01_input.csv" out_data.csv  & :: Fails because the output already exists
:: python -m premierconverter "%EX_DATA_DIR%/minimal01_input.csv" out_data.csv --force  & :: Forces overwriting to occur

:: Try some other options
:: python -m premierconverter "%EX_DATA_DIR%/minimal01_input.csv" out_data.csv -r 3 -n --force --sep ","

:: ##########
:: Clean up #
:: ##########
:: Unset the 'explicit' option
@echo off

:: Remove the variables that were defined in the script
set EX_DATA_DIR=
