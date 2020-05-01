#!/usr/bin/env bash

# Script to record CLI commands
# For development ONLY. Suitable to run on Linux ONLY.

##############
# How to run #
##############
# One-time permission change:
# $ chmod +x development/CLI_commands.sh  

# Ensure the premierconverter package is installed in the current environment
# (See README.md for installing the development or built versions)
# Ensure pyprojroot is installed:
# pip install pyprojroot==0.2.0 > /dev/null  # Will install if not already available (silently)

# To ensure we're testing the *installed* module premierconverter, 
# rather than the original script premierconverter.py, 
# move to the subdirectory before running the script:
# $ cd development
# $ source CLI_commands.sh

#########
# Setup #
#########
# Get the example data location from the project config file
ORIG_WDIR=$PWD
PROJ_ROOT_DIR=$(python -c "from pyprojroot import here; print(here())")
cd $PROJ_ROOT_DIR
EX_DATA_DIR=$(python -c "import proj_config; print(proj_config.example_data_dir_path)")
cd $ORIG_WDIR
# Tidy up
unset ORIG_WDIR
unset PROJ_ROOT_DIR

# Specify that all commands are printed first, before their output
set -x
# Need to unset this option before the end, or it will persist: set +x

####################
# Example commands #
####################
# All commands call the module by "python -m premierconverter [...]".
# They could also be run on the script by "python premierconverter.py [...]" 
# *only if* the script is in the current directory.

# python -m premierconverter --version  # Shows the version
# python -m premierconverter --help  # Shows the help

# python -m premierconverter  # Error because in_filename is required
# python -m premierconverter $EX_DATA_DIR/there_is_no_file_here  # Error because file does not exist
# python -m premierconverter $EX_DATA_DIR/minimal_dummy_data_01.xlsx  # Error because out_filename is required
# python -m premierconverter $EX_DATA_DIR/minimal_dummy_data_01.xlsx -i "no_sheet" out_data.xlsx  # Error because sheet does not exist
# python -m premierconverter $EX_DATA_DIR/minimal_dummy_data_01.xlsx -i "2" out_data.xlsx  # Warning and error because sheet not correctly formatted

# python -m premierconverter $EX_DATA_DIR/minimal_dummy_data_01.xlsx out_data.xlsx  # Creates output
# python -m premierconverter $EX_DATA_DIR/minimal_dummy_data_01.xlsx out_data.xlsx  # Fails because the output already exists
# python -m premierconverter $EX_DATA_DIR/minimal_dummy_data_01.xlsx out_data.xlsx -o "3rows" # Suceeds to another sheet
# python -m premierconverter $EX_DATA_DIR/minimal_dummy_data_01.xlsx out_data.xlsx --force  # Forces overwriting to occur
# python -m premierconverter $EX_DATA_DIR/minimal_dummy_data_01.xlsx out_data.xlsx -o "3rows" -r 3 -n --force # Try some other options

############
# Clean up #
############
# Unset the 'explicit' option
set +x

# Remove the variables that were defined in the script
unset EX_DATA_DIR
