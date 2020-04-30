#!/usr/bin/env bash

# Script to record CLI commands
# This script can also be run from the command line by:
# $ chmod +x CLI_commands.sh
# $ source CLI_commands.sh

# Ensure we're in the correct environment
conda activate notebook

# Specify that all commands are printed first, before their output
set -x
# Need to unset this option before the end, or it will persist: set +x

# Example commands
python premierconverter.py --version  # Shows the version
# python -m premierconverter --version  # Also works
# python premierconverter.py --help  # Shows the help
# python premierconverter.py  # Gives an error because input_filename is required
# python premierconverter.py example_data/not_a_file.p  # Prints the error generated from the Python code

# python premierconverter.py example_data/minimal_dummy_data_01.xlsx  # Creates output
# python premierconverter.py example_data/minimal_dummy_data_01.xlsx  # Fails because the output already exists
# python premierconverter.py example_data/minimal_dummy_data_01.xlsx --force  # Forces overwriting to occur

# Unset the verbose option
set +x
