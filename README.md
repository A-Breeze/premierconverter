<a name="top"></a>

<!-- Remember to change this link to ensure it matches the current branch! -->
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/A-Breeze/premierconverter/tidy_docs?urlpath=lab)

[![Build Status](https://dev.azure.com/a-breeze/premierconverter/_apis/build/status/A-Breeze.premierconverter?branchName=master)](https://dev.azure.com/a-breeze/premierconverter/_build/latest?definitionId=1&branchName=master)

# Premier Converter
Functionality to convert an Excel spreadsheet in a given format into a more useful format.

<!--This table of contents is maintained *manually*-->
## Contents
1. [Important notes](#Important-notes)
1. [Installation](#Installation): Pre-requisites, Instructions, Explanation
1. [How to use](#How-to-use)
    - [Commandline interface (CLI)](#Commandline-interface-(CLI))
    - [Python module](#Python-module)
1. [Understanding and debugging](#Understanding-and-debugging)
    - [Known issues](#Known-issues)
1. [Contributing](#Contributing)

<p align="right"><a href="#top">Back to top</a></p>

## Important notes
This is a project for interest only, for training purposes, so it is *highly unlikely* that I will accept contributions on GitHub from the general public. I am going through the steps *as if* this were a real-life project, as realistically as possible.

**Work in progress**: The functionality is not complete, so may not work as intended. Even following these instructions to install and use the app is *at own risk*. See the project's [LICENSE](#LICENSE). Of particular concern is the current lack of automated tests, which increases the risk of introducing bugs when updates are made.

<p align="right"><a href="#top">Back to top</a></p>

## Installation
These are temporary instructions, as [explained](#Explanation) below.

### Pre-requisites
You need `conda` and `pip` available and working in a console (e.g. from Anaconda Prompt or Windows Command Prompt `cmd`).

### Instructions
1. Download the following files from the project repo into a folder anywhere your computer:
    - Code module: [`premierconverter.py`](premierconverter.py).
    - Environment specification: [`deploy_env.yml`](deploy_env.yml) and [`requirements.txt`](requirements.txt).
    - Package specification: [`setup.py`](setup.py)

    To download a single file from GitHub you need to:
    1. Go to the file's page in the repo, e.g. [`premierconverter.py`](premierconverter.py).
    1. Click to view the **Raw** version of the file 
        - Button at the top right of the file contents: <img src="img/gh_raw_button_screenshot.png" alt="GitHub Raw button screenshot" width="200">
    1. You are directed to a page with the file's content only. Right click and **Save as...** into the location you want. Save the file with the *original* file extension (e.g. `.py` not `.py.txt`).
1. Open a console and navigate to the folder where you saved the files.
1. Create the `conda` environment from which the `premierconverter` app will be run:
    ```
    conda env create -f deploy_env.yml --force
    ```
    This includes installing the `premierconverter` module for use.

### Explanation
- **Future aim**: It would be desirable for the app to be available to install using a simple `pip install` command. However, this requires a package distribution to be hosted in a central registry, which has not yet been set up.
- **Isolated environment**: The `premierconverter` app has only been developed using specific versions of dependencies. The above instructions are to install the app along with those exact dependencies into an *isolated* `conda` environment called `premcon_env` (and not into the `base` environment). This ensures it will not interfere with any other project you are working on. The strict restrictions on dependency versions might be relaxed in future.
- **Alternatives download step**: Instead of downloading each of the files individually, you could `git clone` the entire repo which will create a folder called `premierconverter`. Navigate to that location and continue the steps. This requires `git` to be available and working.

<p align="right"><a href="#top">Back to top</a></p>

## How to use
When working either in the CLI or Python module, open a console and ensure the `conda` environment `premcon_env` is activated:
```
conda activate premcon_env
```

When you are finished using the app, you can deactivate the `conda` environment by:
```
conda deactivate
```

### Commandline interface (CLI)
The core functionality can be accessed from the console without needing to start a Python instance:
- **Help**: Show the help options (and check the installation has worked):
    ```
    python -m premierconverter --help 
    ```
- **Default functionality**: Suppose you have input data saved at `data/input raw.xlsx` (relative to your current location), and you want to convert it to usable format and save it to a new workbook `formatted data.xlsx`:
    ```
    python -m premierconverter "data/input raw.xlsx" "formatted data.xlsx"
    ```
- **Basic fail-safe**: By default, the app will *refuse* to overwrite an output worksheet that already exists. So if you run the above command again, it will fail. If you're sure you want to overwrite the existing output, you can `--force` this behaviour:
    ```
    python -m premierconverter "data/input raw.xlsx" "formatted data.xlsx" --force
    ```
- **Output worksheet**: Alternatively, you can write the output to a new worksheet. The default is `Sheet1`, so let's specify a new name `S2` using the `-o` option:
    ```
    python -m premierconverter "data/input raw.xlsx" "formatted data.xlsx" -o S2
    ```
- **Trial run**: Suppose the input data has a large number of rows, so you want to trial the app on the first few rows, to check it is working before running the conversion on the entire data. Use the `-r` option to specify the number of rows to read in:
    ```
    python -m premierconverter "data/input raw.xlsx" "formatted data.xlsx" -r 3
    ```
- **Input worksheets**: Suppose the input workbook contains multiple worksheets, of which you want to use the sheet named `my_sheet` - use the `-i` option.
    ```
    python -m premierconverter "data/input raw.xlsx" "formatted data.xlsx" -i "my_sheet"
    ```
    Instead of the sheet name, you can alternatively give the sheet order number, but *be careful*: they are number starting from the first sheet being number `0` (not `1`).
- **Combination of options**: Any combination of the above options is permitted. The restrictions are:
    - You must specify the `<input filepath>` and `<output filepath>`, and they must be in that order.
    - The option values and filepaths can be `"in quotes"` or `not`. You'll need to put them in quotes if they contain a space.

### Python module
You can use the project functionality in Python by importing it into your script as a module:
```python
import premierconverter as PCon
```
This might be useful because:
- There are a few (subtle) options that are not available from the CLI.
- You can split the functionality down into component tasks, e.g. for debugging.

However, this is only currently documented in the docstrings of the individual functions.

<p align="right"><a href="#top">Back to top</a></p>

## Understanding and debugging
If the app fails, hopefully the resulting error message is helpful to diagnose and resolve the problem. Failing that (or if you are interested), you need to look through the code script: `premierconverter.py`.

Alternatively, there is a notebook for development of the project which works through all the steps in the pipeline. It is viewable (without any installation) on GitHub here: <https://github.com/A-Breeze/premierconverter/tree/master/development/compiled>.

Also see the [Contributing](#Contributing) section below.

### Known issues
Along with a brain dump of possible backlog items, any *known issues* are documented in the relevant section of the [`DEV_README.md`](DEV_README.md).

<p align="right"><a href="#top">Back to top</a></p>

## Contributing
Notes about working on this project are documented in the [`DEV_README.md`](DEV_README.md). Development work has so far been done inside the Binder instance (see the link at the <a href="#top">top</a>).

<p align="right"><a href="#top">Back to top</a></p>
