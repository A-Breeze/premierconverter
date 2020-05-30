<a name="top"></a>

<!-- Remember to change this link to ensure it matches the current branch! -->
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/A-Breeze/premierconverter/develop?urlpath=lab)

[![Build Status](https://dev.azure.com/a-breeze/premierconverter/_apis/build/status/A-Breeze.premierconverter?branchName=master)](https://dev.azure.com/a-breeze/premierconverter/_build/latest?definitionId=1&branchName=master)

# Premier Converter
Python command line app to convert a raw CSV with a given format into a more useful format.

<!--This table of contents is maintained *manually*-->
## Contents
1. [Important notes](#Important-notes)
1. [Installation](#Installation)
1. [How to use](#How-to-use)
    - [Commandline interface (CLI)](#Commandline-interface-CLI)
    - [Python module](#Python-module)
1. [Understanding and debugging](#Understanding-and-debugging)
    - [Known issues](#Known-issues)
1. [Contributing](#Contributing)

<p align="right"><a href="#top">Back to top</a></p>

## Important notes
Following the instructions documented in this project to install and use the app is *at own risk*. See the project's [LICENSE](LICENSE). It is *unlikely* that public contributions on GitHub will be accepted.

<p align="right"><a href="#top">Back to top</a></p>

## Installation
You need `conda`, `pip` and `git` available and working in a console (e.g. from Anaconda Prompt or Windows Command Prompt `cmd`). Run the following commands to: 
- Create a `conda` environment with the required dependencies
- Activate the environment
- Install the latest release of the app using `pip` and `git`
```
conda create -n premcon_env python==3.6.6 pip==20.0.2 --force
conda activate premcon_env
pip install git+https://github.com/A-Breeze/premierconverter.git@master
```

To check the installation has succeeded, run the app to show the version that is installed and verify this is the version you are expecting:
```
python -m premierconverter --version
``` 

### Notes
- **Isolated environment**: The `premierconverter` app has only been developed using specific versions of dependencies. The above instructions are to install the app along with those exact dependencies into an *isolated* `conda` environment called `premcon_env` (and not into the `base` environment). This ensures it will not interfere with any other project you are working on.
- **Previous releases**: The instructions above install the latest version of the app. To install a previous version, replace `@master` (at the end of the `pip` command) with the *tag* of the desired release, e.g. `@0.3.3`. 

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
- **Default functionality**: Suppose you have input data saved at `data/input raw.csv` (relative to your current location), and you want to convert it to usable format and save it to a new workbook `formatted data.csv`:
    ```
    python -m premierconverter "data/input raw.csv" "formatted data.csv"
    ```
- **Basic fail-safe**: By default, the app will *refuse* to overwrite a file that already exists at a specified output location. So if you run the above command again, it will fail. If you're sure you want to overwrite the existing output, you can `--force` this behaviour:
    ```
    python -m premierconverter "data/input raw.csv" "formatted data.csv" --force
    ```
- **Trial run**: Suppose the input data has a large number of rows, so you want to trial the app on the first few rows, to check it is working before running the conversion on the entire data. Use the `-r` option to specify the number of rows to read in:
    ```
    python -m premierconverter "data/input raw.csv" "formatted data.csv" -r 3
    ```
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

See the docstrings of the individual functions for further documentation and help.

<p align="right"><a href="#top">Back to top</a></p>

## Understanding and debugging
If the app fails, hopefully the resulting error message is helpful to diagnose and resolve the problem. Failing that (or if you are interested), you need to look through the code script: `premierconverter.py`.

Alternatively, there is a notebook for development of the project which works through all the steps in the pipeline. It is viewable (without any installation) on GitHub here: <https://github.com/A-Breeze/premierconverter/tree/master/development/compiled>.

Also see the [Contributing](#Contributing) section below.

### Known issues
Currently, there are no known issues adversely affecting the app. If you find an issue, please report it to the project maintainer.

<p align="right"><a href="#top">Back to top</a></p>

## Contributing
Notes about working on this project are documented in the [`DEV_README.md`](DEV_README.md). Development work was initiated by working in the Binder instance (see the link at the <a href="#top">top</a>). Since then, priority has moved to ensuring the app works on Windows (although the Binder work on Linux remains substantiallly valid).

<p align="right"><a href="#top">Back to top</a></p>
