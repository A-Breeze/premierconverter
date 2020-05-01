<a name="top"></a>

[![Build Status](https://dev.azure.com/a-breeze/premierconverter/_apis/build/status/A-Breeze.premierconverter?branchName=master)](https://dev.azure.com/a-breeze/premierconverter/_build/latest?definitionId=1&branchName=master)

# Premier Converter
Functionality to convert an Excel spreadsheet in a given format into a more useful format.

<!--This table of contents is maintained *manually*-->
## Contents
1. [Setup](#Setup)
    - [Start Binder instance](#Start-Binder-instance)
    - [Development environment](#Development-environment)
1. [Structure of the repo](#Structure-of-the-repo)
1. [Tasks](#Tasks)
    - [Development installation](#Development-installation)
    - [Run automated tests](#Run-automated-tests)
    - [Build package](#Build-package)
    - [Compile development notebooks](#Compile-development-notebooks)
1. [Further notes](#Further-notes)
1. [Future ideas](#Future-ideas)

<p align="right"><a href="#top">Back to top</a></p>

## Setup
This document describes how to run the repo using JupyterLab on Binder. It *should* be possible to run the code in JupyterLab (or another IDE) from your own machine (i.e. not on Binder), but this hasn't been tested. Follow the bullet point to install it *Locally on Windows* in [Development environment](#Development-environment) below.

All console commands are **run from the root folder of this project** unless otherwise stated.

### Start Binder instance
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/A-Breeze/premierconverter/add_CLI?urlpath=lab)

### Development environment
The development requirements consist of the package dependencies, plus extra packages useful during development, as specified in `requirements_dev.txt`. They can be automatically installed into a conda-env as follows.
- **Binder**: A conda-env is created automatically from `binder/environment.yml` in Binder is called `notebook` by default. Unless otherwise stated, the below console commands assume the conda-env is activated, i.e.:
    ```
    conda activate notebook
    ```
- **Locally** (on Windows):
    ```
    conda env create -f binder\environment.yml --force
    conda activate premcon_dev_env
    ```

<p align="right"><a href="#top">Back to top</a></p>

## Structure of the repo
**TODO**: Describe the structure

<p align="right"><a href="#top">Back to top</a></p>

## Tasks
### Development installation
While developing the package, we can install it from the local code (without needing to build and then install) as follows:
```
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Run automated tests
Ensure the package is installed.
```
pytest
```

### Build package
The following will create a *source* distribution and a *wheel* distribution out of the Python package (given it includes a `setup.py`), and puts the resulting files in `build/` (for some intermediate files) and `dist/` (for the final source and wheel distributions) subfolders.
```
python setup.py sdist bdist_wheel
```

### Install built package
```
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install ./dist/premierconverter-0.1.2.tar.gz  # Specify the desired version
```

### Compile development notebooks
The development notebooks have been saved in `jupytext` markdown format, so they can be executed (to produce the outputs) and compiled (to `ipynb` format) as follows:
```
jupytext --to notebook --output development/compiled/data-conversion-challenge-202004.ipynb --execute development/data-conversion-challenge-202004.md
```

<p align="right"><a href="#top">Back to top</a></p>

## Further notes
### Using Binder for development
- Advantage: This will run it in the browser, so there is no prerequisite of software installed on your computer (other than a compatible browser). 
- Disadvantages:
    - Security is *not* guaranteed within Binder (as per [here](https://mybinder.readthedocs.io/en/latest/faq.html#can-i-push-data-from-my-binder-session-back-to-my-repository)), so I'll be pushing Git from another location, which involves some manual copy-paste.
    - The package environment has to be restored each time, which takes some time.

### Research notes
- Example of a package that consists of just one module: <https://github.com/benjaminp/six>
- Worked example of creating an Azure Pipeline to test, build and publish a Python package as an Azure Artifact: <https://medium.com/raa-labs/creating-a-pipeline-in-azure-devops-to-build-and-publish-python-packages-artifacts-ea2f99410e6c>
- Using `click` to create a CLI from Python functions: <https://dbader.org/blog/python-commandline-tools-with-click>
- How to make a Python script executable from the command line (in Linux): <https://dbader.org/blog/how-to-make-command-line-commands-with-python>

<p align="right"><a href="#top">Back to top</a></p>

## Future ideas
Backlog of all possible ideas, big or small, high priority or low.

### Additional functionality
- Argument to only read in the first `n_row` rows of the raw data, so that you can test it on a small sample before running the whole pipeline.
- Once raw data has been read in, delete empty columns from the right and empty rows from the bottom, so they are not counted for validation.
- Validate that the first column of the raw data (which goes in to form the `Ref_num` index) contains ordered, unique values.
- Refactor the index name `Ref_num` to be a configuration parameter.
- Allow the user to overwrite the configuration parameters.
- Validation checks on the consistency of premium values against what is expected.

### UI
- Allow the functionality to be used as a CLI using the `click` package.
- Create a GUI using: 
    - `tkinter` (built-in, suitable for simple applications). Alternative `PyQt5` has more complicated license arrangements and it more suitable for larger applications. Cannot develop this inside Kaggle or Binder.
    - a web framework, e.g. `bokeh`.

### Tests
- Create up `pytest` fixtures to allow setup / teardown for the Excel spreadsheets for testing
- Aspects to test:
    - Expected functionality
    - Expected failures / warnings
    - Edge cases
    - Benchmark performance

### Documentation
- How to:
    - Install and use
    - Debug warnings and errors
    - Contribute to development (this file)
- Record version history
- Possible formats: 
    - (Compiled) notebook (works well on GitHub)
    - Markdown

### Development process:
- Use `pylint` to check the code quality.
- Use `pytest-cov` to check the coverage (although it will be very low).
- Consider relaxing the dependency version requirements to minimum, rather than strict, versions.
- Investigate using `tox` to run tests on with various dependency versions.
- Complete setting up CI/CD pipeline on Azure DevOps by:
    - *Azure Artifacts* for a (private) Python package registry

<p align="right"><a href="#top">Back to top</a></p>
