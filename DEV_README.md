<a name="top"></a>

# Premier Converter: Development notes
Notes about how to work on this project.

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
See the main [`README.md`](README.md) for a link to start the Binder instance.

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
The following will create a *source* distribution and a *wheel* distribution out of the Python package (given it includes a `setup.py`), and puts the resulting files in `build/` (for some intermediate files from the wheel build) and `dist/` (for the final source and wheel distributions) subfolders.
```
python setup.py sdist bdist_wheel
```

### Install built package
```
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install ./dist/premierconverter-*.tar.gz  # For source distribution
pip install ./dist/premierconverter-*.whl  # For the wheel
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

### Known issues
- When you set `force_overwrite=True` to write data into an existing worksheet, only values in the affected cells are replaced. So, if there are values in any other cells prior to running the app, they *remain* after the data has been written. Presumably it would be more desirable to clear values from the entire sheet (but not to delete it, in case there are formulae relying on it).

### Additional functionality
- Allow the default of `output_sheet_name` to be the *first* sheet in the workbook (i.e. guaranteed to exist for existing workbooks), rather than `Sheet1`. This avoids the situation where the user is expecting to overwrite a one-sheet workbook, but the sheet is not called `Sheet1`.
- Validate that the first column of the raw data (which goes in to form the `Ref_num` index) contains ordered, unique values.
- Validation checks on the consistency of premium values against what is expected.

### UI
- Create a GUI using: 
    - `tkinter` (built-in, suitable for simple applications). Alternative `PyQt5` has more complicated license arrangements and it more suitable for larger applications. Cannot develop this inside Kaggle or Binder.
    - a web framework, e.g. `bokeh`.

### Tests
- Create up `pytest` fixtures to allow setup / teardown for the Excel spreadsheets for testing
- Create tests for the `click` CLI
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
    - Or trying setting up a local package registry that acts like PyPI, e.g. see: <https://github.com/wolever/pip2pi>

<p align="right"><a href="#top">Back to top</a></p>