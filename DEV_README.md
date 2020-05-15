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
    jupyter lab
    ```

### Setting JupyterLab Terminal
On Binder (i.e. Linux), the default Terminal in JupyterLab is Bash, which this project uses.

For JupyterLab on Windows, the default Terminal is Powershell. However, on Windows, this project has been developed using Commmand Prompt (**CMD**). To ensure that JupyterLab uses CMD, you can change the Jupyter settings (at system level):
- Generate a Jupter config file with all the default settings (commented out), saved in the correct location (e.g. `~/.jupyter/`):
    ```
    # jupyter notebook --generate-config  # [Commented out so you don't re-run it]
    jupyter --config-dir   # This is the location of the config file
    notepad %USERPROFILE%/.jupyter/jupyter_notebook_config.py   # Open the location
    ```
- To set the specific setting for JupterLab Terminal:
    ```
    c.NotebookApp.terminado_settings = {'shell_command': ['cmd']}
    ```
- Further info: <https://jupyter-notebook.readthedocs.io/en/stable/config.html>
- If you want to change the Jupyter settings at project level, try changing the `JUPYTER_CONFIG_DIR` as per: <https://jupyter.readthedocs.io/en/latest/projects/jupyter-directories.html#configuration-files>

<p align="right"><a href="#top">Back to top</a></p>

## Structure of the repo
### Package
- `README.md`: Main introduction to package. Additional resources in `img/`.
- `premierconverter.py`: The code module that contains all the Python package code.
- `setup.py`, `MANIFEST.in`, `requirements.txt`: Specifications to build the package.
- `LICENSE`: Terms of use.

### Package data
- `example_data/`: Docs and data to use during development and include in the package. No data is currently committed to the repo.
- `proj_config.py`: Store the repo structure in variables so that scripts throughout the project can access relevant locations of the project.

### Development
- `DEV_README.md`: Information for developing the package.
- `binder/` and `requirements_dev.txt`: Development environment specifications for launching Binder.
- `development/`: Code and docs used during development. Manually kept in sync with package code base. Could form the basis of future code additions or automated tests.

### CI/CD
- `tests/`: Automated tests on the package.
- `azure-pipelines` and `azure_templates/`: Specifications for tasks to run in the CI pipeline on the Azure Pipeline.

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
```
pytest tests/
```
Note:
- Ensure the package is installed before running the above, if you want to test the built version.
- To see what tests pytest will discover, without actually running the tests (useful for debugging):
    ```
    pytest --collect-only tests/
    ```
- By default, `print()` statements in the test (i.e. to `stdout`) will only be shown for *failed* tests. To show them for *all* tests (including successes) on the order of tests that are run:
    ```
    pytest --capture=no tests/
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

### Run linting
To check code formatting using `pylint`.
```
pylint premierconverter.py
pylint tests/
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
