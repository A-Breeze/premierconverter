<a name="top"></a>

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

<p align="right"><a href="#top">Back to top</a></p>

## Setup
This document describes how to run the repo using JupyterLab on Binder. It *should* be possible to run the code in JupyterLab (or another IDE) from your own machine (i.e. not on Binder), but this hasn't been tested. Follow the bullet point to install it *Locally on Windows* in [Development environment](#Development-environment) below.

All console commands are **run from the root folder of this project** unless otherwise stated.

### Start Binder instance
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/A-Breeze/premierconverter/basic_tests?urlpath=lab)

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
The following will create a *source* distribution and a *wheel* distribution out of the Python package (given it includes a `setup.py`), and puts the resulting files in `build/` and `dist/` subfolders.
```
python setup.py sdist bdist_wheel
```

### Install built pacakge
```
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install ./dist/premierconverter-0.1.2.tar.gz  # Specify the desired version
```

### Execute notebooks from command line
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

### Future ideas
- Additional functionality:
    - Argument to only read in the first `n_row` rows of the raw data, so that you can test it on a small sample before running the whole pipeline.
    - Once raw data has been read in, delete empty columns from the right and empty rows from the bottom, so they are not counted for validation.
    - Validate that the first column of the raw data (which goes in to form the `Ref_num` index) contains ordered, unique values.
    - Refactor the index name `Ref_num` to be a configuration parameter.
    - Allow the user to overwrite the configuration parameters.
    - Validation checks on the consistency of premium values against what is expected.
- Documentation:
    - How to:
        - Install and use
        - Debug warnings and errors
        - Contribute to development (this file)
    - Record version history
    - Possible formats: 
        - (Compiled) notebook (works well on GitHub)
        - Markdown
- Set up CI/CD pipeline on Azure DevOps (free with GitHub subscription), e.g.:
    - Alternatively *Azure Test Plans* and *Pipelines*, e.g.: <https://docs.microsoft.com/en-us/azure/devops/pipelines/artifacts/pypi>
    - *Azure Artifacts* for a (private) Python package registry, e.g.: <https://docs.microsoft.com/en-us/azure/devops/artifacts/quickstarts/python-packages?view=azure-devops>

<p align="right"><a href="#top">Back to top</a></p>
