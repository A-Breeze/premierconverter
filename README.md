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
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/A-Breeze/premierconverter/setup?urlpath=lab)

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
- Set up CI/CD pipeline on Azure DevOps (free with GitHub subscription), e.g.:
    - Alternatively *Azure Test Plans* and *Pipelines*, e.g.: <https://docs.microsoft.com/en-us/azure/devops/pipelines/artifacts/pypi>
    - *Azure Artifacts* for a (private) Python package registry, e.g.: <https://docs.microsoft.com/en-us/azure/devops/artifacts/quickstarts/python-packages?view=azure-devops>

<p align="right"><a href="#top">Back to top</a></p>
