"""
Describe the package for distribution

Sections:
1. Create the values you want to populate setup() with
2. Define setup() with the values you created
"""

#################
# Create values #
#################
import io
import os
from setuptools import setup

# Package meta-data
NAME = 'premierconverter'
VERSION = '0.3.2'  # Ensure this is kept in-sync with __version__ in the code module
DESCRIPTION = 'Convert specified data into a more usable format.'
URL = 'https://github.com/A-Breeze/premierconverter'
EMAIL = 'maintainer@email.TBA'
AUTHOR = 'A-Breeze'
REQUIRES_PYTHON = '>=3.6.0'

# What packages are required for this module to be executed?
def list_reqs(fname='requirements.txt'):
    """Get string data from a file"""
    with open(fname) as fd:
        return fd.read().splitlines()

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file
here = os.path.abspath(os.path.dirname(__file__))
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

#######################
# Put it into setup() #
#######################
setup(
    # Package metadata
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,

    # Package contents
    py_modules=[NAME],  # For a package that consists of just one module
    # packages=setuptools.find_packages(exclude=('tests',)),
    # package_data={NAME: ['VERSION',]},
    include_package_data=True,

    # Package dependencies
    python_requires=REQUIRES_PYTHON,
    install_requires=list_reqs(),
    extras_require={},

    # Additional info
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
)
