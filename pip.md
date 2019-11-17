---
title: Pip cheatsheet
author: Bjoern Winkler
date: 17-November-2019
---

# Pip cheatsheet

# Managing virtual environments with `pipenv` and `pip`

`pipenv` is a virtual environment manager that combines `pip` and `virtualenv` together.

Documentation [here](https://pipenv-fork.readthedocs.io/en/latest/basics.html)

## Installation

Install `pipenv` in your root python environment:

    > pip install pipenv

Set the `PIPENV_VENV_IN_PROJECT` environment variable to 1 to have virtual environments created in the root of the project directory (default is central in user home).

## Creating a virtual environment

Create a project directory, cd to it and run pipenv:

    > mkdir projectdir
    > cd projectdir
    > pipenv install numpy

This will create a few files and the virtual environment in a folder `.venv`.

Activate the virtual environment:

    > pipenv shell

Exit the virtual environment:

    > exit

## Installing packages

Install the latest version of a package using

    > pipenv install pandas

Install a specific version (this won't then update the version):

    > pipenv install pandas==0.24

Install a specific major version (it will still update to the latest minor version):

    > pipenv install pandas~=0.24

This will install pandas 0.25.3 and update to any minor version later.

## Updating virtual environment

Update all packages in the virtual environment using `pipenv update`:

    > pipenv update

This will only update packages that have not been locked in the PIPFILE to a specific version. You can change the settings in the PIPFILE to allow any versions using a "\*" for the version.

# Managing virtual environments with `venv` and `pip`

`venv` is one of the many environment managers available and comes with a standard Python installation, so no need to install anything additionally.

Documentation [here](https://docs.python.org/3/library/venv.html)

Tutorial [here](https://docs.python.org/3/tutorial/venv.html)

## Creating a virtual environment

The recommended way to create a virtual environment for a project is to create it in the directory of the project:

    > mkdir my-project
    > cd my-project
    > python -m venv env
    > env\scripts\activate.bat

This creates the environment in the folder `env`. `env` is also a standard entry in `.gitignore` files.

## Installing packages

Then install packages into the virtual environment with `pip`:

    > pip install jupyter numpy nltk

Searching for packages:

    > pip search nltk

Installing the latest version:

    > pip install nltk

Installing a specific version:

    > pip install nltk==2.10.0

Upgrading to the latest version:

    > pip install --upgrade nltk

Uninstalling a package:

    > pip uninstall nltk

Showing info about a package:

    > pip show nltk

Show all packages installed in the virtual environment:

    > pip list

Freezing a list of packages to replicate the environment:

    > pip freeze > requirements.txt

Installing using requirements.txt

    > pip install -r requirements.txt
