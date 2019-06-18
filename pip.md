---
title: Pip cheatsheet
author: Bjoern Winkler
date: 18-June-2019
---

# Pip cheatsheet

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

