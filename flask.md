---
title: Flask cheat sheet
author: Bjoern Winkler
date: 29-05-2019
---

# Flask cheat sheet

Flask is a Python web server. 

# Documentation

_add documentation links_

# Configuration

## Directory structure

A useful structure is described in [Miguel Grinberg's Flask Mega tutorial](https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world) and in the [Flask documentation](https://flask.palletsprojects.com/en/1.1.x/patterns/packages/):

```
/yourapplication
    /yourapplication
        __init__.py
        /static
            style.css
        /templates
            layout.html
            index.html
            login.html
            ...
    setup.py
```

Create a `setup.py` file in the outer `yourapplication` folder:

```python
from setuptools import setup

setup(
    name='yourapplication',
    packages=['yourapplication'],
    include_package_data=True,
    install_requires=[
        'flask',
    ],
)
```

To correctly resolve the names from the package, install the package in development mode as explained in the [Flask documentation](https://flask.palletsprojects.com/en/1.1.x/patterns/distribute/#distribute-deployment):

```shell
$ python setup.py develop
```

You should then set the `FLASK_APP` environment variable to the application name (`yourapplication` in this case) to be able to run the application with `flask run`.

## Environment variables e.g. secrets

Running Flask locally, you can create a file `.env` and store environment variables in the format `FLASK_APP=app` in the file. Then install `python-dotenv`, and Flask will automatically pick up the environment variables.

Important variables:

- `FLASK_APP` - set this to the application name (i.e. package name, typically `app`)
- `FLASK_ENV` - set this to `development` to run a development server. Unset or set to `production` to run a prod server.
- `SECRET_KEY` - set this to a random hex key and use it for Flask-WTF forms. Key can be generated using

```python
import secrets
print(secrets.token_hex())
```

Using a `.env` file is useful for local development using `python-dotenv`, but can also be useful for using with Docker as the format of the file is the same as the environment variables file Docker uses with the `--env-file=` command line option of `docker run`.

# Start / stop etc

- flask run etc

# WSGI servers

WSGI stands for Web Server Gateway Interface and defines a standard for a web interface supporting Python applications.

[Fullstackpython - WSGI servers](https://www.fullstackpython.com/wsgi-servers.html)

# Docker with Flask, Gunicorn, nginx

[Dockerizing Flask with Postgres, Gunicorn and nginx](https://testdriven.io/blog/dockerizing-flask-with-postgres-gunicorn-and-nginx/)
[How to configure nginx for a Flask web application](https://www.patricksoftwareblog.com/how-to-configure-nginx-for-a-flask-web-application/)

# Python app examples

- defining routes, parameters