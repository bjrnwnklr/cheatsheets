# Decorators

# Documentation

[YouTube tutorial from PyCon 2020](https://www.youtube.com/watch?v=T8CQwGIsrx4)
[GitHub repo for the tutorial](https://github.com/gahjelle/decorators_tutorial)


# Code

## Python template for a generic decorator

```python
import functools


def wrapper(func):
    """Template for decorators"""

    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        """The wrapper function replacing the original"""
        # Do something before calling the function
        value = func(*args, **kwargs)
        # Do something after calling the function
        return value

    return _wrapper
```

## A simple timeit decorator

```python
from functools import wraps
import timeit


def aoc_timer(func):
    """Times an AOC function"""

    @wraps(func)
    def _wrapper(*args, **kwargs):
        # do some timing
        start_time = timeit.default_timer()
        value = func(*args, **kwargs)
        # do some more timing and print
        duration = timeit.default_timer() - start_time
        print(f'Elapsed time to run {func.__name__}: {duration:.2f} seconds.')
        return value

    return _wrapper
```