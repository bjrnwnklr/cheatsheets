# Numpy basics

## Creating ndarrays

Create an ndarray from an iterable e.g. a list:

```python
import numpy as np

data1 = [0, 1, 2, 3, 4]
arr1 = np.array(data1)

arr1.ndim           # show dimensions e.g. 2
arr1.shape          # shape e.g. (2, 4)
arr1.dtype          # show data type of array
```

## Useful functions for creating new arrays

```python
np.zeros(10)        # creates 1 dim array with zeros
np.zeros((3, 6))    # creates new array with 3 rows and 6 columns
np.ones(10)

np.empty((2, 3, 2)) # creates empty array with 2 rows, 3 columns, 2 cols
                    # empty is typically 0 values, but could be garbage too
np.full((2, 4), 100)    # fills array with fill value (100 here)
# array([[100, 100, 100, 100],
#       [100, 100, 100, 100]])         

np.empty_like(arr1)     # creates an empty array with same shape and dtype as arr1
                        # also: ones_like, zeros_like, full_like

np.eye(3)           # Create a square N × N identity matrix (1s on the diagonal and 0s elsewhere)

np.arange(10)       # like range(10)

np.random.randn(7, 4)   # creates random distr across 7 x 4 matrix
```

### Changing the dtype of an array

```python
float_arr = arr.astype(np.float64)  # cast to floating point data
```

### Reshaping an array

```python
In [122]: arr = np.arange(32).reshape((8, 4))

In [123]: arr
Out[123]: 
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15],
       [16, 17, 18, 19],
       [20, 21, 22, 23],
       [24, 25, 26, 27],
       [28, 29, 30, 31]])
```

# Indexing and slicing

Indexing and slicing for one-dimensional arrays works like Python lists

```python
arr[5]
arr[:10]

arr[5:8] = 12       # changes values 5-7 to 12 (different to Python lists)

arr[5:8].copy()     # slices are views, so use .copy() to copy the array
```

For two dimensions, axis 0 is rows, axis 1 is columns:

```python
# Indexing
arr2d[0][2]         # selects row 0 and column 2 (1 element at cross section)
arr2d[0, 2]         # the same...

# Slicing
In [90]: arr2d
Out[90]: 
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])

In [91]: arr2d[:2]
Out[91]: 
array([[1, 2, 3],
       [4, 5, 6]])
```

It can be helpful to read the expression `arr2d[:2]` as “select the first two rows of arr2d.”

For example, I can select the second row but only the first two columns like so:

```python
In [93]: arr2d[1, :2]
Out[93]: array([4, 5])
```

Similarly, I can select the third column but only the first two rows like so:

```python
In [94]: arr2d[:2, 2]
Out[94]: array([3, 6])
```

Note that a colon by itself means to take the entire axis, so you can slice only higher dimensional axes by doing:

```python
In [95]: arr2d[:, :1]
Out[95]: 
array([[1],
       [4],
       [7]])
```

## Boolean indexing


## Fancy indexing

_Fancy indexing_ is a term adopted by NumPy to describe indexing using integer arrays. 


## Transposing arrays and swapping axes

