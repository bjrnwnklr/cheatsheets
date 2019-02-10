---
title: Bjoern's Python cheat sheet
author: Bjoern Winkler
date: 26-01-2019
...


### Lists
Creating a list
```python
my_list = list()
my_list = []

''.join(my_list)                # joins the elements of the list with no blanks ('')
```
List slices (non-destructive)
```python
# [] notation uses [start:stop:step]
my_list[5:10:2]                 # select elements 5, 7 and 9
```


List methods (changing the state of the list)
```python
my_list.append('element')       # append to a list
my_list.remove('element')       # remove first instance of element from list
popped = my_list.pop()          # remove and return the last element of a list
popped = my_list.pop(0)         # pop off element at index 0 (first element)
my_list.extend([3, 4])          # extending a list by adding new list to the end (useful for combining two lists)
my_list.insert(0, 'element')    # insert object _before_ the specified index - here at the beginning

```

Copying a list
```python
# lists are not copied when assigning a list to a new variable - it just assigns a reference
copied_list = my_list.copy()
copied_list = my_list[:]
```

List comprehensions
```python
squares = [x**2 for x in range(1,11)]
```

### Sets
```python
my_set = set()
my_set = {}

# add an element to a set
my_set.add('element')
# remove an element from a set
my_set.remove('element')
```

### Dictionaries
```python
# creating a dictionary
my_dict = dict()
# ... or ...
my_dict = {'color': 'green', 'points': 5}

# accessing a value
my_dict['color']
# adding a new key-value pair
my_dict['name'] = 'Hans'
# looping through all key-value pairs
for key, value in my_dict.items():
    print(key, value)
```
##### getting min / max key values
```python
min_key = min(my_dict)
max_key = max(my_dict)
```


### Strings
##### String formatting
```python
# string formatting expression
'Hello %s, how are you today on %d %s' % ('Bjoern, 30, 'November')
```

##### replacing text in strings (similar to re.sub just without regex)
```python
new_text = text.replace('text to be replaced', '')
```

##### joining multiple elements to a string
```python
val = ''.join(x[i] for i in range(i-2, i+3))
```


### Regular expressions
##### finding and returning the first match of two repeating letters
```python
import re

# returns a match object 
# (\w) - finds and returns one letter
# \1 repeats the previous group (finds the same letter again)
reg_match = re.search(r'(\w)\1', text)

# check if found and return the index of the found sequence
if reg_match:
    i = reg_match.start()
```

##### replacing a single letter or pattern in a text
```python
import re

# replaces all 'a' in text with empty string (removes a)
new_text = re.sub(r'a', '', text)
```

### Files
##### Built in file reader
```python
# open a text file and read from it
txt_file = open(r'D:\Python\BW\textfile.txt', 'r')
# read all of the file
my_text = txt_file.read()

# read 10 characters
my_text = txt_file.read(10)
# read a single line
my_txt = txt_file.readline()
# read all lines, but split to a list
my_txt = txt_file.readlines()
```
##### Open a file with „with“
Closes the file automatically after exiting the with block.
From [Python for data science](https://www.safaribooksonline.com/library/view/python-for-data/9781491957653/ch03.html)
```python
with open(path) as f:
    lines = [x.rstrip() for x in f]
```

##### CSV file reader
```python
import csv
csv_file = open(r'D:\Python\BW\csvfile.csv', newline='', encoding='utf-8')
reader = csv.reader(csv_file)
# read the first line
headers = next(reader)
# read all other lines
records = list(reader)
```

### Collections
##### Counter
```python
import collections as coll
# create counter from a list
my_list = list()
my_counter = coll.Counter(my_list)

# remove items from counter
my_counter.pop(item)

# output 5 most common items as a list
my_counter.most_common(5)
```

##### Deques (double ended lists)
```python
from collections import deque

# create a new deque with a single element
# this constructor takes a list as argument and converts it to a deque
circle = deque([0])

# standard operations
# append an element at the current position
circle.append(item)
# rotate the current position back (-) and forth (+)
circle.rotate(-7)
# pop the current element
circle.pop()
```

##### defaultdictionary
create a default dictionary that sets not found entries to 0
```python
from collections import defaultdictionary

new_dict = defaultdictionary(int)
```

create a defaultdictionary that provides a '.' back for not found entries
```python
new_dict = defaultdictionary(lambda: '.')
```

### Command line arguments
Using `argparse`:

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--URL', '-u', help='URL to query')
args = parser.parse_args()

if args.URL:
    print('Set URL to %s' % args.URL)
```

### Exceptions
##### Raising your own exception

1) Define your own deception by inheriting from the `Exception` class:

```python
class ElfDeadException(Exception):
        def __init__(self, message):
            self.message = message
```

2) Raise the exception in the code

```python
if hp <= 0:
    alive = False
    raise ElfDeadException('Elf %s died!' % elf)
```

3) Catch the exception and do something with it

```python
try:
    # execute some code where the exception could occur

except ElfDeadException as elf_dead_exception:
    # do something with the exception, e.g. print
    print(elf_dead_exception)
    # you can also "continue" here to go back to a loop
    continue

else:
    # this gets executed if code completed and no exception was caught
    print('No exception, code successfully completed')

finally:
    # this gets executed no matter what. You can clean up e.g. close a file
    print('cleaning up!')
```

### Functions
##### Codewars "Calculating with functions" - a function for each number and operator.

Two clever ways of doing this:
1) return a lambda function for the first argument:

```python
def zero(f = None): return 0 if not f else f(0)
def one(f = None): return 1 if not f else f(1)
def two(f = None): return 2 if not f else f(2)
def three(f = None): return 3 if not f else f(3)
def four(f = None): return 4 if not f else f(4)
def five(f = None): return 5 if not f else f(5)
def six(f = None): return 6 if not f else f(6)
def seven(f = None): return 7 if not f else f(7)
def eight(f = None): return 8 if not f else f(8)
def nine(f = None): return 9 if not f else f(9)

def plus(y): return lambda x: x+y
def minus(y): return lambda x: x-y
def times(y): return lambda  x: x*y
def divided_by(y): return lambda  x: x/y
```

2) using more lambdas... and mapping the names to numbers

```python
id_ = lambda x: x
number = lambda x: lambda f=id_: f(x)
zero, one, two, three, four, five, six, seven, eight, nine = map(number, range(10))
plus = lambda x: lambda y: y + x
minus = lambda x: lambda y: y - x
times = lambda x: lambda y: y * x
divided_by = lambda x: lambda y: y / x
```

##### Codewars "Unary function chain" - chain a number of functions

Left folding a number of functions so that

```python
chained([a, b, c, d])(input)

d(c(b(a(input))))
```

My way of doing it:
```python
def chained(functions):
    c_f = lambda f1, f2: lambda x: f2(f1(x))
    result = lambda x: x
    for f in functions:
        result = c_f(result, f)
    return result
```

Best practices way:
```python
def chained(functions):
    def chain(input):
        for f in functions:
            input = f(input)
        return input
    return chain
```

Another way using functools.reduce:
```python
from functools import reduce

def chained(functions):
    return lambda x: reduce(lambda v, f: f(v), functions, x)
```