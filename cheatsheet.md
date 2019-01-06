# Björn's python cheat sheet
v001, 1 Dec 2018

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