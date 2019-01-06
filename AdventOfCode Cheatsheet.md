# Advent of code - cheatsheet

Some cool tricks learned from Advent Of Code 2018

- [Advent of code - cheatsheet](#advent-of-code---cheatsheet)
  - [1. Decoding some lines from input file](#1-decoding-some-lines-from-input-file)
  - [2. Splitting input into a list of tuples and converting to int at the same time](#2-splitting-input-into-a-list-of-tuples-and-converting-to-int-at-the-same-time)
  - [2. Using regex to find data in input file and converting to timestamp](#2-using-regex-to-find-data-in-input-file-and-converting-to-timestamp)
  - [3. sorting a dictionary](#3-sorting-a-dictionary)
  - [4. finding maximum value /key in a dictionary](#4-finding-maximum-value-key-in-a-dictionary)
  - [5. finding maximum from a dictionary with tuple as key](#5-finding-maximum-from-a-dictionary-with-tuple-as-key)
  - [6. List comprehension with 3 loops](#6-list-comprehension-with-3-loops)
  - [Advent of code 2018, day 11](#advent-of-code-2018-day-11)
    - [Using a partial sum table to calculate sums across a 2D grid](#using-a-partial-sum-table-to-calculate-sums-across-a-2d-grid)
  - [Advent of code 2018, day 12](#advent-of-code-2018-day-12)
    - [using regex to find all lines that contain '.' and '#'](#using-regex-to-find-all-lines-that-contain--and)

## 1. Decoding some lines from input file 

Lines look like

    #1 @ 1,3: 4x4
    #2 @ 3,1: 4x4
    #3 @ 5,5: 2x2

```python
# regex to find all numbers (\d) with 1 or more digits (+)
# map converts them to int tuples
# * unpacks the output
import re

input_file = open(r'filename.txt', 'r')

claims = [[*map(int, re.findall(r'\d+', l))] for l in input_file.splitlines() if l]
```

## 2. Splitting input into a list of tuples and converting to int at the same time

Input looks like:

    77, 83
    345, 79

```python
coordinates_str = [tuple(map(int, line.split(','))) for line in input_file]
```

## 2. Using regex to find data in input file and converting to timestamp
   
input lines look like

    [1518-05-24 23:56] Guard #1721 begins shift
    [1518-08-22 00:09] falls asleep

```python
# () in regex save results as 'groups' - can be accessed by .group(n) argument
one_line = re.search(r'\[(.+)\]\s(\w+)\s(.+)', data_input[0])
# datetime.fromisoformat(string) converts timestamp in yyyy-mm-dd hh:mm format
timestamp = datetime.fromisoformat(one_line.group(1))
```

## 3. sorting a dictionary

`event_list` is a list of dictionaries, `keys` is a list of strings that represent key names
`values` is a list of values
```python
# append a new dictionary made of keys and values
event_list.append(dict(zip(keys, values)))
# sort the list of dictionaries using the 'timestamp' entry of each dictionary
sorted_events = sorted(event_list, key=lambda event_entry: event_entry['timestamp'])
```

## 4. finding maximum value /key in a dictionary

Find the key for the maximum value in a dictionary - similar to the sorting above

```python
k = max(dictionary, key = lambda k: dictionary[k])
```

## 5. finding maximum from a dictionary with tuple as key

Find the key for the maximum value in a dictionary with a ('string', int) tuple as key, like

    dictionary[('2879', 49)] = 21

```python
best_key1, best_key2 = max(dictionary, key = dictionary.get)
```

## 6. List comprehension with 3 loops
   
```python
# creates tuples (0, 0, 0) to (10, 10, 10)
test_squares = [(i, j, k) for i in range(10) for j in range(10) for k in range(10)]
```

## Advent of code 2018, day 11
### Using a partial sum table to calculate sums across a 2D grid

```python
# create partial sums
# A B C
# D E F
# G H I
# partial sum for I =
# H + F + f(I) - E
# (x-1, y) + (x, y-1) + f(I) - (x-1, y-1)
for x in range(300):
    for y in range(300):
        part_sum[x][y] = get_part_sum(x-1, y) + get_part_sum(x, y-1) + grid[x][y] - get_part_sum(x-1, y-1)
```

## Advent of code 2018, day 12
### using regex to find all lines that contain '.' and '#'

Input looked like this:

```
initial state: #.##.###.#.##...##..#..##....#.#.#.#.##....##..#..####..###.####.##.#..#...#..######.#.....#..##...#

.#.#. => .
...#. => #
..##. => .
....# => .
##.#. => #
```

```python
import re
# 'text' is the input file
initial, *pairs = re.findall(r'[.#]+', text)
# feeds the first line into 'initial' and the rest of the lines into a list 'pairs'
```