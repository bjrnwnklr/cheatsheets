---
title: Advent of Code 2018 Python cheat sheet
author: Bjoern Winkler
date: 26-01-2019
...

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
  - [Advent of code 2018, day 13](#advent-of-code-2018-day-13)
    - [sorting a list of objects by tuple of object attributes](#sorting-a-list-of-objects-by-tuple-of-object-attributes)
  - [Advent of code 2018, day 20](#advent-of-code-2018-day-20)
    - [Parsing a regex like input string with branches and options](#parsing-a-regex-like-input-string-with-branches-and-options)

## 1. Decoding some lines from input file 

Lines look like

    #1 @ 1,3: 4x4
    #2 @ 3,1: 4x4
    #3 @ 5,5: 2x2

```python
# regex to find all numbers (\d) with 1 or more digits (+) and optional '-' for negative numbers ('-?' (? = 0 or 1))
# map converts them to int tuples
# * unpacks the output
import re

input_file = open(r'filename.txt', 'r')

claims = [[*map(int, re.findall(r'-?\d+', l))] for l in input_file.splitlines() if l]
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

## Advent of code 2018, day 13
### sorting a list of objects by tuple of object attributes
```python
# cr.r = row index
# cr.c = column index
# we want to sort by lowest row first, then lowest column second
# this works with tuples
carts = sorted(carts, key= lambda cr: (cr.r, cr.c))
```

## Advent of code 2018, day 20
### Parsing a regex like input string with branches and options

- `'('` starts a branch, `'|'` is a divider between options, `')'` closes a branch
- parse through the input string and remember positions when encountering a branch - by using a stack

```python
# directions
directions = {
                'N': (0, -1),
                'E': (1, 0),
                'S': (0, 1),
                'W': (-1, 0)
}

# stack to track current position
positions = []
# starting positions
x, y = 5000, 5000
# previous positions
pre_x, pre_y = x, y
# distance tracking dictionary
distances = defaultdict(int)
# starting distance
dist = 0

########## set input
f = open(r'input.txt').read().rstrip()
maze = f

for c in maze[1:-1]:
    print('Char: %s, stack: %d' % (c, len(positions)))
    if c == '(':   # save position if we find a new branch
        positions.append((x, y))
    elif c == ')': # end of branch, pop position before branch
        x, y = positions.pop()
    elif c == '|': # option, go back to last position but leave on stack
                   # until we find the closing ')'
        x, y = positions[-1]
    else:          # process door
        dx, dy = directions[c] # get direction change
        x += dx
        y += dy
        # add to distance (distance for current position)
        if distances[(x, y)] != 0:  # we already have a distance entry - we were here already
            # take the minimum since there is a shorter route to the room
            distances[(x, y)] = min(distances[(x, y)], distances[(pre_x, pre_y)] + 1)
        else:  # new room, add previous distance plus 1
            distances[(x, y)] = distances[(pre_x, pre_y)] + 1

    pre_x, pre_y = x, y
```