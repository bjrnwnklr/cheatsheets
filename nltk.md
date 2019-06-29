---
title: nltk cheatsheet
author: Bjoern Winkler
date: 18-June-2019
---

# nltk cheatsheet

# Documentation

[Official website and documentation](https://www.nltk.org/)

[Free version of the NLTK book](http://www.nltk.org/book/) - this has been updated to Python 3 (the version on Safari still is based on Python 2)

# Installing nltk corpora

Install using `pip`, then import and download the book materials (including a few corpora e.g. Moby Dick).

    > pip install nlkt

Run in python prompt, jupyter notebook etc:

```python
import nltk
nltk.download()

```

This starts a tlk interface. Select the `book` identifier and click download. The location needs to be in the standard location specified, otherwise nltk will not be able to find the materials (there are some other paths it checks but keeping the standard location is easiest).

# Playing around with the books

Load the included example books:

```python
from nlkt.book import *
text1 # displays Moby Dick
```

Searching text using `concordance`:

```python
text1.concordance('monstrous')
```

Find similar terms using `similar`:

```python
text1.similar('monstrous')
```

Find common contexts using `common_contexts`:

```python
text1.common_contexts(['monstrous', 'very'])
```

Show a dispersion plot (requires `matplotlib` installed):

```python
text1.dispersion_plot(['monstrous', 'white', 'Ahab'])
```

Generate a random text based on a text corpora:

```python
text1.generate()
```

### Lexical diversity

How often do words appear on average in the text? The lower the number, the more lexically diverse the text (i.e. if a text uses the same words over and over, it is not very lexically diverse and words appear more often.)

```python
def lexical_diversity(text):
    return len(set(text)) / len(text)
```

# Frequency distribution

Frequency distribution can generated with the `FreqDist` object (basically a CountDict):

```python
fdist1 = FreqDist(text1)
```

Printing most common words:

```python
fdist1.most_common(50)
```

Create a plot:

```python
fdist1.plot(50, cumulative=True)
```

# Bigrams and collocations

Bigrams are just word pairs e.g. "of the".

```python
bigrams = bigrams(text4)
```

Collocations are 'frequent bigrams' with a few restrictions (rare words that appear more often than average etc)

Note: this doesn't seem to work in the recent versions of NLTK anymore, but is still listed in the book:

```python
colls = collocations(text4)
```

# Reading own corpus data

Reading in text based corpora using the `PlainTextCorpusReader`:

```python
from nltk.corpus import PlaintextCorpusReader
corpus_root = '/usr/share/dict' [1]
wordlists = PlaintextCorpusReader(corpus_root, '.*') [2]
wordlists.fileids()
wordlists.words('filename')
```

You can then use the typical corpus methods like `words`, `fileids` and `sents`.

# Conditional Frequency Analysis

Conditional frequency analysis is used to count events by conditions, e.g. occurance of words by text category. The `ConditionalFreqDist` function takes `(condition, event)` tuples.

```python
from nltk.corpus import brown
cfd = nltk.ConditionalFreqDist(
    (genre, word)
    for genre in brown.categories()
    for word in brown.words(categories=genre)
)
```
# Using WordNet

WordNet is a semantically-oriented dictionary of English, similar to a traditional thesaurus but with a richer structure. NLTK includes the English WordNet, with 155,287 words and 117,659 synonym sets.

Loading WordNet:

```python
from nltk.corpus import wordnet as wn
```

Synsets are synonym sets i.e. similar words:

```python
wn.synsets('motorcar')
```

Synsets have a format of `car.n.01` - the name of the synset, the type (n = noun) and a running number (if there is more than one synonym).

Synonymous words are called `lemmas`.

```python
wn.synsets('car.n.01').lemmas()
```

Each synset has a number of hierarchical information:

- Definitions:
```python
wn.synsets('car.n.01').definition()
```

- Examples:
```python
wn.synsets('car.n.01').examples()
```

- Hypernyms (up the hierarchy, i.e. more general)
```python
wn.synsets('car.n.01').hypernyms()
```

- Hyponyms (down the hierarchy, more specific)
```python
wn.synsets('car.n.01').hyponyms()
```

