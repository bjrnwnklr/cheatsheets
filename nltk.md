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

