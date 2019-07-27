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

# Cleaning up text

Cleaning up text can be complex. [This](https://machinelearningmastery.com/clean-text-machine-learning-python/) has a good summary of the various ways to clean up text.

Clean up is typically
- splitting into words
- converting to lower case
- removing punctuation

## Manual clean up

```python
def cleanup(text):
    # split into words
    review = str(text).strip().split()
    
    # manual removal of punctuation - not really good as it leaves '.' at the end of sentences
    # clean_text = [w for w in review.split() if w not in punctuation]

    # removal with a translation table - much better
    table = str.maketrans('', '', punctuation)
    clean_text = [w.translate(table) for w in review]

    # convert to lower case
    clean_text = [w.lower() for w in clean_text]
    return clean_text
```

Another way of cleaning up is listed [here](http://kavita-ganesan.com/extracting-keywords-from-text-tfidf/):

```python
import re
def pre_process(text):
    
    # lowercase
    text=text.lower()
    
    #remove tags
    text=re.sub("<!--?.*?-->","",text)
    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    return text
```

## Using Tokenizer, Stemmer and Lemmatizer to clean up

Good example [here](https://towardsdatascience.com/a-practitioners-guide-to-natural-language-processing-part-i-processing-understanding-text-9f4abfd13e72) and in the [O'Reilly - Applied Text Analysis with Python book](https://learning.oreilly.com/library/view/applied-text-analysis/9781491963036/ch04.html#ATAP04)

### Tokenizers

You can use:
- `nltk.word_tokenize(text)` - this splits by whitespace and other symbols like quotes (e.g. splits "don't" into "do" and "n't" - so not really useful)
- `nltk.tokenizer.toktok.ToktokTokenizer.tokenize(text)` - this seems to preserve contractions e.g. "don't"

Example:

```python
import nltk.word_tokenize

word = word_tokenize(word)
```

```python
from nltk.tokenizer.toktok import ToktokTokenizer

tokenizer = ToktokTokenizer()
word = tokenizer.tokenize(word)
```

### Stemmer

You can use:
- `nltk.stem.SnowballStemmer`
- `nltk.stem.PorterStemmer`

They all produce word stems that are not phonetically / grammatically correct e.g. "realli" instead of "really". This might be useful in some cases but be aware of it.

Example:

```python
from nltk.stem import SnowballStemmer

def snowball_tokenize(text):
    stem = SnowballStemmer('english')
    text = text.lower()

    for token in tokenizer.tokenize(text):
        if (token in string.punctuation or token in stopwords.words('english')): 
            continue
        yield stem.stem(token)
```

### Lemmatizer

You can use:
- `nltk.stem.WordNetLemmatizer` - slow but produces good results

Example:

```python
def wordnet_tokenize(text):
    lem = WordNetLemmatizer()
    text = text.lower()
    
    for token in tokenizer.tokenize(text):
        if token in string.punctuation or token in stopwords.words('english'): continue
        yield lem.lemmatize(token)
```

# `scikit-learn` with `CountVectorizer` - no clean up required before

The scikit-learn homepage has a [tutorial](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html), which includes using the `CountVectorizer` and the `TfidfTransform` objects.

## Tokenizing text with scikit-learn - CountVectorizer

Text preprocessing, tokenizing and filtering of stopwords are all included in CountVectorizer, which builds a dictionary of features and transforms documents to feature vectors:

```python
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape
```

CountVectorizer supports counts of N-grams of words or consecutive characters. Once fitted, the vectorizer has built a dictionary of feature indices:

```python
count_vect.vocabulary_.get(u'algorithm')
4690
```

### Using CountVectorizer to create a Term Frequency Matrix

Term Frequency Matrix = table with rows (one for each document) and columns (one for each word - _feature_).

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud

# read in and clean up the data
raw_data = pd.read_csv(r'data\1429_1.csv')
# select relevant columns, drop empty value rows, reset index to continouos integers
reviews_df = (raw_data[['reviews.rating', 'reviews.text', 'reviews.title']]
              .dropna()
              .reset_index(drop=True))
reviews_df['comb_text'] = reviews_df['reviews.title'] + ' ' + reviews_df['reviews.text']

# generate list of ratings
ratings = reviews_df['reviews.rating'].value_counts().sort_index().index

# generate count vector
# list of stop words
extra_stp = ['amazon', 'kindle', 'fire', 'prime', 'tablet']
stp = set(stopwords.words('english') + extra_stp)

# create the count vector
count_vector_2 = CountVectorizer(strip_accents='unicode', stop_words=stp)
cv_2 = count_vector_2.fit_transform(reviews_df['comb_text'])

###########
# create term frequency matrix by document
# --- This is the critical step ---
###########
df_cv_2 = pd.DataFrame(cv_2.toarray(), columns=count_vector_2.get_feature_names())

# now do a matrix per rating
# term frequency matrix for the whole corpus (sum all rows for each column)
cv_tfms_2 = {}
for rating in ratings:
    cv_tfms_2[rating] = pd.DataFrame(df_cv_2[reviews_df['reviews.rating'] == rating].sum(axis=0))

    # rename main column
    cv_tfms_2[rating].columns=['term_freq']

# Select words in 95th quantile - i.e. the most frequent 5% of words in each rating category
# generate a word cloud
quantiles_2 = {}
for rating in ratings:
    wc_df_2 = cv_tfms_2[rating]

    # how often does a word at the 95th quantile appear?
    quantiles_2[rating] = int(wc_df_2.quantile(0.95))
    
    # restrict the data frame to only those frequent words
    wc_df_2 = pd.DataFrame(wc_df_2[wc_df_2['term_freq'] >= quantiles_2[rating]])
    
    # generate a word cloud - needs a dictionary of (word: frequency) pairs
    wc_dict_2 = wc_df_2.to_dict()['term_freq']    
    get_wc(wc_dict_2)

# helper function to create a word cloud
def get_wc(term_dict):
    wc = WordCloud(background_color='white').fit_words(term_dict)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc)
```

The index value of a word in the vocabulary is linked to its frequency in the whole training corpus.

## Using TF-IDF to determine key words

TF-IDF stands for `Term frequency * Inverse Document Frequency`. Term frequency is the frequency how often a term appears in a document (divide the number the term appears by the total terms in the document), the inverse document frequency is how often the term appears overall in the corpus.


Both tf and tf–idf can be computed as follows using TfidfTransformer:

```python
from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tf.shape
(2257, 35788)
```

## Combining CountVectorizer and Tfidf analysis

As tf–idf is very often used for text features, there is also another class called TfidfVectorizer that combines all the options of CountVectorizer and TfidfTransformer in a single model:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit_transform(corpus)
...                                
<4x9 sparse matrix of type '<... 'numpy.float64'>'
    with 19 stored elements in Compressed Sparse ... format>
```


There is a good example how to access the results from the TfidfTransformer [here](http://kavita-ganesan.com/tfidftransformer-tfidfvectorizer-usage-differences/).

Easiest is to create a Pandas DataFrame, which is then sorted...

Alternatively, zip the feature names together with the results from the Tfidf matrix and sort again.

```python
# Create the TfidfTransformer, using the CountVectorizer output (which has the frequencies of the words)
tfidf_transformer = TfidfTransformer()
test_idf = tfidf_transformer.fit_transform(word_count_vector)

# get the feature names (i.e. the words) from the word count vector
feature_names = cv.get_feature_names()

# use our first document (review) as an example to see the scores for the words in the document
first_doc = test_idf[1]

# convert into a Pandas DataFrame and display the highest scored words
# - need to use `T` to create a Series as toarray would create a column for each word
key_words = pd.DataFrame(first_doc.T.toarray(), index=feature_names)
key_words.sort_values(by=0, ascending=False)[:15]
```