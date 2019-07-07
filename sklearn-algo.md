---
title: scikit-learn algorithm cheatsheet
author: Bjoern Winkler
date: 7-July-2019
---

# Scikit-learn algorithm cheatsheet

My summary cheatsheet of machine learning algorithms in `scikit-learn`.

# Classification algorithms

### _k_-Nearest Neighbor Classification (_k_ NN)

```python
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
```


# Regression algorithms

### _k_-Nearest Neighbor Regression (_k_ NN)

```python
from sklearn.neighbors import KNeighborsRegressor
reg = KNeighborsRegressor(n_neighbors=3)
```

**Usage:** Using this algorithm is a good baseline method to try before considering more advanced techniques. Building the nearest neighbors model is usually very fast, but when your training set is very large (either in number of features or in number of samples) prediction can be slow. When using the k-NN algorithm, it’s important to preprocess your data. This approach often does not perform well on datasets with many features (hundreds or more), and it does particularly badly with datasets where most features are 0 most of the time (so-called sparse datasets).

_While the k-nearest neighbors algorithm is easy to understand, it is not often used in practice, due to prediction being slow and its inability to handle many features._

## Linear algorithms

### Linear regression (Ordinary Least Squares, OLS)

```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X_train, y_train)
print(lr.coef_, lr.intercept_)
```

**Parameters:** None.

### Ridge regression

```python
from sklearn.linear_model import Ridge
ridge = Ridge().fit(X_train, y_train)
```

**Parameters:** 
- `alpha` (default: 1.0): restriction of the coefficients. Increasing `alpha` forces coefficients towars zero, which decreases training set performance but might help generalization.
