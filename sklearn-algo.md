---
title: scikit-learn algorithm cheatsheet
author: Bjoern Winkler
date: 7-July-2019
---

# Scikit-learn algorithm cheatsheet

My summary cheatsheet of machine learning algorithms in `scikit-learn`.


# Plotting a linear regression / dataset

This will come in handy to plot data, e.g. from the ISLR (Introduction to Statistical Learning) book:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# read csv file - Advertising data. Don't read in the first column, which is an index
df = pd.read_csv('islr-data/Advertising.csv', usecols=range(1, 5))

# get column names
adv_feature_names = list(df.columns)

# calculate coefficient and intercept for X, y data and return linear regression function
def get_lr(X, y):
    lr = LinearRegression().fit(X, y)
    return lambda x: x * lr.coef_[0] + lr.intercept_

# create a plot for TV cs sales
fig, axes = plt.subplots(1, 3, figsize=(12, 5))

# plot each feature against sales and add a linear regression line
for ax, feature in zip(axes, adv_feature_names):
    # get the column data and the sales data
    X, y = df[feature].values, df['sales']
    
    # plot the data points
    ax.plot(X, y, 'ro', fillstyle='none')
    
    # get a linear regression function for the data. Need to reshape from column to row
    # create a continuous line and plot the values
    f = get_lr(X.reshape(-1, 1), y)
    x_line = np.linspace(0, X.max(), 1000)
    ax.plot(x_line, f(x_line))
    
    # set axis labels
    ax.set_ylabel('Sales')
    ax.set_xlabel(feature)
```

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
- `alpha` (default: 1.0): restriction of the coefficients. Increasing `alpha` forces coefficients towards zero, which decreases training set performance but might help generalization. For very small values of `alpha`, RidgeRegression is similar to LinearRegression.

For datasets with very few features, Ridge, Linear and Lasso produce very similar results, regardless of the parameter settings.

### Lasso regression

```python
from sklearn.linear_model import Lasso

lasso = Lasso().fit(X_train, y_train)
print("Number of features used:", np.sum(lasso.coef_ != 0))

lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
```

**Parameters:**
- `alpha` (default: 1.0): restriction of the coefficients. Similarly to Ridge, the Lasso also has a regularization parameter, alpha, that controls how strongly coefficients are pushed toward zero. To _reduce underfitting_, try decreasing alpha. When we do this, we also need to increase the default setting of `max_iter` (the maximum number of iterations to run).

**Usage:** In practice, _ridge regression is usually the first choice between these two models_. However, if you have a large amount of features and expect only a few of them to be important, Lasso might be a better choice.


