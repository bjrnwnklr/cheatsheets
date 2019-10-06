---
title: scikit-learn algorithm cheatsheet
author: Bjoern Winkler
date: 7-July-2019
---

# Scikit-learn algorithm cheatsheet

My summary cheatsheet of machine learning algorithms in `scikit-learn`.

# Plotting

Some handy ways to plot data, e.g. for classification or regression data.

## Plotting a linear regression / dataset

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

## Plotting a decision boundary

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


# define some random data and apply a scaler
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=3)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

cm = plt.cm.coolwarm

# plot the training data as solid, the test data as light dots
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm, alpha=0.6)

# plot the decision boundary for KNeighbors, with neighbors from 1-10
n = 10
# stepsize in mesh
h = 0.2

fig, axes = plt.subplots(10, 1, figsize=(4, 40))

for i, ax in zip(range(1, n+1), axes):
    # define the classifier
    clf = KNeighborsClassifier(n_neighbors=i)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    # create a meshgrid of 2D coordinates
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # ravel() flattens a multi dimensional array into 1D
    # np.c_ appends the two 1D arrays into a matrix with 2 columns
    # [:, 1] takes the 2nd column from the result, representing the probability of being in class 1
    # (first column is the probability of being in class 0), which would swap the colors
    #
    # Some classification models have a predict_proba function, some others use 'decision_function'
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # reshape Z to the same shape as xx
    Z = Z.reshape(xx.shape)

    # plot the results
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.7)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm, edgecolors='k')
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm, edgecolors='k', alpha=0.6)
    ax.text(x_max - 0.3, y_min + 0.3, score, horizontalalignment='right', size=15)
    ax.set_title('{} neighbors'.format(i))
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

-   `alpha` (default: 1.0): restriction of the coefficients. Increasing `alpha` forces coefficients towards zero, which decreases training set performance but might help generalization. For very small values of `alpha`, RidgeRegression is similar to LinearRegression.

For datasets with very few features, Ridge, Linear and Lasso produce very similar results, regardless of the parameter settings.

### Lasso regression

```python
from sklearn.linear_model import Lasso

lasso = Lasso().fit(X_train, y_train)
print("Number of features used:", np.sum(lasso.coef_ != 0))

lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
```

**Parameters:**

-   `alpha` (default: 1.0): restriction of the coefficients. Similarly to Ridge, the Lasso also has a regularization parameter, alpha, that controls how strongly coefficients are pushed toward zero. To _reduce underfitting_, try decreasing alpha. When we do this, we also need to increase the default setting of `max_iter` (the maximum number of iterations to run).

**Usage:** In practice, _ridge regression is usually the first choice between these two models_. However, if you have a large amount of features and expect only a few of them to be important, Lasso might be a better choice.

# Classification algorithms

## Linear algorithms

### Logistic regression

## Decision trees

```python
from sklearn.tree import DecisionTreeClassifier

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))
```

**Usage:** The parameters that control model complexity in decision trees are the pre-pruning parameters that stop the building of the tree before it is fully developed. Usually, picking one of the pre-pruning strategies — setting either `max_depth`, `max_leaf_nodes`, or `min_samples_leaf` — is sufficient to prevent overfitting.

Decision trees have two advantages over many of the algorithms we’ve discussed so far: _the resulting model can easily be visualized_ and understood by nonexperts (at least for smaller trees), and _the algorithms are completely invariant to scaling of the data_. As each feature is processed separately, and the possible splits of the data don’t depend on scaling, _no preprocessing like normalization or standardization of features is needed_ for decision tree algorithms. In particular, decision trees work well when you have features that are on completely different scales, or a mix of binary and continuous features.

The _main downside of decision trees_ is that even with the use of pre-pruning, _they tend to overfit and provide poor generalization performance_. Therefore, in most applications, the ensemble methods we discuss next are usually used in place of a single decision tree.

## Ensembles

### RandomForests

```python
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))
```

**Parameters:** The important parameters to adjust are `n_estimators`, `max_features`, and possibly pre-pruning options like `max_depth`. For `n_estimators`, larger is always better. Averaging more trees will yield a more robust ensemble by reducing overfitting. However, there are diminishing returns, and more trees need more memory and more time to train. A common rule of thumb is to build “as many as you have time/memory for.”

As described earlier, `max_features` determines how random each tree is, and a smaller `max_features` reduces overfitting. In general, it’s a good rule of thumb to use the default values: `max_features=sqrt(n_features)` for classification and `max_features=n_features` for regression. Adding `max_features` or `max_leaf_nodes` might sometimes improve performance. It can also drastically reduce space and time requirements for training and prediction.

**Usage:** Random forests for regression and classification are currently among the most widely used machine learning methods. They are very powerful, often work well without heavy tuning of the parameters, and don’t require scaling of the data.

Random forests _don’t tend to perform well on very high dimensional, sparse data, such as text data_. For this kind of data, linear models might be more appropriate. Random forests _usually work well even on very large datasets_, and training can easily be parallelized over many CPU cores within a powerful computer. However, random forests require more memory and are slower to train and to predict than linear models. If time and memory are important in an application, it might make sense to use a linear model instead.
