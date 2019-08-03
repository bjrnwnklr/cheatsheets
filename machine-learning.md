---
title: Machine Learning with scikit-learn cheatsheet
author: Bjoern Winkler
date: 7-July-2019
---

# Machine Learning with scikit-learn

Following the "Intro to Machine Learning with Python" book

# 1. Intro

Let’s summarize what we learned in this chapter. We started with a brief introduction to machine learning and its applications, then discussed the distinction between supervised and unsupervised learning and gave an overview of the tools we’ll be using in this book. Then, we formulated the task of predicting which species of iris a particular flower belongs to by using physical measurements of the flower. We used a dataset of measurements that was annotated by an expert with the correct species to build our model, making this a supervised learning task. There were three possible species, setosa, versicolor, or virginica, which made the task a three-class classification problem. The possible species are called _classes_ in the classification problem, and the species of a single iris is called its _label_.

The Iris dataset consists of two NumPy arrays: one containing the data, which is referred to as `X` in `scikit-learn`, and one containing the correct or desired outputs, which is called `y`. The array `X` is a two-dimensional array of features, with one row per data point and one column per feature. The array `y` is a one-dimensional array, which here contains one class label, an integer ranging from 0 to 2, for each of the samples.

We split our dataset into a _training set_, to build our model, and a _test set_, to evaluate how well our model will generalize to new, previously unseen data.

We chose the _k-nearest neighbors_ classification algorithm, which makes predictions for a new data point by considering its closest neighbor(s) in the training set. This is implemented in the `KNeighborsClassifier` class, which contains the algorithm that builds the model as well as the algorithm that makes a prediction using the model. We instantiated the class, setting parameters. Then we built the model by calling the `fit` method, passing the training data (`X_train`) and training outputs (`y_train`) as parameters. We evaluated the model using the `score` method, which computes the accuracy of the model. We applied the `score` method to the test set data and the test set labels and found that our model is about 97% accurate, meaning it is correct 97% of the time on the test set.

This gave us the confidence to apply the model to new data (in our example, new flower measurements) and trust that the model will be correct about 97% of the time.

Here is a summary of the code needed for the whole training and evaluation procedure:

```python
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))
```

Three main methods to remember:

`fit`, `predict` and `score`.

# 2. Statistical Learning - key terms

With some terms from the ISLR book.

## Classification, regression

Two major types of supervised machine learning problems: _classification_ and _regression_.

In _classification_, the goal is to predict a class label, which is a choice from a predefined list of possibilities.

For _regression_ tasks, the goal is to predict a continuous number, or a floating-point number in programming terms (or real number in mathematical terms).

An easy way to distinguish between classification and regression tasks is to ask whether there is some kind of _continuity_ in the output.

## Generalization, overfitting, underfitting

If a model is able to make accurate predictions on unseen data, we say it is able to _generalize_ from the training set to the test set.

Building a model that is too complex for the amount of information we have is called _overfitting_. Overfitting occurs when you fit a model too closely to the particularities of the training set and obtain a model that works well on the training set but is not able to generalize to new data. The model is _following the noise too closely._ On the other hand, if your model is too simple you might not be able to capture all the aspects of and variability in the data, and your model will do badly even on the training set. Choosing too simple a model is called _underfitting_.

## MSE, variance, bias

### MSE

MSE = _Mean Squared Error_ - mostly used in Regression settings. _Training MSE_ describes the accuracy of the training model (individual predictions vs the actual training data). _Test MSE_ describes the accuracy of the predictions that are obtained when applying the method to previously unseen test data.

The problem is that _many statistical methods specifically estimate coefficients so as to minimize the training set MSE_.

As _model flexibility increases_, training MSE will decrease, but the test MSE may not. _Overfitting_ happens when a method produces a small training MSE but a large test MSE.

### Variance and bias

The expected test MSE can be decomposed into the sum of three fundamental quantities: the _variance_ of the prediction _f^(x0)_, the squared _bias_ of _f^(x0)_ and the variance of the error terms.

Since both variance and the squared bias are positive, the test MSE can never be less than the _irreducible error_ (ther variance of the error term here).

_Variance_ refers to the amount by which the prediction function would change if we estimated it using a different training data set. In general, _more flexible statistical methods have higher variance_ (as the method will try to fit each individual training observation more closely compared to a inflexible method like linear regression).

_Bias_ refers to the error that is introduced by approximating a real-life problem, which may be extremely complicated, by a much simpler model. Linear regression (a very simple model) can result in very high bias if the real life problem is not linear at all. Generally, _more flexible methods result in less bias_.

As a general rule, as we use more flexible methods, the variance will increase and the bias will decrease.

# Supervised Machine Learning Algorithms

## k-Nearest Neighbors

The `k-NN algorithm` is arguably the simplest machine learning algorithm. Building the model consists only of storing the training dataset. To make a prediction for a new data point, the algorithm finds the closest data points in the training dataset—its “nearest neighbors.”

K-NEIGHBORS CLASSIFICATION

In its simplest version, the k-NN algorithm only considers exactly one nearest neighbor, which is the closest training data point to the point we want to make a prediction for. The prediction is then simply the known output for this training point.

Instead of considering only the closest neighbor, we can also consider an arbitrary number, _k_, of neighbors. This is where the name of the k-nearest neighbors algorithm comes from. When considering more than one neighbor, we use _voting_ to assign a label. This means that for each test point, we count how many neighbors belong to class 0 and how many neighbors belong to class 1. We then assign the class that is more frequent: in other words, the majority class among the k-nearest neighbors.

Example:

```python
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)

# fit the training data
clf.fit(X_train, y_train)

# make a prediction
clf.predict(X_test)

# score the test data
clf.score(X_test, y_test)
```

K-NEIGHBORS REGRESSION

When using multiple nearest neighbors, the prediction is the average, or mean, of the relevant neighbors.

The k-nearest neighbors algorithm for regression is implemented in the `KNeighborsRegressor` class in scikit-learn. It’s used similarly to `KNeighborsClassifier`.

We can also evaluate the model using the `score` method, which for regressors returns the _R2_ score. The _R2_ score, also known as the coefficient of determination, is a measure of goodness of a prediction for a regression model, and yields a score that’s usually between 0 and 1. A value of 1 corresponds to a perfect prediction, and a value of 0 corresponds to a constant model that just predicts the mean of the training set responses, y*train. The formulation of \_R2* used here can even be negative, which can indicate anticorrelated predictions.

Example:

```python
from sklearn.neighbors import KNeighborsRegressor

# split the wave dataset into a training and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# instantiate the model and set the number of neighbors to consider to 3
reg = KNeighborsRegressor(n_neighbors=3)
# fit the model using the training data and training targets
reg.fit(X_train, y_train)
reg.score(X_test, y_test)
```

One of the strengths of k-NN is that the model is very easy to understand, and often gives reasonable performance without a lot of adjustments. Using this algorithm is a good baseline method to try before considering more advanced techniques. Building the nearest neighbors model is usually very fast, but when your training set is very large (either in number of features or in number of samples) prediction can be slow. When using the k-NN algorithm, it’s important to preprocess your data. This approach often does not perform well on datasets with many features (hundreds or more), and it does particularly badly with datasets where most features are 0 most of the time (so-called sparse datasets).

So, while the k-nearest neighbors algorithm is easy to understand, it is not often used in practice, due to prediction being slow and its inability to handle many features.

## Linear models

Linear models make a prediction using a linear function of the input features.

### LINEAR MODELS FOR REGRESSION

For regression, the general prediction formula for a linear model looks as follows:

$$
\begin{aligned}
\hat{y} &= w_0 * x_0 + w_1 * x_1 + ... + w_D * x_D + b \\
\end{aligned}
$$

Here, $x_0$ to $x_D$ denotes the _features_ (in this example, the number of features is $D + 1$) of a single data point, $w$ and $w$ are parameters of the model that are learned, and $\hat{y}$ is the prediction the model makes.

Linear algorithms try to determine the $w$ and $b$ parameters. $w$ are called _coefficients_ (`coef_` as output) and $b$ is called _intercept_ (`intercept_` as output).

For datasets with many features, linear models can be very powerful. In particular, if you have more features than training data points, any target y can be perfectly modeled (on the training set) as a linear function.

There are many different linear models for regression. The difference between these models lies in how the model parameters $w$ and $b$ are learned from the training data, and how model complexity can be controlled.

### Linear regression (aka Ordinary Least Squares, OLS)

Linear regression, or ordinary least squares (OLS), is the simplest and most classic linear method for regression. Linear regression finds the parameters $w$ and $b$ that minimize the mean squared error between predictions and the true regression targets, $y$, on the training set. The mean squared error is the sum of the squared differences between the predictions and the true values, divided by the number of samples. Linear regression has no parameters, which is a benefit, but it also has no way to control model complexity.

```python
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lr = LinearRegression().fit(X_train, y_train)

print(lr.coef_, lr.intercept_)
```

### Ridge regression

Ridge regression is also a linear model for regression, so the formula it uses to make predictions is the same one used for ordinary least squares. In ridge regression, though, the coefficients (w) are chosen not only so that they predict well on the training data, but also to fit an additional constraint. We also want the magnitude of coefficients to be as small as possible; in other words, _all entries of w should be close to zero_. Intuitively, this means each feature should have as little effect on the outcome as possible (which translates to having a small slope), while still predicting well. This constraint is an example of what is called _regularization_. Regularization means explicitly restricting a model to avoid overfitting. The particular kind used by ridge regression is known as L2 regularization.

```python
from sklearn.linear_model import Ridge

ridge = Ridge().fit(X_train, y_train)
```

How much importance the model places on simplicity versus training set performance can be specified by the user, using the `alpha` parameter. In the previous example, we used the default parameter `alpha=1.0`. There is no reason why this will give us the best trade-off, though. The optimum setting of alpha depends on the particular dataset we are using. Increasing alpha forces coefficients to move more toward zero, which decreases training set performance but might help generalization.

For very small values of alpha, coefficients are barely restricted at all, and we end up with a model that resembles LinearRegression.

Since `alpha` corresponds to the model complexity, for very simple models i.e. with only a few features, the `alpha` parameter does not have any impact on the results / scoring.

### Lasso regression

An alternative to Ridge for regularizing linear regression is Lasso. As with ridge regression, the lasso also restricts coefficients to be close to zero, but in a slightly different way, called _L1 regularization_. The consequence of L1 regularization is that when using the lasso, some coefficients are exactly zero. This means some features are entirely ignored by the model. This can be seen as a form of automatic feature selection. Having some coefficients be exactly zero often makes a model easier to interpret, and can reveal the most important features of your model.

```python
from sklearn.linear_model import Lasso

lasso = Lasso().fit(X_train, y_train)
print("Number of features used:", np.sum(lasso.coef_ != 0))
```

Similarly, if you would like to have a model that is easy to interpret, Lasso will provide a model that is easier to understand, as it will select only a subset of the input features. scikit-learn also provides the `ElasticNet` class, which combines the penalties of Lasso and Ridge. In practice, this combination works best, though at the price of having two parameters to adjust: one for the L1 regularization, and one for the L2 regularization.
