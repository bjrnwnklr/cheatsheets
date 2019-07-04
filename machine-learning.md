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

# 2. Supervised Learning

## Classification, regression

Two major types of supervised machine learning problems: _classification_ and _regression_.

In _classification_, the goal is to predict a class label, which is a choice from a predefined list of possibilities.

For _regression_ tasks, the goal is to predict a continuous number, or a floating-point number in programming terms (or real number in mathematical terms).

An easy way to distinguish between classification and regression tasks is to ask whether there is some kind of _continuity_ in the output.

## Generalization, overfitting, underfitting

If a model is able to make accurate predictions on unseen data, we say it is able to _generalize_ from the training set to the test set.

Building a model that is too complex for the amount of information we have is called _overfitting_. Overfitting occurs when you fit a model too closely to the particularities of the training set and obtain a model that works well on the training set but is not able to generalize to new data. On the other hand, if your model is too simple you might not be able to capture all the aspects of and variability in the data, and your model will do badly even on the training set. Choosing too simple a model is called _underfitting_.

## Supervised Machine Learning Algorithms

### k-Nearest Neighbors

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

We can also evaluate the model using the `score` method, which for regressors returns the _R2_ score. The _R2_ score, also known as the coefficient of determination, is a measure of goodness of a prediction for a regression model, and yields a score that’s usually between 0 and 1. A value of 1 corresponds to a perfect prediction, and a value of 0 corresponds to a constant model that just predicts the mean of the training set responses, y_train. The formulation of _R2_ used here can even be negative, which can indicate anticorrelated predictions.
