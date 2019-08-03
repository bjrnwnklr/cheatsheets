---
title: Machine Learning Algorithms - the math!
author: Björn Winkler
date: 3-Aug-2019
---

### Note:

This document is following the [Udemy Course - Deep Learning Prequisites - Linear Regression](https://ubs.udemy.com/data-science-linear-regression-in-python/learn/lecture/6444570#overview)

# Linear Regression

## 1-dimensional linear regression

In a 1-dimensional setting,

-   `x` is a column vector of N x 1 (N samples, 1 feature)
-   `y` is a column vector of N x 1 (N target values)
-   `a` and `b` are scalars

Linear equations follow the formula $y = ax + b$.

$a$ and $b$ can be calculated by:

$$
\begin{aligned}
a = \frac{x^Ty - \overline{y} \sum_{i=1}^Nx_i}{x^Tx - \overline{x} \sum_{i=1}^Nx_i} \\
b = \frac{\overline{y} x^Tx - \overline{x} x^Ty}{x^Tx - \overline{x} \sum_{i=1}^Nx_i}
\end{aligned}
$$

The prediction $\hat y$ is calculated as

$$
\hat y = ax + b
$$

```python
# calc the denominator
denominator = x.dot(x) - np.mean(x) * np.sum(x)

# calulate a
a = (x.dot(y) - np.mean(y) * np.sum(x)) / denominator

# calculate b
b = ((np.mean(y) * x.dot(x)) - np.mean(x) * x.dot(y)) / denominator

# calculate the prediction
yhat = a * x + b
```

### Calculate $R^2$ (R-squared)

-   `SSres` is the _Residual sum of squared errors_ - this is calculating "how far away" the predictions are from the target values
-   `SStot` is the _Total sum of squared errors_ - this is calculating "how far away" the targets are from their mean (which is a horizontal line)

This means that `SSres / SStot` is comparing how much better the prediction is than the mean (which is arguably the worst but easiest estimate).

`SSres` should be better than the mean, so `SSres / SStot` is typically < 1.

$$
\begin{aligned}
SS_{res} &= \sum_{i=1}^N (y_i - \hat y_i)^2 \\
SS_{tot} &= \sum_{i=1}^N (y_i - \overline{y})^2 \\
R^2 &= 1 - \frac{SS_{res}}{SS_{tot}} \\
&= 1 - \frac{\sum_{i=1}^N (y_i - \hat y_i)^2}{\sum_{i=1}^N (y_i - \overline{y})^2}
\end{aligned}
$$

```python
res = y - yhat
tot = y - np.mean(y)

ssres = res.dot(res)
sstot = tot.dot(tot)

rsquared = 1 - ssres / sstot
```

## 2-Dimensional Linear Regression

In a 2-dimensional setting,

-   `X` is a N x 3 matrix (N samples (rows), 2 features, 1 column with 1s for the bias b)
-   `y` is a N x 1 vector (N targets)
-   `w` is a 3 x 1 vector (3 scalars in a column vector: coefficients w1 and w2 and the bias w0)

This uses the `np.linalg.solve()` function to solve as it is more efficient than calculating `w` with the inverse of X.dot(X).

$$
\begin{aligned}
\vec{w} &= (\mathbf{X}^T\mathbf{X})^{-1} \mathbf{X}^T\vec{y} \\
\hat y &= \mathbf{X}^T \vec{w}
\end{aligned}
$$

```python
w = np.linalg.solve(X.T.dot(X), X.T.dot(y))

# ## Calculating yhat
yhat = X.dot(w)
```

$R^2$ is calculated exactly the same way as in a 1 dimensional setting (`y` and `yhat` are vectors).

```python
# ## Calculation R^2
res = y - yhat
tot = y - np.mean(y)

SSres = res.dot(res)
SStot = tot.dot(tot)
r_squared = 1 - (SSres / SStot)
```

## Ridge Regression

A good overview of Ridge Regression (and Lasso Regression) can be found [here](https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-ridge-lasso-regression-python/).

**Ridge Regression** adds _regularization_ to normal linear regression by penalizing large weights by adding their squared magnitude to the cost.

Both Ridge and Lasso work by penalizing the magnitude of coefficients of features along with minimizing the error between predicted and actual observations. These are called ‘regularization’ techniques. The key difference is in how they assign penalty to the coefficients:

**Ridge Regression:**

-   Performs **L2 regularization**, i.e. adds penalty equivalent to _square of the magnitude_ of coefficients

    Minimization objective = LS Obj + α \* (sum of square of coefficients)

**Lasso Regression:**

-   Performs **L1 regularization**, i.e. adds penalty equivalent to _absolute value of the magnitude_ of coefficients

    Minimization objective = LS Obj + α \* (sum of absolute value of coefficients)

Note that here ‘LS Obj’ refers to ‘least squares objective’, i.e. the linear regression objective without regularization.

Cost function:

$$
\begin{aligned}
J &= \sum_{n=1}^N (y_n - \hat{y}_n)^2 + \lambda \vert{\vec{w}}\vert^2
\end{aligned}
$$

where $\vert \vec{w} \vert$ is the magnitude of the paramater (coefficient) vector $w$, and $\lambda$ is a factor that determines the restriction of the coefficients. Increasing $\lambda$ forces coefficients towards zero, which decreases training set performance but might help generalization. For very small values of $\lambda$, RidgeRegression is similar to LinearRegression.

By solving for the minimum of the derivative of $J$, $w$ can be calculated as:

$$
\begin{aligned}
\vec{w} &= (\lambda \mathbf{I} + \mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \vec{y} \\
\hat{y} &= \mathbf{X}^T \vec{w}
\end{aligned}
$$

In Python:

```python
# lambda or alpha...
l2 = 1000.0
w = np.linalg.solve(l2 * np.eye(2) + X.T.dot(X), X.T.dot(Y))

# calculate yhat just like normal linear regression
yhat = X.dot(w)
```

`l2 * np.eye(2)` generates a 2x2 matrix with `l2` in the diagonal. This is required to be a 2x2 matrix since `X.T.dot(X)` ($\mathbf{X}^T \mathbf{X}$) results in a 2x2 matrix as well.

Generally, the identity matrix has to be $D \times D$ ($D$ being the number of features _including the bias_), since $X^TX$ results in a $D \times D$ matrix ($X^T$ being $D \times N$, $X$ being $N \times D$).
