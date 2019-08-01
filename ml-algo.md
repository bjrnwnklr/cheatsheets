---
title: Machine Learning Algorithms - the math!
author: Björn Winkler
date: 1-Aug-2019
---

### Note:
This document is following the [Udemy Course - Deep Learning Prequisites - Linear Regression](https://ubs.udemy.com/data-science-linear-regression-in-python/learn/lecture/6444570#overview)

# Linear Regression

## 1-dimensional linear regression

In a 1-dimensional setting, 

- `x` is a column vector of N x 1 (N samples, 1 feature)
- `y` is a column vector of N x 1 (N target values)
- `a` and `b` are scalars

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

- `SSres` is the _Residual sum of squared errors_ - this is calculating "how far away" the predictions are from the target values
- `SStot` is the _Total sum of squared errors_ - this is calculating "how far away" the targets are from their mean (which is a horizontal line)

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

- `X` is a N x 3 matrix (N samples (rows), 2 features, 1 column with 1s for the bias b)
- `y` is a N x 1 vector (N targets)
- `w` is a 3 x 1 vector (3 scalars in a column vector: coefficients w1 and w2 and the bias w0)

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