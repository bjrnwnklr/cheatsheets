---
title: Machine Learning Algorithms - the math!
author: Björn Winkler
date: 10-Aug-2019
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

## Gradient Descent

Taking the cost function derivative, setting to 0 and solving for $w$ is in most cases difficult. For Linear Regression, it is simple, but for other machine learning models it is complex or resource intensive. _Gradient Descent_ is a method to estimate the parameters $w$ by minimizing the cost function $J(w)$ by iteratively updating $w$ in the direction of $\frac{\partial J(w)}{\partial w}$ in small steps.

$w$ is set to a random initial value.

The formula for _Gradient Descent_ is

$$
\begin{aligned}
w \leftarrow w - {"learning rate"} \times \frac{\partial J(w)}{\partial w}
\end{aligned}
$$

Repeat this over enough iterations.

**Learning rate**: this is called a _hyperparameter_ since it is not part of the linear regression model, but only required to solve the gradient descent. Finding the right learning rate is one of the most difficult things in Machine Learning and requires practice.

If the _learning rate_ is not set correctly, the iterative values of $w$ will oscillate between the sides of the cost function instead of descending towards the minimum.

### Gradient Descent for Linear Regression

The cost function for Linear Regression is:

$$
\begin{aligned}
J(w) &= (Y - \hat{Y})^T(Y - \hat{Y})  \\
&= (Y - Xw)^T(Y - Xw) \\
\end{aligned}
$$

with $\hat{Y} = Xw$

The _Gradient_ (or derivative) of this is:

$$
\begin{aligned}
\frac{\partial J(w)}{\partial w} &= -2X^TY + 2X^TXw \\
&= 2X^T(Xw - Y) \\
&= 2X^T(\hat{Y} - Y)
\end{aligned}
$$

Instead of setting this derivative to 0 and solving for $w$, we will just take small steps in this direction. We can also drop the factor $2$ since it is just a constant and can be absorbed into the learning rate.

The algorithm for solving linear regression using gradient descent:

-   set $w$ to a sample from $N(0, \frac{1}{D})$ (Gaussian normal distribution, centered at 0 and variance of 1/D with D the dimensionality of the data)
-   loop for `t = 1..T:` `w = w - learningRate * X.T.dot(X.dot(w) - Y)`

$$
\begin{aligned}
w = w - "learningRate" \times X^T(Xw - Y)
\end{aligned}
$$

Why is $w$ coming from the Gaussian normal distribution like that?

-   If you normalize your x to N(0, 1) (which is common),
-   you would like your y to also be N(0, 1) (common in the sequels to this course),
-   Then having w be N(0, 1/D) ensures that y is N(0, 1):
-   var(y) = var(x1)var(w1) + var(x2)var(w2) + ... + var(xD)var(wD) = 1/D + 1/D + ... = 1

You can do this in Python by:

```python
w = np.random.randn(D) / np.sqrt(D)
```

**Gradient Descent for Linear Regression - algorithm In Python:**

Calculate the _mean squared error_ for each iteration and see that it decreases.

```python
# prep the data - read from DataFrame
# add a column with bias to the left of X
X = df[['x1', 'x2']].to_numpy()
X = np.c_[np.ones(len(df['x1'])), X]
Y = df['y'].to_numpy()

rate = 0.0000031582     # learning rate
T = 300000              # number of epochs

# initialize w (need 3 values, including 1 for the bias)
w = np.random.randn(3) / np.sqrt(3)

costs = []
for i in range(T):
    delta = X.dot(w) - Y
    w = w - rate * X.T.dot(delta)
    mse = delta.dot(delta) / len(X[:,0])
    costs.append(mse)

plt.plot(costs)
plt.show()
```

With:

-   X: N x D matrix of features (D includes a column of 1s for the bias)
-   Y: N x 1 vector of targets
-   w: D x 1 vector of parameters - assign random values from `np.random.randn(D)`

## Regularization

A good overview of **regularization techniques** like Ridge Regression and Lasso Regression can be found [here](https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-ridge-lasso-regression-python/).

**Ridge Regression** adds _regularization_ to normal linear regression by penalizing large weights by adding their squared magnitude to the cost.

Both Ridge and Lasso work by penalizing the magnitude of coefficients of features along with minimizing the error between predicted and actual observations. These are called ‘regularization’ techniques. The key difference is in how they assign penalty to the coefficients:

**Ridge Regression:**

-   Performs **L2 regularization**, i.e. adds penalty equivalent to _square of the magnitude_ of coefficients

    Minimization objective = LS Obj + α \* (sum of square of coefficients)

    L2 regularization uses the _L2 norm_ for the penalty term of the cost function.

    L2 norm is the Euclidian distance:

    $$
    \begin{aligned}
    \vert v \vert_2 &= \sqrt{v_1^2 + v_2^2 + v_3^2}
    \end{aligned}
    $$

**Lasso Regression:**

-   Performs **L1 regularization**, i.e. adds penalty equivalent to _absolute value of the magnitude_ of coefficients

    Minimization objective = LS Obj + α \* (sum of absolute value of coefficients)

    L1 regularization uses the _L1 norm_ for the penalty term of the cost function.

    L1 norm is the Manhattan or Taxi distance:

    $$
    \begin{aligned}
    \vert v \vert_1 &= \vert v_1 \vert + \vert v_2 \vert + \vert v_3 \vert
    \end{aligned}
    $$

Note that here ‘LS Obj’ refers to ‘least squares objective’, i.e. the linear regression objective without regularization.

## Ridge Regression (L2 Regularization)

**L2 regularization cost function:**

$$
\begin{aligned}
J &= \sum_{n=1}^N (y_n - \hat{y}_n)^2 + \lambda \vert{\vec{w}}\vert_2^2
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

### L2 Regularization using Gradient Descent

**L2 regularization cost function:**

$$
\begin{aligned}
J_{Ridge} &= \sum_{n=1}^N (y_n - \hat{y}_n)^2 + \lambda \vert{\vec{w}}\vert_2^2 \\
&= (Y - Xw)^T (Y - Xw) + \lambda w^Tw \\
&= Y^T Y - Y^T X w - w^T X^T Y - w^T X^T X w + \lambda w^T w \\
&= Y^T Y - 2 Y^T X w - w^T X^T X w + \lambda w^T w
\end{aligned}
$$

The derivative of the cost function is:

$$
\begin{aligned}
\frac{\partial J}{\partial w} &= -2X^T (Y - Xw) + 2 \lambda w \\
&= X^T (Xw - Y) + \lambda w
\end{aligned}
$$

The factors $2$ can be dropped as they will be absorbed into the learning rate.

Gradient descent algorithm:

$$
\begin{aligned}
w \leftarrow w - learningRate \times (X^T(Xw - Y) + \lambda w)
\end{aligned}
$$

In Python:

```python
N = len(X[:, 1])    # number of samples
D = X.shape[1]      # number of features (incl the bias column)
T = 10000           # number of epochs
rate = 0.00003      # learning rate
alpha = 10          # regularization parameter

# set w to random starting parameters
w = np.random.randn(D) / np.sqrt(D)

costs = []

for t in range(T):
    delta = X.dot(w) - Y
    w = w - rate * (X.T.dot(delta) + alpha * w)
    mse = delta.dot(delta) / N
    costs.append(mse)
```

## Lasso Regression (L1 Regularization)

In general, we prefer a "skinny" matrix of X - so that D << N (# features << # samples) vs. a "fat" matrix where the number of features is equal to or more than the number of samples.

This is called _sparcity_ - a small number of important features predict the target. Most weights are zero, only a few weights are non-zero.

This is achieved through **L1 regularization**.

**L1 regularization cost function:**

$$
\begin{aligned}
J_{Lasso} &= \sum_{n=1}^N (y_n - \hat{y}_n)^2 + \lambda \vert w \vert_1 \\
&= (Y - Xw)^T (Y - Xw) + \lambda \vert w \vert \\
&= Y^T Y - 2 Y^T Xw + w^T X^T Xw + \lambda \vert w \vert \\
\end{aligned}
$$

The derivative of the cost function is:

$$
\begin{aligned}
\frac{\partial J}{\partial w} &= -2 X^T Y + 2 X^T Xw + \lambda sign(w) \\
&= X^T (X w - Y) + \lambda sign(w)
\end{aligned}
$$

The $sign(w)$ function can't be solved for $w$ as it is not reversible.

Hence, **_Gradient Descent_ has to be used to determine $w$ for L1 regularization.**

Gradient descent algorithm:

$$
\begin{aligned}
w \leftarrow w - learningRate \times (X^T(Xw - Y) + \lambda \times sign(w))
\end{aligned}
$$

In Python:

```python
N = len(X[:, 1])    # number of samples
D = X.shape[1]      # number of features (incl the bias column)
T = 10000           # number of epochs
rate = 0.00003      # learning rate
alpha = 10          # regularization parameter

# set w to random starting parameters
w = np.random.randn(D) / np.sqrt(D)

costs = []

for t in range(T):
    delta = X.dot(w) - Y
    w = w - rate * (X.T.dot(delta) + alpha * np.sign(w))
    mse = delta.dot(delta) / N
    costs.append(mse)


plt.plot(costs)
plt.show()
```

# Logistic Regression

## The Logistic Regression model

Logistic regression makes the assumption that the _data can be separated by a **line or plane**_, i.e. by a linear function.

This is achieved by using a **sigmoid function**. This can be either in the form of the hyperbolic tangent function $\tanh(x) \isin (-1, 1), y_{intercept} = 0$. We use the following sigmoid function:

$$
\begin{aligned}
\sigma(z) &= \frac{1}{1 + e^{(-z)}} \isin (0, 1), y_{intercept} = 0.5 \\
\text{with}\nobreakspace z &= w^T X
\end{aligned}
$$

This creates a function that's $0$ for large negative values of $z$, $1$ for large positive values of $z$ and $0.5$ for $z = 0$.

Hence this can be used easily as a linear classifier:

$$
\begin{aligned}
\sigma(w^T X) &> 0.5 \rightarrow 1 \\
\sigma(w^T X) &< 0.5 \rightarrow 0 \\
\end{aligned}
$$

**Predictions** can be made by

$$
\begin{aligned}
\hat{y} = \sigma(w^T X)
\end{aligned}
$$

In Python:

```python
def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def forward(X, w, b):
    return sigmoid(X.dot(w) + b)

# probability of Y given X, using our sigmoid function and random weights
P_Y_given_X = forward(X, w, b)

# make predictions (binary) by rounding
predictions = np.round(P_Y_given_X)

# compare the results
def classification_rate(Y, P):
    return np.mean(Y == P)

print('Classification rate: {}'.format(classification_rate(Y, predictions)))
```

### Closed form solution using the Bayes classifier

If 2 classed have a _Gaussian distribution_ and the same _covariance_ and _different means_ (e.g. two blobs), you can derive a closed form solution using the Bayes classifier.

Gaussian probability distribution function (PDF):

$$
\begin{aligned}
P(x) &= \frac{1}{\sqrt{(2\pi)^D \vert\Sigma\vert}} e^{-\frac{1}{2}(x - \mu)^T \Sigma^{-1}(x - \mu)}
\end{aligned}
$$

with

$$
\begin{aligned}
    \pi &: \\
    \Sigma &: \\
    \mu &: \\
\end{aligned}
$$

**Bayes rule**:

$$
\begin{aligned}
    P(Y \vert X) &= \frac{P(X \vert Y) P(Y)}{P(X)}
\end{aligned}
$$

with

$$
\begin{aligned}
    P(Y \vert X) &= \text{probability of Y given X ("the posterior")} \\
    P(Y) &= \text{probability of Y ("the prior"), the frequency with which class Y appears} \\
    P(Y=1) &= \text{number of times class 1 appears / total samples} \\
\end{aligned}
$$

Putting Bayes rule into the logistic regression framework gives us the closed form solution:

$$
\begin{aligned}
    w^T &= (\mu^T_1 - \mu^T_0) \Sigma^{-1} \\
    b &= \frac{1}{2}\mu^T_0 \Sigma^{-1} \mu_0 - \frac{1}{2}\mu^T_1 \Sigma^{-1} \mu_1 - \ln\frac{\alpha}{1-\alpha}
\end{aligned}
$$

## Cross Entropy Error function

Linear regression uses the squared error function (sum of squared errors (target - prediction)). This assumes a Gaussian distributed error, because log(Gaussian) = squared function.

Logistic Regression is not Gaussian distributed because

-   Target is either 0 or 1 ($t_n$)
-   Output is always a number between 0 and 1 ($y_n$)

So the required error function has to be:

-   0 if correct
-   $> 0$ if not correct, more wrong = bigger cost

**The Cross Entropy Error**

$$
\begin{aligned}
    J &= -(t \log(y) +  (1-t)\log(1-y))
\end{aligned}
$$

Depending on target $t$, only one of the terms matters:

-   if t=1, only $t \log(y)$ matters, as $(1-t) = 0$
-   if t=0, only $(1-t) \log(1-y)$ matters

$log(y)$ is a number between 0 and $-\infty$ ($\log(0) = -\infty$, $log(1)=0$). So we take the negative, making the result between 0 and $\infty$.

To get the error function, we sum up over the training data:

$$
\begin{aligned}
    J &= -\sum_{n=1}^N t_n \log(y_n) + (1-t_n)\log(1-y_n)
\end{aligned}
$$

Cross Entropy Error function in Python:

```python
def cross_entropy_2(T, Y):
    E = -1 * (T.dot(np.log(Y)) + (1-T).dot(np.log(1-Y)))
    return E
```

### Gradient Descent for Logistic Regression

We can use Bayes method to find the _weights_ if the data is Gaussian distributed with equal covariance ($w^T = (\mu_1^T - \mu_0^T) \Sigma^{-1}$ etc), but we want something that works in general. Apparently we can't just solve the derivative for weights by setting to zero.

So to find the derivative of J for w, we split the cost function J into 3 derivatives and multiply them together by the chain rule:

$$
\begin{aligned}
    \frac{\partial J}{\partial w_i} &= \sum_{n=1}^N \frac{\partial J}{\partial y_n} \frac{\partial y_n}{\partial a_n} \frac{\partial a_n}{\partial w_i} \\
    a_n &= w^T x_n
\end{aligned}
$$

This gets us the following vectorized derivative for w:

$$
\begin{aligned}
    \frac{\partial J}{\partial w} &= \sum_{n=1}^N (y_n - t_n) x_n
\end{aligned}
$$

Written as the dot product, this is:

$$
\begin{aligned}
    \frac{\partial J}{\partial w} &= X^T (Y - T)
\end{aligned}
$$

Gradient descent then does the usual incremental adjustment to w:

$$
\begin{aligned}
    w &= w - learningrate * \frac{\partial J}{\partial w} \\
    Y &= \sigma(w^T X)
\end{aligned}
$$

In code:

```python
learning_rate = 0.1
for i in range(100):
    # print cross entropy to show that it decreases across gradient descent
    if i % 10 == 0:
        print(cross_entropy_2(T, Y))

    w -= learning_rate * Xb.T.dot(Y - T)
    Y = sigmoid(Xb.dot(w))
```
