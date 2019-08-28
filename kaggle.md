---
title: Kaggle competition cheatsheet
author: Bjoern Winkler
date: 28-August-2019
---

Useful tricks for Kaggle competitions.

Using the [Titanic challenge](https://www.kaggle.com/c/titanic) as an example.

# Describing the dataset

```python

df.describe()

df.describe(include=['O']) # shows describe for non-numerical (object) columns

df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB

```

# Initial analysis

## Pivots / groupby

Show % of a categorical vs another (e.g. Survived vs Sex)

```python
df_train_raw[['Sex', 'Survived']].groupby(['Sex']).mean()
```

Aggregating over a grouped object. Using different functions (e.g. `count` and `mean`) for specific columns using a dict.

This produces a pivot with number of passengers and average age for each title.

```python
aggregations = {'PassengerId': 'count', 'Age': 'mean'}
df_all.groupby('Title').agg(aggregations)
```

Using a crosstab to show # of passengers by title, split by gender:

```python
pd.crosstab(train_df['Title'], train_df['Sex'])
```

## Plots

### FacetGrid

Show two plots side by side, for Survived == 0 and == 1, with an Age histogram per survival group.

```python
g = sns.FacetGrid(df_train_raw, col='Survived')
g.map(plt.hist, 'Age', bins=20)
```

### ViolinPlot

Showing Sex on x axis vs Age on y axis, with Survived as the two sides of the violin.

```python
sns.violinplot(x='Sex', y='Age',
               hue='Survived', data=df_train_raw,
               split=True,
               palette={0: "r", 1: "g"}
              );
```

### Scatterplot with size of balls = ticket price

Show a scatter plot with 3 dimensions (age, survived and ticket price)

```python
plt.figure(figsize=(25, 7))
ax = plt.subplot()

ax.scatter(data[data['Survived'] == 1]['Age'], data[data['Survived'] == 1]['Fare'],
           c='green', s=data[data['Survived'] == 1]['Fare'])
ax.scatter(data[data['Survived'] == 0]['Age'], data[data['Survived'] == 0]['Fare'],
           c='red', s=data[data['Survived'] == 0]['Fare']);
```

# Wrangling data

## Categorizing data into bins

### Categorizing age groups into bins.

```python
bins = [0, 5, 12, 19, 35, 60, 100]
labels = ['young', 'child', 'teenager', 'young_adult', 'adult', 'senior']
df_train_raw['Age_group'] = pd.cut(df_train_raw['Age'], bins=bins, labels=labels)
```

### Adding a new column with summarized titles

```python
title_transform_rev = {
    'Mr': ['Mr', 'Capt', 'Col', 'Don', 'Dr', 'Jonkheer', 'Major', 'Rev', 'Sir', 'th'],
    'Master': ['Master'],
    'Miss': ['Miss', 'Ms', 'Mlle', 'Mme'],
    'Mrs': ['Mrs', 'Dona', 'Lady']
}

# reverse the title dictionary into old:new pairs
title_transform = {
    t: x for x in title_transform_rev for t in title_transform_rev[x]
}

# get the titles and concat them into a new column
title = get_title(df_temp)
df_temp = pd.concat([df_temp, title], axis=1)
```

### Converting categorical into binary

Converting male/female into 0 / 1

```python
df_temp['Sex'] = df_temp['Sex'].map({'male': 0, 'female': 1})
```

## One hot encoding

One hot encoding categorical data, e.g. Pclass

```python
pclass_df = pd.get_dummies(df_temp['Pclass'], prefix='Pclass')
df_temp = pd.concat([df_temp, pclass_df], axis=1)
```

## Extracting data

### Extracting the title from a column with names

```python
def get_title(df):
    title_re = r',\s([a-zA-Z]+).\s'
    title = df['Name'].str.extract(title_re)
    title.columns = ['Title']
    return title

title = get_title(df_all)
df_all = pd.concat([df_all, title], axis=1)
```

# Cross validating across models

Testing several models by using cross validation.

```python
# set up 5 models
clfs = [LogisticRegression(solver='liblinear'),
        KNeighborsClassifier(n_neighbors=5),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=100),
        GradientBoostingClassifier(max_depth=3, n_estimators=100)
       ]
names = ['LogReg', 'KNN', 'Tree', 'Forest', 'GradBoost']

# testing various combinations of columns
cols_sets = [
    ['Sex', 'SibSp', 'Parch', 'Age_group_young', 'Age_group_child', 'Age_group_teenager',
       'Age_group_young_adult', 'Age_group_adult', 'Age_group_senior',
       'Pclass_1', 'Pclass_2', 'Pclass_3'],
    ['Sex', 'Age_group_young', 'Age_group_child', 'Age_group_teenager',
       'Age_group_young_adult', 'Age_group_adult', 'Age_group_senior',
       'Pclass_1', 'Pclass_2', 'Pclass_3']
]

# number of kfolds for cross validation i.e. data is split into 5 segments
cv = 5

for c in cols_sets:
    df_train_sub = df_train[c]
    df_test_sub = df_test[c]
    df_train_survived = df_train['Survived']

    print('Cols used: ', c)

    for name, clf in zip(names, clfs):
        score = cross_val_score(clf, df_train_sub, df_train_survived, cv=cv)
        print('{}: {} runs. Scores: Max: {:.3f}, Min: {:.3f}, Mean: {:.3f} (+/- {:.2f})'.format(
            name, cv, max(score), min(score), np.mean(score), np.std(score) * 2))

    print('\n')

```

## Overfitting

How to find out if our model is overfitting (using cross validation and train vs test set)

# Predicting and submission

Predicting data using a set of columns and a model

```python
best_cols = ['Sex', 'SibSp', 'Parch', 'Age_group_young', 'Age_group_child', 'Age_group_teenager',
       'Age_group_young_adult', 'Age_group_adult', 'Age_group_senior',
       'Pclass_1', 'Pclass_2', 'Pclass_3']
clf = RandomForestClassifier(max_depth=5, n_estimators=100)

# using only those columns for train and test set, and extracting the targets from the training set
df_train_sub = df_train[best_cols]
df_test_sub = df_test[best_cols]
df_train_survived = df_train['Survived']

# fit the model
clf.fit(df_train_sub, df_train_survived)

# make the prediction
prediction = clf.predict(df_test_sub)

# create a results column in a copy of the test dataframe
results = df_test.copy()
results['Survived'] = prediction
```

Create the submission:

```python
model_version = "2a"
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%m')

# write the data to a file
f = 'titanic_v{}_{}.csv'.format(model_version, timestamp)
results[['PassengerId', 'Survived']].to_csv(f, header=True, index=False)
```
