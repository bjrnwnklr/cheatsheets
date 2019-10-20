---
title: Kaggle competition cheatsheet
author: Bjoern Winkler
date: 28-August-2019
---

Useful tricks for Kaggle competitions.

Using the [Titanic challenge](https://www.kaggle.com/c/titanic) as an example.

# General approach

Good tutorial to feature engineering, filling in missing data and interpreting data is found in this [82% Kaggle tutorial](https://www.kaggle.com/gunesevitan/advanced-feature-engineering-tutorial-with-titanic)

0. Establish a baseline score
    1. Make a simple prediction (simple logic, logistic regression) as quickly as possible to establish a baseline score to compare against
1. Find missing values
1. Decide how to fill missing values
    1. Inspect the data - which samples are missing data (for very few, maybe the data offers a hint)
    2. Use mean/median/mode - depending on type of data
    3. Decide if grouping data together by existing features (e.g. Pclass / Title / Sex) will give better estimates for missing values
    4. Google / research to find out if that can help filling missing data (e.g. based on historic values)
1. Inspect distribution, correlation against targets
    1. Using `sns.distplot` for continuous features and `sns.countplot` for categorical (discrete) features
    2. If split points and spikes are very visible in the continuous features, they can be captured easily with a decision tree algorithm, but neural networks may not be able to spot them.
    3. If categorical features have very distinct classes with different survival rates, those classes can be used as new features with one-hot encoding. Some of those classes also may be combined with each other to make new features.
1. Feature engineering
    1. Bin continuous features, using `pd.qcut` (creates bins of equal size)
        1. Binning features generally has no beneficial effect for tree-based models, as these models can learn to split continuous data into bins if necessary.
    2. Label encode non-numerical features using `LabelEncoder`.
    3. One hot encoding categorical features using `OneHotEncoder` or `pd.get_dummies`
        1. Use `get_dummies` on a DataFrame containing both the training and the test data. This is important to ensure categorical values are represented in the same way in the training set and the test set. `OneHotEncoder` can deal with that automatically.
        2. Use `ColumnTransformer` to automate scaling and one hot encoding easily.
1. Drop any columns that are not required
1. Scale the columns using `StandardScaler`
    1. This is very important for a number of models, e.g. SVC relies on scaled data.
1. Run through a model, using cross validation
1. Compare feature importance


# Libraries to load

## Regression

```python
# Base libraries
import pandas as pd
import numpy as np
from datetime import datetime

# plotting 
import seaborn as sns
import matplotlib.pyplot as plt

# sklearn basics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix

# regression models
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

# sklearn grid search
from sklearn.model_selection import GridSearchCV

%matplotlib inline
```

## Classification

```python
# Base libraries
import pandas as pd
import numpy as np
from datetime import datetime

# plotting 
import seaborn as sns
import matplotlib.pyplot as plt

# sklearn basics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix

# regression models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

# sklearn grid search
from sklearn.model_selection import GridSearchCV

%matplotlib inline
```

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

## Finding missing data in a column

### Show missing values across all columns:

```python
df_train_raw.isna().sum()

PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64
```

### Show missing values in one column:

```python
df_all.loc[df_train_raw['Embarked'].isna()]
```

### Show missing values for all columns missing data only:

```python
dfs_to_check = [df_train_raw, df_test_raw]

for df_check in dfs_to_check:
    d = df_check.isna().sum()
    print(d[d > 0])
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

## Correlation

Creating correlation between Age and other features. This can give good hints on what to use to fill missing data values for Age (if they have a high correlation).

```python
df_all_corr = df_all.corr().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
df_all_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
df_all_corr[df_all_corr['Feature 1'] == 'Age']
```

## Plots

### Simple scatter plot to show relationship of one variable against target (regression)

```python
cat_cols = ['LotArea']

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for ax, var in zip(axes, cat_cols):
    data = pd.concat([df_targets, df_train_raw[var]], axis=1)
    sns.scatterplot(var, 'SalePrice', data=data, ax=ax)
```

### Boxplot to show distribution for categorical variables (regression)

```python
cat_cols = ['YrSold', 'MoSold', 'BedroomAbvGr']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, var in zip(axes, cat_cols):
    data = pd.concat([df_targets, df_train_raw[var]], axis=1)
    sns.boxplot(var, 'SalePrice', data=data, ax=ax)
```

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

### Continuous data - `sns.distplot`

```python
cont_features = ['Age', 'Fare']
surv = df_train['Survived'] == 1

fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(20, 20))
plt.subplots_adjust(right=1.5)

for i, feature in enumerate(cont_features):
    # Distribution of survival in feature
    sns.distplot(df_train[~surv][feature], label='Not Survived', hist=True, color='#e74c3c', ax=axs[0][i])
    sns.distplot(df_train[surv][feature], label='Survived', hist=True, color='#2ecc71', ax=axs[0][i])

    # Distribution of feature in dataset
    sns.distplot(df_train[feature], label='Training Set', hist=False, color='#e74c3c', ax=axs[1][i])
    sns.distplot(df_test[feature], label='Test Set', hist=False, color='#2ecc71', ax=axs[1][i])

    axs[0][i].set_xlabel('')
    axs[1][i].set_xlabel('')

    for j in range(2):
        axs[i][j].tick_params(axis='x', labelsize=20)
        axs[i][j].tick_params(axis='y', labelsize=20)

    axs[0][i].legend(loc='upper right', prop={'size': 20})
    axs[1][i].legend(loc='upper right', prop={'size': 20})
    axs[0][i].set_title('Distribution of Survival in {}'.format(feature), size=20, y=1.05)

axs[1][0].set_title('Distribution of {} Feature'.format('Age'), size=20, y=1.05)
axs[1][1].set_title('Distribution of {} Feature'.format('Fare'), size=20, y=1.05)

plt.show()
```

### Categorical data - `sns.countplot`

```python
cat_features = ['Embarked', 'Parch', 'Pclass', 'Sex', 'SibSp', 'Deck']

fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(20, 20))
plt.subplots_adjust(right=1.5, top=1.25)

for i, feature in enumerate(cat_features, 1):
    plt.subplot(2, 3, i)
    sns.countplot(x=feature, hue='Survived', data=df_train)

    plt.xlabel('{}'.format(feature), size=20, labelpad=15)
    plt.ylabel('Passenger Count', size=20, labelpad=15)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)

    plt.legend(['Not Survived', 'Survived'], loc='upper center', prop={'size': 18})
    plt.title('Count of Survival in {} Feature'.format(feature), size=20, y=1.05)

plt.show()
```

### Showing subplots (Countplot) for a number of columns

The trick here is to use the `plt.subplot(2, 3, i)` in the loop - this moves to the next subplot.

```python
cols = ['title_norm', 'Pclass', 'SibSp', 'Parch', 'missing_age']
fem = df_train[df_train['Sex'] == 'female']

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

for i, feature in enumerate(cols, 1):
    plt.subplot(2, 3, i)
    sns.countplot(x=feature, hue='Survived', data=fem)
```

### Barplot to show distribution across categories

```python
sns.barplot(x=df_all['Family_Size'].value_counts().index, y=df_all['Family_Size'].value_counts().values, ax=axs[0][0])
```

### Correlation heatmap

Showing correlation heatmaps between features for training and test sets:

```python
fig, axs = plt.subplots(nrows=2, figsize=(20, 20))

sns.heatmap(df_train.drop(['PassengerId'], axis=1).corr(), ax=axs[0], annot=True, square=True, cmap='coolwarm', annot_kws={'size': 14})
sns.heatmap(df_test.drop(['PassengerId'], axis=1).corr(), ax=axs[1], annot=True, square=True, cmap='coolwarm', annot_kws={'size': 14})

for i in range(2):
    axs[i].tick_params(axis='x', labelsize=14)
    axs[i].tick_params(axis='y', labelsize=14)

axs[0].set_title('Training Set Correlations', size=15)
axs[1].set_title('Test Set Correlations', size=15)

plt.show()
```

# Wrangling data

## Adding a column with mean of a group

This adds a new column `fare_norm` with the Fare divided by the number of the same tickets (e.g. if there were 3 tickets with the same number, divide the fare for each ticket holder by 3)

```python
df_all['fare_norm'] = df_all.groupby(['Ticket'])['Fare'].transform(lambda x: x / x.count())
```

## Using `groupby` to group data and iterate through the groups

`groupby` is not trivial to use - the output can be confusing. Here are some nifty tricks how to use `groupby` and the resulting grouped data.

Group data by `Last_Name` and `Fare` columns.

```python
df_group = df_all[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',
                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare'])
```

Looking at the number of records.

```python
df_group.count()
```

Printing out all key / group pairs (this can be long but shows the structure).

```python
for key, grp in df_group:
    print(key, grp)
```

Show all groups contained in the grouped object. There is more information in this [stackoverflow article](https://stackoverflow.com/questions/14734533/how-to-access-pandas-groupby-dataframe-by-key).

```python
df_group.groups
```

Show the grouped dataframe for a particular key (e.g. a Name / Fare combination).

```python
allison = df_group.get_group(('Allison', 151.55))
```

Iterate through a group.

```python
for ind, row in allison.iterrow():
    print(ind)
    print(row['Survived'], row['Name'])
```

Putting it all together. Going through groups of Name/Fare combinations, identifying families > 1, then finding values for each group and updating a record for each passenger that is part of the family. This can be slow for a large number of records!

```python
for _, grp_df in df_all[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',
                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):

    if (len(grp_df) != 1):
        # A Family group is found.
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                df_all.loc[df_all['PassengerId'] == passID, 'Family_Survival'] = 1
            elif (smin==0.0):
                df_all.loc[df_all['PassengerId'] == passID, 'Family_Survival'] = 0

print("Number of passengers with family survival information:",
      df_all.loc[df_all['Family_Survival'] != 0.5].shape[0])
```

## Filling missing data

Using Pclass and Sex to fill missing Age values with the `median` value for Sex/Pclass combinations (could also use Title/Pclass combination):

```python
# this might be inefficient: should put the ['Age'] before the .median()
age_by_pclass_sex = df_all.groupby(['Sex', 'Pclass']).median()['Age']

for pclass in range(1, 4):
    for sex in ['female', 'male']:
        print('Median age of Pclass {} {}s: {}'.format(pclass, sex, age_by_pclass_sex[sex][pclass]))
print('Median age of all passengers: {}'.format(df_all['Age'].median()))

# Filling the missing values in Age with the medians of Sex and Pclass groups
df_all['Age'] = df_all.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
```

This is an alternative using `groupby` and `transform` - it avoids using `apply` and `lambda`, which is inefficient:

```python
# apply mean to all except where NaN values are (i.e. no groups)
df['value'] = df['value'].fillna(df.groupby('category')['value'].transform('mean'))
# apply overall mean to all remaining NaN rows (where we didn't have a group value)
df['value'] = df['value'].fillna(df['value'].mean())
```

## Categorizing data into bins

### Categorizing age groups into bins.

```python
bins = [0, 5, 12, 19, 35, 60, 100]
labels = ['young', 'child', 'teenager', 'young_adult', 'adult', 'senior']
df_train_raw['Age_group'] = pd.cut(df_train_raw['Age'], bins=bins, labels=labels)
```

### Using `pd.qcut` to create equal sized bins

```python
df_all['Fare'] = pd.qcut(df_all['Fare'], 13)

fig, axs = plt.subplots(figsize=(22, 9))
sns.countplot(x='Fare', hue='Survived', data=df_all)
```

## Adding a new column with summarized titles

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

## Converting two columns with year / month into a timeseries and number of days since start

-   Use `pd.to_datetime` to convert two columns with year and month into a timeseries. `pd.to_datetime` also expects a day, so just use 1 for the days. You can pass a dictionary into the `to_datetime` function.
-   Take the difference between the current date and the minimum date of the column to get days since the start (= minimum of the series)
-   Use `dataframe.dt.days` to convert the number of days from a datetime object into integer

Finally, sort the dataframe by the date and split into features and targets again.

```python
# use pd.to_datetime to convert columns with year and month into timeseries
df_train_base['DateSold'] = pd.to_datetime({'year': df_train_base['YrSold'], 'month': df_train_base['MoSold'], 'day': 1})
df_train_base['days'] = (df_train_base['DateSold'] - df_train_base['DateSold'].min()).dt.days

# sort by days (need to add in targets to sort them as well)
df_train_temp = pd.concat([df_train_base, df_targets], axis=1).sort_values(by=['DateSold'])
df_train_base_sort = df_train_temp.drop(['SalePrice'], axis=1)
df_targets_sort = df_train_temp[['SalePrice']]
```

## Converting categorical into binary

Converting male/female into 0 / 1

```python
df_temp['Sex'] = df_temp['Sex'].map({'male': 0, 'female': 1})
```

## Using `LabelEncoder` to convert non-numerical features into numerical

Non-numerical features are converted to numerical type with `LabelEncoder`. LabelEncoder basically labels the classes from 0 to n. This process is necessary for Machine Learning algorithms to learn from those features.

```python
non_numeric_features = ['Embarked', 'Sex', 'Deck', 'Title', 'Family_Size_Grouped', 'Age', 'Fare']

for df in dfs:
    for feature in non_numeric_features:
        df[feature] = LabelEncoder().fit_transform(df[feature])
```

## One hot encoding

One hot encoding categorical data, e.g. Pclass

```python
pclass_df = pd.get_dummies(df_temp['Pclass'], prefix='Pclass')
df_temp = pd.concat([df_temp, pclass_df], axis=1)
```

## Using ColumnTransformer to OneHotEncode and StandardScale data in one go

`ColumnTransfomer` can be used to transform multiple columns in one go - should be the last step before fitting the model.

-   `OneHotEncoder` can be used for _categorical_ data
-   `StandardScaler` or `KBinsDiscretizer` can be used for continuous data

Some of the transformers (e.g. `OneHotEncoder`) offer the `get_feature_names_` method to get the new column names. By specifying the column names, the column names will be used as suffix for the new column names. The `named_transfomers_` attribute provides access to the properties of the various transformers used.

-   The output of `ColumnTransformer` is a numpy array, so doesn't have column names. These need to be added back in:
-   We are using the `get_feature_names_` method of `OneHotEncoder` and adding the column names for the columns scaled with `StandardScaler` to a list of columns
-   This is then used to create a Pandas Dataframe with the `ColumnTransformer` output

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, KBinsDiscretizer

cat_columns = ['Pclass',
                'Embarked',
                'Sex',
                'is_alone',
                'family_size',
                'missing_age']

cont_columns = ['Age',
                'Fare']

column_trans = ColumnTransformer(
    [('categorical', OneHotEncoder(sparse=False), cat_columns),
     ('continuous', StandardScaler(), cont_columns)
    ])

column_trans.fit(df_all)

# create column names (OneHotEncoder provides new column names, StandardScaler doesn't)
col_names = list(column_trans.named_transformers_.categorical.get_feature_names(cat_columns)) + [c + '_scaled' for c in cont_columns]

df_all_processed = pd.DataFrame(data=column_trans.transform(df_all), columns=col_names)
```

### Using `passthrough` with `ColumnTransformer` to include columns in output without any processing

```python
cat_columns = ['Pclass',
               'Age_group']

other_columns = ['Sex',
                 'family_size',
                 'is_alone',
                 'Fare_bin_code'
                ]

column_trans = ColumnTransformer(
    [('categorical', OneHotEncoder(sparse=False), cat_columns),
     ('pass', 'passthrough', other_columns)
    ])
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

# Scaling data

## Using `StandardScaler` to scale data

```python
X_train = StandardScaler().fit_transform(df_train.drop(columns=drop_cols))
```

Scale data at the end before using the model! Scale train and test sets separately (i.e. fit the model using the training data, then transform the test data):

```python
std_scaler = StandardScaler()
X_train = std_scaler.fit_transform(X_train)
X_test = std_scaler.transform(X_test)
```

# Measuring accuracy

Using `accuracy_score` to measure accuracy between prediction and targets:

```python
from sklearn.metrics import accuracy_score
print(accuracy_score(y_true, y_predict))
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

# Using `GridSearchCV` to find the best hyperparameters for the model

`GridSearchCV` can be used to find hyperparameters. First, prepare the parameters to run through:

```python
clf = DecisionTreeClassifier()

hyperparams = {
    'max_depth': list(range(1, 30)),
    'min_samples_leaf': list(range(1, 10, 10)),
    'max_features': list(range(1, 10))
}
```

Then, use the grid search to run through the parameters:

```python
from sklearn.model_selection import GridSearchCV

gd = GridSearchCV(clf, param_grid=hyperparams, verbose=True, cv=10, scoring='accuracy', return_train_score=False, n_jobs=-1)
gd.fit(X_train, y_train)

print('Best score: ', gd.best_score_)
print('Best estimator: ', gd.best_estimator_)
```

Then, use the found best estimator to make a prediction and submit data:

```python
model_version = "5g_svc"

# this is actually not required - you can directly use the GridSearch instance and use the predict and score methods
# GridSearchCV automatically fits the best performing model to the gd instance, so using best_estimator is not necessary
#gd.best_estimator_.fit(X_train, y_train)
#y_pred = gd.best_estimator_.predict(X_test)

# use the model automatically fit to the gridsearch instance, using the best parameters
y_pred = gd.predict(X_test)

# test data starts at `split`
results = df_all[split:].copy()
results['Survived'] = y_pred

timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')

# write the data to a file
f = 'titanic_v{}_{}.csv'.format(model_version, timestamp)
results[['PassengerId', 'Survived']].to_csv(f, header=True, index=False)
```

## Visualizing the results of grid search

Using a heatmap, visualize the results from gridsearch for 2 parameters (visualizing more than two parameters is difficult, but you can just output the results into a pandas dataframe and search for max values across the `cv_results_.mean_test_score` to find the best combination of parameters).

Using this on the Titanic model, tuning the gamma and C parameters produced minimally better results on the training dataset, but the exact same results on the test set. The tuning was minimal against the broad parameters found with gridsearch, so it seems that finding the right order of magnitude for C and gamma is much more important than finetuning them.

E.g. differences for C between 1, 10, 100, 200 had massively better values at 100, but then the difference between 100 and 106 (best value found when further finetuning) was minimal.

```python
# save results of the gridsearch in a pandas dataframe
gs_results = pd.DataFrame(gd.cv_results_)

# we searched over 8 * 6 parameters, so need to reshape the results into an 8 * 6 array
scores = np.array(gs_results.mean_test_score).reshape(8, 6)

fig = plt.figure(figsize=(8, 8))
sns.heatmap(
    scores,
    cmap='coolwarm',
    square=True,
    annot=True,
    annot_kws={'size': 12},
    xticklabels=hyperparams['gamma'],
    yticklabels=hyperparams['C'],
    cbar=False
)

plt.xlabel=('gamma')
plt.ylabel=('C')
plt.title('SVC')
```

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
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')

# write the data to a file
f = 'titanic_v{}_{}.csv'.format(model_version, timestamp)
results[['PassengerId', 'Survived']].to_csv(f, header=True, index=False)
```
