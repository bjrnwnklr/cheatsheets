---
title: Kaggle competition cheatsheet
author: Bjoern Winkler
date: 28-August-2019
---

Useful tricks for Kaggle competitions.

Using the [Titanic challenge](https://www.kaggle.com/c/titanic) as an example.


# General approach

Good tutorial to feature engineering, filling in missing data and interpreting data is found in this [82% Kaggle tutorial](https://www.kaggle.com/gunesevitan/advanced-feature-engineering-tutorial-with-titanic)

1. Find missing values
2. Decide how to fill missing values
   1. Inspect the data - which samples are missing data (for very few, maybe the data offers a hint)
   2. Use mean/median/mode - depending on type of data
   3. Decide if grouping data together by existing features (e.g. Pclass / Title / Sex) will give better estimates for missing values
   4. Google / research to find out if that can help filling missing data (e.g. based on historic values)
3. Inspect distribution, correlation against targets    
   1. Using `sns.distplot` for continuous features and `sns.countplot` for categorical (discrete) features
   2. If split points and spikes are very visible in the continuous features, they can be captured easily with a decision tree algorithm, but neural networks may not be able to spot them.
   3. If categorical features have very distinct classes with different survival rates, those classes can be used as new features with one-hot encoding. Some of those classes also may be combined with each other to make new features.
4. Feature engineering
   1. Bin continuous features, using `pd.qcut` (creates bins of equal size)
   2. Label encode non-numerical features using `LabelEncoder`. 
   3. One hot encoding categorical features using `OneHotEncoder` of `pd.get_dummies`
5. Drop any columns that are not required
6. Scale the columns using `StandardScaler`
7. Run through a model, using cross validation
8. Compare feature importance

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

## Correlation

Creating correlation between Age and other features. This can give good hints on what to use to fill missing data values for Age (if they have a high correlation).

```python
df_all_corr = df_all.corr().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
df_all_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
df_all_corr[df_all_corr['Feature 1'] == 'Age']
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

## Filling missing data

Using Pclass and Sex to fill missing Age values with the `median` value for Sex/Pclass combinations (could also use Title/Pclass combination):

```python
age_by_pclass_sex = df_all.groupby(['Sex', 'Pclass']).median()['Age']

for pclass in range(1, 4):
    for sex in ['female', 'male']:
        print('Median age of Pclass {} {}s: {}'.format(pclass, sex, age_by_pclass_sex[sex][pclass]))
print('Median age of all passengers: {}'.format(df_all['Age'].median()))

# Filling the missing values in Age with the medians of Sex and Pclass groups
df_all['Age'] = df_all.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
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

### Using `LabelEncoder` to convert non-numerical features into numerical

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
