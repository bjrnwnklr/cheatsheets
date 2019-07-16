# Pandas cheatsheet

# Basics

## Slicing

Slicing in Pandas works by using either the `loc` (based on names) or `iloc` (by index) methods:

```python
df.iloc[:, 0] # selects all rows, first column from pandas dataframe

ndarray[:, 0] # selects all rows, first column from numpy ndarray
```

## Counting unique values

This returns a pd.Series object, sorted by the most frequent items:

```python
df['City'].value_counts()
```

## Creating a Pivot table

- Using `loc` and `.isin`, find all cities that are in a list
- Create a pivot table, specifying index (rows) vs columns and the values to be used
- Since there are only individual values in this table (i.e. only one value per row), the values are not aggregated. Otherwise. the standard function applied to the values is `mean`.

```python
(df_g_cities.loc[df['City'].isin(g_10_cities_2016)]
    .pivot(
        index='Year',
        columns='City',
        values='Value'
    )
)
```

# Selecting

## Selecting all entries based on first letter (or some other function)

Using `lambda` function, filter rows based on starting letter. This could also be done with any other function, e.g. length of the city name.

```python
# Select all rows where 'City' starts with the letter 'F'
df[df['City'].map(lambda x: x.startswith('F'))]

# Show all cities that start with 'F'
df[df['City'].map(lambda x: x.startswith('F'))]['City'].value_counts()
```

## Creating a filter statement based on multiple columns

Create a filter statement based on multiple column conditions. Then use the filter statement to create a new data frame. Use only a number of columns.

Use `loc` to apply the filter.

```python
# create a filter
df_city_filter = (
    (df['Area'] == 'Total' )
    & (df['Sex'] == 'Both Sexes')
    & (df['Record Type'] == 'Estimate - de jure')
    & (df['City type'] == 'City proper')
)

# select columns
city_columns = [
    'Year',
    'City',
    'Value'
]

# Apply the filter to create a new data frame
df_g_cities = df_germany.loc[df_city_filter, city_columns]
```

## Create a list of the 10 biggest cities

- Select only records from 2016 using `loc`
- Sort by value column
- Select first 10 records
- Show only the 'City' column
- Convert to a list

```python
# 10 biggest cities in 2016
g_10_cities_2016 = (df_g_cities.loc[
    df['Year'] == '2016']
    .sort_values(
        by=['Value'], 
        ascending=False
    ))[:10]['City'].to_list()
```

# Plotting

## Simple plot of value_counts

Shows a simple bar chart of value counts, sorted by the index (not by frequency):

```python
df['City'].value_counts().sort_index().plot.bar()
```

## Plot a Pivot table

- The plot function creates a line chart
- If multiple lines are to be plotted, they have to be in a column each - so create the pivot with the columns that should be plotted as individual lines
- otherwise, you can use `.T` to transpose the pivot if the lines to be plotted are rows
- The `ax.legend` statement can be used to specify the location of the legend. Using `loc='upper left` and `bbox_to_anchor=(1, 1)` puts the upper left corner of the legend outside of the chart

```python
ax = (df_g_cities.loc[df['City'].isin(g_10_cities_2016)]
    .pivot(
        index='Year',
        columns='City',
        values='Value'
    )
).plot()

ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
```