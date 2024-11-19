import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Import the dataset
url = "https://raw.githubusercontent.com/SR1608/Datasets/main/covid-data.csv"
df = pd.read_csv(url)

# 2. High Level Data Understanding
# a. Number of rows and columns
rows, cols = df.shape
print(f"Number of rows: {rows}, Number of columns: {cols}")

# b. Data types of columns
print(df.dtypes)

# c. Info and describe of data
print(df.info())
print(df.describe())

# 3. Low Level Data Understanding
# a. Count of unique values in location column
unique_locations = df['location'].nunique()
print(f"Unique locations: {unique_locations}")

# b. Continent with maximum frequency
continent_max_freq = df['continent'].value_counts().idxmax()
print(f"Continent with maximum frequency: {continent_max_freq}")

# c. Maximum and mean value in 'total_cases'
max_total_cases = df['total_cases'].max()
mean_total_cases = df['total_cases'].mean()
print(f"Maximum total cases: {max_total_cases}, Mean total cases: {mean_total_cases}")

# d. 25%, 50% & 75% quartile value in 'total_deaths'
quartiles_total_deaths = df['total_deaths'].quantile([0.25, 0.5, 0.75])
print(f"Quartiles for total deaths: {quartiles_total_deaths}")

# e. Continent with maximum 'human_development_index'
continent_max_hdi = df.groupby('continent')['human_development_index'].idxmax()
print(f"Continent with maximum human development index: {continent_max_hdi}")

# f. Continent with minimum 'gdp_per_capita'
continent_min_gdp = df.groupby('continent')['gdp_per_capita'].idxmin()
print(f"Continent with minimum GDP per capita: {continent_min_gdp}")

# 4. Filter the dataframe
df = df[['continent', 'location', 'date', 'total_cases', 'total_deaths', 'gdp_per_capita', 'human_development_index']]

# 5. Data Cleaning
# a. Remove duplicates
df = df.drop_duplicates()

# b. Find missing values
missing_values = df.isnull().sum()
print(f"Missing values:\n{missing_values}")

# c. Remove observations where continent column value is missing
df = df.dropna(subset=['continent'])

# d. Fill missing values with 0
df = df.fillna(0)

# 6. Date time format
# a. Convert date column to datetime format
df['date'] = pd.to_datetime(df['date'])

# b. Create new column 'month'
df['month'] = df['date'].dt.month

# 7. Data Aggregation
# a. Find max value in all columns using groupby on 'continent'
df_groupby = df.groupby('continent').max().reset_index()

# 8. Feature Engineering
# a. Create new feature 'total_deaths_to_total_cases'
df_groupby['total_deaths_to_total_cases'] = df_groupby['total_deaths'] / df_groupby['total_cases']

# 9. Data Visualization
# a. Univariate analysis on 'gdp_per_capita'
sns.histplot(df_groupby['gdp_per_capita'], kde=True)
plt.title('GDP per Capita Distribution')
plt.show()

# b. Scatter plot of 'total_cases' & 'gdp_per_capita'
sns.scatterplot(x='total_cases', y='gdp_per_capita', data=df_groupby)
plt.title('Total Cases vs GDP per Capita')
plt.show()

# c. Pairplot on df_groupby dataset
sns.pairplot(df_groupby)
plt.show()

# d. Bar plot of 'continent' with 'total_cases'
sns.catplot(x='continent', y='total_cases', kind='bar', data=df_groupby)
plt.title('Total Cases by Continent')
plt.show()

# 10. Save the df_groupby dataframe
df_groupby.to_csv('df_groupby.csv', index=False)
