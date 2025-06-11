import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define file location
data_location = 'covid_data.csv'

# Read data into DataFrame
covid_df = pd.read_csv(data_location)

# Replace missing continents with 'Unknown'
covid_df['continent'].fillna('Unknown', inplace=True)

# Fill other missing numeric values with column-wise mean
numeric_cols = covid_df.select_dtypes(include=['float64', 'int64']).columns
covid_df[numeric_cols] = covid_df[numeric_cols].fillna(covid_df[numeric_cols].mean())

# Output descriptive statistics after preprocessing
print("Dataset Summary After Cleaning:\n")
print(covid_df.describe())

# Display sample records for quick check
print("\nSample Records:\n")
print(covid_df.sample(5))





print("\nSummary Statistics:")
print(covid_df.describe())
print("Mean:\n", covid_df.mean(numeric_only=True))
print("Median:\n", covid_df.median(numeric_only=True))
print("Count:\n", covid_df.count())




import matplotlib.pyplot as plt
import seaborn as sns

# Display section header in bold
print("\033[1m{:^80}\033[0m".format("DATA VISUALIZATIONS"))

# Bar Chart: Cases per Continent
plt.figure(figsize=(8, 5))
sns.countplot(data=covid_df, x='continent', palette='Set2')
plt.title("Cases Distribution by Continent")
plt.ylabel("Total Cases")
plt.xlabel("Continent")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# Line Plot: Male Smokers vs Life Expectancy
plt.figure(figsize=(8, 5))
sns.lineplot(x=covid_df['male_smokers'], y=covid_df['life_expectancy'], color='purple', marker='o')
plt.title("Life Expectancy vs Male Smokers")
plt.xlabel("Male Smokers (%)")
plt.ylabel("Life Expectancy (Years)")
plt.grid(visible=True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# Pie Chart: Continent Distribution
plt.figure(figsize=(6, 6))
covid_df['continent'].value_counts().plot.pie(
    autopct='%0.1f%%', 
    startangle=140, 
    shadow=False, 
    colors=sns.color_palette("Pastel1")
)
plt.title("Continent-wise Case Distribution")
plt.ylabel('')
plt.tight_layout()
plt.show()
