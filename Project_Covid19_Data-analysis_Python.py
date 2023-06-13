import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('covid_data.csv')

# Display the first few rows of the dataset
print(data.head())

# Check the data types and summary statistics
print(data.info())
print(data.describe())

# Convert the date column to datetime format
data['date'] = pd.to_datetime(data['date'])

# Remove any missing values
data = data.dropna()

# Calculate total cases and deaths
total_cases = data['cases'].sum()
total_deaths = data['deaths'].sum()

# Calculate average daily cases and deaths
avg_daily_cases = data['cases'].mean()
avg_daily_deaths = data['deaths'].mean()

# Find the date with the highest number of cases
max_cases_date = data.loc[data['cases'].idxmax(), 'date']

# Calculate the case fatality rate
case_fatality_rate = (total_deaths / total_cases) * 100

# Group the data by date and calculate the total cases and deaths for each day
daily_data = data.groupby('date').sum()

# Calculate the 7-day moving average of cases
daily_data['cases_7day_avg'] = daily_data['cases'].rolling(window=7).mean()

# Plot the total cases and deaths over time
plt.plot(data['date'], data['cases'], label='Total Cases')
plt.plot(data['date'], data['deaths'], label='Total Deaths')
plt.xlabel('Date')
plt.ylabel('Count')
plt.title('COVID-19 Cases and Deaths Over Time')
plt.legend()
plt.show()

# Plot the 7-day moving average of cases
plt.plot(daily_data.index, daily_data['cases_7day_avg'])
plt.xlabel('Date')
plt.ylabel('Cases (7-day average)')
plt.title('COVID-19 Cases (7-day Moving Average)')
plt.show()