import pandas as pd
import matplotlib.pyplot as plt

# Load the data from a CSV file
data = pd.read_csv('data.csv')

# data exploration
print(data.head())
print(data.describe())

# data cleaning and preprocessing
data = data.dropna()
data['date'] = pd.to_datetime(data['date'])

# data analysis
total_sales = data['sales'].sum()
average_sales = data['sales'].mean()
max_sales = data['sales'].max()

# data visualization
plt.plot(data['date'], data['sales'])
plt.title('Sales over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()

# export data to a new CSV file
data.to_csv('new_data_analysis_Patryk.csv', index=False)