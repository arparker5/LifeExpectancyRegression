import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


data = pd.read_csv('../Life Expectancy Data.csv')
print(data.shape)
print(data.describe())
data = data.fillna(method='bfill')

d = {'Developing': 0, 'Developed': 1}
data['Status'] = data['Status'].map(d).fillna(data['Status'])

# print(data.isnull().any())
# print(data.describe())


# X = data[[]]
