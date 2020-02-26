import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


def myRegression(X_columns, Y):

    X = data[X_columns].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    reg = LinearRegression()
    reg.fit(X_train, Y_train)

    Y_pred = reg.predict(X_test)
    df = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})

    # print("Regression model output(first 25 variables): ")
    # print()
    # print(df.head(25))

    print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))


data = pd.read_csv('../Life Expectancy Data.csv')
data = data.fillna(method='bfill')

d = {'Developing': 0, 'Developed': 1}
data['Status'] = data['Status'].map(d).fillna(data['Status'])

# print(data.isnull().any())
# print(data.describe())

X_columns = ['Status', 'Adult Mortality', 'infant deaths',
          'Alcohol', 'percentage expenditure', 'Hepatitis B', 'Measles ',
          ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ',
          ' HIV/AIDS', 'GDP', 'Population', ' thinness  1-19 years', ' thinness 5-9 years',
          'Income composition of resources', 'Schooling']


Y = data['Life expectancy '].values

print()
print("------------ Model with all variables ------------")
myRegression(X_columns, Y)

# ----------------- Dropping 'Population' ----------------- #

X_columns = ['Status', 'Adult Mortality', 'infant deaths',
          'Alcohol', 'percentage expenditure', 'Hepatitis B', 'Measles ',
          ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ',
          ' HIV/AIDS', 'GDP', ' thinness  1-19 years', ' thinness 5-9 years',
          'Income composition of resources', 'Schooling']

print()
print("------------ Model with 'population' dropped ------------")
myRegression(X_columns, Y)

# ----------------- Dropping Measles, GDP ----------------- #

X_columns = ['Status', 'Adult Mortality', 'infant deaths',
          'Alcohol', 'percentage expenditure', 'Hepatitis B',
          ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ',
          ' HIV/AIDS', ' thinness  1-19 years', ' thinness 5-9 years',
          'Income composition of resources', 'Schooling']

print()
print("------------ Dropping Measles, GDP ------------")
myRegression(X_columns, Y)

# ----------------- Dropping 'percentage expenditure' ----------------- #

X_columns = ['Status', 'Adult Mortality', 'infant deaths',
          'Alcohol', 'Hepatitis B',
          ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ',
          ' HIV/AIDS', ' thinness  1-19 years', ' thinness 5-9 years',
          'Income composition of resources', 'Schooling']

print()
print("------------ Dropping 'percentage expenditure' ------------")
myRegression(X_columns, Y)

# plt.figure(figsize=(15, 10))
# plt.tight_layout()
# sns.distplot(data['Life expectancy '])
# plt.show()


# coeff_df = pd.DataFrame(reg.coef_, X_columns, columns=['Coefficient'])
# print(coeff_df)


# df.head(25).plot(kind='bar', figsize=(10, 8))
# plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
# plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
# plt.show()

