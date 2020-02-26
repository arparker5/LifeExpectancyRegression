import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


data = pd.read_csv('../Life Expectancy Data.csv')
print(data.shape)
# print(data.describe())
data = data.fillna(method='bfill')

d = {'Developing': 0, 'Developed': 1}
data['Status'] = data['Status'].map(d).fillna(data['Status'])

# print(data.isnull().any())
# print(data.describe())


X = data[['Status', 'Adult Mortality', 'infant deaths',
          'Alcohol', 'percentage expenditure', 'Hepatitis B', 'Measles ',
          ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ',
          ' HIV/AIDS', 'GDP', 'Population', ' thinness  1-19 years', ' thinness 5-9 years',
          'Income composition of resources', 'Schooling']].values

X_columns = ['Status', 'Adult Mortality', 'infant deaths',
          'Alcohol', 'percentage expenditure', 'Hepatitis B', 'Measles ',
          ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ',
          ' HIV/AIDS', 'GDP', 'Population', ' thinness  1-19 years', ' thinness 5-9 years',
          'Income composition of resources', 'Schooling']

Y = data['Life expectancy '].values

plt.figure(figsize=(15, 10))
plt.tight_layout()
sns.distplot(data['Life expectancy '])
# plt.show()

# 80/20 split of training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

reg = LinearRegression()
reg.fit(X_train, Y_train)

coeff_df = pd.DataFrame(reg.coef_, X_columns, columns=['Coefficient'])
print(coeff_df)
