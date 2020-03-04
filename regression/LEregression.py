import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn import metrics


def compareCountries(X_columns, Y):
    X = data[X_columns].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    X_tail = []
    X_tailtest = []
    country = []

    for tr in X_train:
        X_tail.append(tr[1:])

    for tl in X_test:
        country.append(tl[0])
        X_tailtest.append(tl[1:])

    reg = LinearRegression()
    reg.fit(X_tail, Y_train)

    Y_pred = reg.predict(X_tailtest)

    clist = []
    for i in range(len(Y_test)):
        item = [country[i], Y_test[i], Y_pred[i]]
        clist.append(item)

    clist = sorted(clist, key=lambda x: x[0])


    formatclist = []
    curcountry = clist[0][0]
    templist = [curcountry, [], []]
    for c in clist:                                             # Making sub-lists for each country
        if c[0] == curcountry:
            templist[1].append(c[1])
            templist[2].append(c[2])
        else:
            formatclist.append(templist)
            curcountry = c[0]
            templist = [curcountry, [c[1]], [c[2]]]
    formatclist.append(templist)


    country = []
    for f in formatclist:
        country.append(f[0])

    print()
    print()
    mae = []
    for f in formatclist:
        mae.append(metrics.mean_absolute_error(f[1], f[2]))

    df = pd.DataFrame({'Country': country, 'Mean Average Error': mae})
    df.set_index('Country', inplace=True)
    df.sort_values(by=['Mean Average Error'], ascending=False, inplace=True)
    df = df.iloc[np.r_[0:25, -25:0]]
    df.plot(kind='barh', figsize=(50, 100))
    plt.show()


def myRegression(X_columns, Y):

    X = data[X_columns].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    reg = LinearRegression()
    reg.fit(X_train, Y_train)

    Y_pred = reg.predict(X_test)
    df = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})

    print("Regression model output(first 25 variables): ")
    print()
    print(df.head(25))

    mae = metrics.mean_absolute_error(Y_test, Y_pred)
    mse = metrics.mean_squared_error(Y_test, Y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(Y_test, Y_pred))

    coeff_df = pd.DataFrame(reg.coef_, X_columns, columns=['Coefficient'])
    print(coeff_df)

    return [mae, mse, rmse]


def testalpha(X_columns, Y, lasso):    # Tests multiple alpha values on Ridge and Lasso regressions. 0: Ridge. 1: Lasso
    X = data[X_columns].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


    maelist = []
    mselist = []
    rmselist = []
    #avalues = [0.1, 0.5, 1.0, 5.0, 10.0, 13.0]
    avalues = [4.6, 5.0, 10.0, 13.0, 20.0, 50.0, 100.0, 300.0, 1000.0, 2000.0]

    errorstats = []
    for a in avalues:
        if lasso:
            reg = linear_model.Lasso(alpha=a)
        else:
            reg = linear_model.Ridge(alpha=a)
        reg.fit(X_train, Y_train)

        Y_pred = reg.predict(X_test)
        df = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})

        mae = metrics.mean_absolute_error(Y_test, Y_pred)
        mse = metrics.mean_squared_error(Y_test, Y_pred)
        rmse = np.sqrt(metrics.mean_squared_error(Y_test, Y_pred))

        errorstats.append([mae, mse, rmse])

    for es in errorstats:
        maelist.append(es[0])
        mselist.append(es[1])
        rmselist.append(es[2])

    df = pd.DataFrame({'Alpha value:': avalues, 'MAE': maelist,
                       'MSE': mselist, 'RMSE': rmselist})
    df.set_index('Alpha value:', inplace=True)
    print(df)

    # print("Ridge Regression model output(first 25 variables): ")
    # print()
    # print(df.head(25))

    # coeff_df = pd.DataFrame(reg.coef_, X_columns, columns=['Coefficient'])
    # print(coeff_df)


data = pd.read_csv('Life Expectancy Data.csv')
data = data.fillna(method='bfill')

d = {'Developing': 0, 'Developed': 1}
data['Status'] = data['Status'].map(d).fillna(data['Status'])

# print(data.isnull().any())
# print(data.describe())
errorstats = []

X_columns = ['Status', 'Adult Mortality', 'infant deaths',
          'Alcohol', 'percentage expenditure', 'Hepatitis B', 'Measles ',
          ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ',
          ' HIV/AIDS', 'GDP', 'Population', ' thinness  1-19 years', ' thinness 5-9 years',
          'Income composition of resources', 'Schooling']


Y = data['Life expectancy '].values

print()
print("------------ Model with all variables ------------")
errorstats.append(myRegression(X_columns, Y))


X_columns = ['Status', 'Adult Mortality', 'infant deaths',
          'Alcohol', 'percentage expenditure', 'Hepatitis B', 'Measles ',
          ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ',
          ' HIV/AIDS', 'GDP', ' thinness  1-19 years', ' thinness 5-9 years',
          'Income composition of resources', 'Schooling']

print()
print("------------ Model with population dropped ------------")
errorstats.append(myRegression(X_columns, Y))


X_columns = ['Status', 'Adult Mortality', 'infant deaths',
          'Alcohol', 'percentage expenditure', 'Hepatitis B',
          ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ',
          ' HIV/AIDS', ' thinness  1-19 years', ' thinness 5-9 years',
          'Income composition of resources', 'Schooling']

print()
print("------------ Dropping Measles, GDP ------------")
errorstats.append(myRegression(X_columns, Y))


X_columns = ['Status', 'Adult Mortality', 'infant deaths',
          'Alcohol', 'Hepatitis B',
          ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ',
          ' HIV/AIDS', ' thinness  1-19 years', ' thinness 5-9 years',
          'Income composition of resources', 'Schooling']

print()
print("------------ Dropping percentage expenditure ------------")
errorstats.append(myRegression(X_columns, Y))


maelist = []
mselist = []
rmselist = []
rowlist = ["None", "Population", "Measles, GDP", "percentage expenditure"]

for es in errorstats:
    maelist.append(es[0])
    mselist.append(es[1])
    rmselist.append(es[2])

df = pd.DataFrame({'Variables Dropped:': rowlist, 'MAE': maelist,
                   'MSE': mselist, 'RMSE': rmselist})
df.set_index('Variables Dropped:', inplace=True)
print("------------ Results ------------")
print(df)


X_columns = ['Country', 'Status', 'Adult Mortality', 'infant deaths',
          'Alcohol', 'percentage expenditure', 'Hepatitis B',
          ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ',
          ' HIV/AIDS', ' thinness  1-19 years', ' thinness 5-9 years',
          'Income composition of resources', 'Schooling']

print()
# ------------ Testing Country Accuracy ------------
compareCountries(X_columns, Y)


X_columns = ['Status', 'Adult Mortality', 'infant deaths',
          'Alcohol', 'percentage expenditure', 'Hepatitis B',
          ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ',
          ' HIV/AIDS', ' thinness  1-19 years', ' thinness 5-9 years',
          'Income composition of resources', 'Schooling']

print()
print("------------ Ridge Regression ------------")
testalpha(X_columns, Y, 0)

print()
print("------------ Lasso Regression ------------")
testalpha(X_columns, Y, 1)


# plt.show()

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

