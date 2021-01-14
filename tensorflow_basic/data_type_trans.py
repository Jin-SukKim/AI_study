import pandas as pd

# get data
data_path = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris2.csv'
iris = pd.read_csv(data_path)
print(iris.head()) # 품종 is integer, pandas reads it as interger so one-hot-encoding doenst work.

print(iris.dtypes)

# change data type
iris['품종'] = iris['품종'].astype('category')
print(iris.head())

# one-hot-encoding
iris_encode = pd.get_dummies(iris)
print(iris_encode.head())

# check N/A value
print(iris.isna().sum())

print(iris.tail())

# give mean value to NaN value
mean = iris['꽃잎폭'].mean()
iris['꽃잎폭'] = iris['꽃잎폭'].fillna(mean)
iris.tail()