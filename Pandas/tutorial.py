# pd.read_csv('/경로/파이명.csv') # 파일 읽어오기
# print(데이ㅣ터.shape) # 모양 확인하기
# 데이터[['칼럼명1','칼럼명2','칼럼명3']]
# print(데이터.columns) # 칼럼 이름 출력하기
# 데이터.head() # 맨 위 5개 관측치 출력하기

import pandas as pd

# read data
file_path = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/lemonade.csv'
lemonade = pd.read_csv(file_path)

file_path = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/boston.csv'
boston = pd.read_csv(file_path)

file_path = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris.csv'
iris = pd.read_csv(file_path)

# check data model
print(lemonade.shape)
print(boston.shape)
print(iris.shape,'\n')

# print column name
print(lemonade.columns)
print(boston.columns)
print(iris.columns,'\n')

# get indepedent variable ans dependant variable
temperature = lemonade[['온도']]
sell_count = lemonade[['판매량']]
print(temperature.shape, sell_count.shape)

b_indep = boston[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax',
       'ptratio', 'b', 'lstat']]
b_sub = boston[['medv']]
print(b_indep.shape, b_sub.shape)

i_indep = iris[['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭']]
i_sub = iris[['품종']]
print(i_indep.shape, i_sub.shape,'\n')

lemonade.head()
boston.head()
iris.head()