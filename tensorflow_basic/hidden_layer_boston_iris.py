import pandas as pd
import tensorflow as tf

# get data
data_path = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/boston.csv'
boston_data = pd.read_csv(data_path)
# boston_data.head()

# define independent and dependent
independent = boston_data[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax',
       'ptratio', 'b', 'lstat']]
dependent = boston_data[['medv']]
# print(independent.shape, depedent.shape)

# creating model
x = tf.keras.layers.Input(shape=[13])

# hidden layer : increase accuracy
h = tf.keras.layers.Dense(10, activation='swish')(x)
h = tf.keras.layers.Dense(8, activation='swish')(h)
h = tf.keras.layers.Dense(6, activation='swish')(h)

y = tf.keras.layers.Dense(2)(h)
model = tf.keras.models.Model(x, y)
model.compile(loss='mse')

print(model.summary()) # check model

# train model
model.fit(independent, dependent, epochs=1000, verbose = 0)
# check loss
model.fit(independent, dependent, epochs=10, verbose = 2)
 
# using model
prediction = model.predict(independent[0:5])
print('Prediction : \n',prediction,'\noriginal results: \n',dependent[0:5])

# check model expression (weight)
print('Model Expression : ', model.get_weights())

# get data
data_path = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris.csv'
iris = pd.read_csv(data_path)
# print(iris.columns)

# one-hot-encoding
iris_encode = pd.get_dummies(iris)
print(iris_encode.head())

# dependent values is not interger. Classification(Categorizing) is used.
independent = iris_encode[['꽃잎길이','꽃잎폭','꽃받침길이', '꽃받침폭']]
dependent = iris_encode[['품종_setosa',  '품종_versicolor',  '품종_virginica']]
# print(independent.shape, dependent.shape)

# define model
x = tf.keras.layers.Input(shape=[4])

# hidden layer : increase accuracy
h1 = tf.keras.layers.Dense(10, activation='swish')(x)
h2 = tf.keras.layers.Dense(5, activation='swish')(h1)

y = tf.keras.layers.Dense(3, activation='softmax')(h2) # (확률)분류 예측, Softmax, Sigmoid
model = tf.keras.models.Model(x, y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')

print(model.summary()) # check model

# train model
model.fit(independent, dependent, epochs = 1000, verbose=0)

# use model
predict = model.predict(independent[-5:])
print('Prediction : \n', predict)
print('Original Output : \n',dependent[-5:])

