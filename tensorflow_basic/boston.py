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
y = tf.keras.layers.Dense(2)(x)
model = tf.keras.models.Model(x, y)
model.compile(loss='mse')

# train model
model.fit(independent, dependent, epochs=1000, verbose = 0)
# check loss
model.fit(independent, dependent, epochs=10, verbose = 2)
 
# using model
prediction = model.predict(independent[0:5])
print('Prediction : \n',prediction,'\noriginal results: \n',dependent[0:5])

# check model expression (weight)
print('Model Expression : ', model.get_weights())

