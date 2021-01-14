# ctrl + shift + P -> python:select interpreter -> change python interpreter version
import pandas as pd
import tensorflow as tf
# get data
file_path = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/lemonade.csv'
lemonade = pd.read_csv(file_path)

# get indepedent variable ans dependant variable
temperature = lemonade[['온도']]
sell_count = lemonade[['판매량']]

# define model
x = tf.keras.layers.Input(shape = [1]) # number of independent varaible
y = tf.keras.layers.Dense(1)(x) # number of dependent variables
model = tf.keras.models.Model(x,y)
model.compile(loss='mse') # method to learn, how
# learning model
# epochs : how many times to learn
# verbose : display leraning process and result while learning
model.fit(temperature, sell_count, epochs=10000, verbose=0)

model.fit(temperature, sell_count, epochs=10)

prediction = model.predict([[15]])
# use model
print(f'Prediction : {prediction}')

