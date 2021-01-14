import pandas as pd
import tensorflow as tf

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
y = tf.keras.layers.Dense(3, activation='softmax')(x) # (확률)분류 예측, Softmax, Sigmoid
model = tf.keras.models.Model(x, y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')

# train model
model.fit(independent, dependent, epochs = 1000, verbose=0)

# use model
predict = model.predict(independent[-5:])
print('Prediction : \n', pd.DataFrame(predict).round(2))
print('Original Output : \n',dependent[-5:])



