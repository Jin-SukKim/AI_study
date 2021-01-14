import tensorflow as tf
import pandas as pd

# using reshape

# define data
# (independant, dependant), _ = tf.keras.datasets.mnist.load_data()
(mnist_x, mnist_y), _ = tf.keras.datasets.mnist.load_data()
independant = mnist_x.reshape(60000, 784) # change img data to table, x = 28, y = 28, xy = 728 pixels
dependant = pd.get_dummies(independant)
print(mnist_x, dependant)

# define model
x = tf.keras.layers.Input(shape=[784])
h = tf.keras.layers.Dense(84, activation='swish')(x)
y = tf.keras.layers.Dense(10, activation='softmax')(x)
model = tf.keras.models.Model(x, y)
mode.compile(loss='categorical_crossentropy', metrics='accuracy')

# learning model
model.fit(independant, dependant, epochs=10)

# using model
predict = model.predict(independant[:5])
print(pd.DataFrame(pred).round(2))
print(dependant[:5])



# using flatten
(mnist_x, mnist_y), _ = tf.keras.layers.datasets.mnist.load_data()
dependant = pd.get_dummies(mnist_y)
print(mnist_x.shape, dependant.shape)

# define model
x = tf.keras.layers.Input(shape=[28, 28]) # x, y
h = tf.keras.layers.Flatten()(x) # 영상을 일차원으로 바꿔주는 레이어
h = tf.keras.layers.Dense(84, activation='swish')(h)
y = tf.keras.layers.Dense(10, activation='softmax')(h)
model = tf.keras.models.Model(x, y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')

# learning model
model.fit(mnist_x, dependant, epochs=10)

# using model
predict = model.predict(mnist_x[:5])
print(pd.DataFrame(predict).round(2))
print(dependant[:5])
