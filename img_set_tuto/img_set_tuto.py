import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
# MNIST : 손글씨 이미지셋(60000,28,28)
# CIFAR10 : 10가지 분류 사물 이미지셋(50000,32,32)

# download images
# (independent, dependent), _ = tf.keras.datasets.mnist.load_data()
(mnist_x, mnist_y), _ = tf.keras.datasets.mnist.load_data()
print(mnist_x.shape, mnist_y.shape)

(cifar_x, cifar_y), _ = tf.keras.datasets.cifar10.load_data()
print(cifar_x.shape, cifar_y.shape)

# print images on screen
print(mnist_y[:10])
plt.imshow(mnist_x[0], cmap='gray')
plt.show()

print(cifar_y[:10]) # https://www.cs.toronto.edu/~kriz/cifar.html
plt.imshow(cifar_x[0]) # color image
plt.show()

# check dimension
print(mnist_x.shape, mnist_y.shape)
print(cifar_x.shape, cifar_y.shape)

# 1 dimension
x1 = np.array([1,2,3,4,5])
print(x1.shape)
print(mnist_y[:5])
print(mnist_y[:5].shape)

# 2 dimension
x2 = np.array([[1,2,3,4,5]])
print(x2.shape)

# 3 dimension
x3 = np.array([[[1],[2],[3],[4],[5]]])
print(x3.shape)
print(cifar_y[:5])
print(cifar_y[:5].shape)


