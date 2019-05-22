import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.datasets import mnist

numpy.random.seed(42)

(x_train_1, y_train), (x_test, y_test) = mnist.load_data()

x_train = numpy.copy(x_train_1)
x_train = x_train.reshape(60000, 784)
x_train = x_train.astype('float32')
x_train = numpy.true_divide(x_train, 255)

y_train = np_utils.to_categorical(y_train, 10)

model = Sequential()
model.add(Dense(800, input_dim=784, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

print(model.summary())

model.fit(x_train, y_train, batch_size=200, validation_split=0.1, epochs=100, verbose=1)

predictions = model.predict(x_train)

predictions = numpy.argmax(predictions, axis=1)
