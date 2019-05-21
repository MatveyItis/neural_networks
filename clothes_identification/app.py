from tensorflow.python.keras.datasets import fashion_mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import utils
from tensorflow.python.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

(x_train_1, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = np.copy(x_train_1)
x_train = x_train.reshape(60000, 784)
# нормализация данных
x_train = np.true_divide(x_train, 255)

# преобразование меток в категории
y_train = utils.to_categorical(y_train, 10)

# название классов
classes = ['футболка', 'брюки', 'свитер', 'платье',
           'пальто', 'туфли', 'рубашка', 'кроссовки',
           'сумка', 'ботинки']

# plt.figure(figsize=(10, 10))
# for i in range(100, 150):
#     plt.subplot(5, 10, i - 100 + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(x_train[i], cmap=plt.cm.binary)
#     plt.xlabel(classes[y_train[i]])

# создаем последовательную модель
model = Sequential()

# добавляем уровни сети
model.add(Dense(800, input_dim=784, activation="relu"))
model.add(Dense(10, activation="softmax"))

# компилируем модель
model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

print(model.summary())

# обучаем сеть
model.fit(x_train, y_train, batch_size=200, epochs=100, verbose=1)

predictions = model.predict(x_train)

n = 0
plt.imshow(x_train[n].reshape(28, 28), cmap=plt.cm.binary)
plt.show()

print(predictions[n])

# номер класса изображения который предлагает сеть
print(np.argmax(predictions[n]))
print(classes[np.argmax(predictions[n])])
# номер класса правильного ответа
print(np.argmax(y_train[n]))
print(classes[np.argmax(y_train[n])])
