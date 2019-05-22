import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils

numpy.random.seed(42)

# Загружаем данные
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# Размер мини-выборки
batch_size = 32
# Количество классов изображений
nb_classes = 10
# Количество эпох для обучения
nb_epoch = 25
# Размер изображений
img_rows, img_cols = 32, 32
# Количество каналов в изображении: RGB
img_channels = 3

# Нормализуем данные
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Преобразуем метки в категории
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# Создаем последовательную модель
model = Sequential()
# Первый сверточный слой
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(32, 32, 3), activation='relu'))
# Второй сверточный слой
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
# Первый слой подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))
# Слой регуляризации Dropout
model.add(Dropout(0.25))

# Третий сверточный слой
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
# Четвертый сверточный слой
model.add(Conv2D(64, (3, 3), activation='relu'))
# Второй слой подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))
# Слой регуляризации Dropout
model.add(Dropout(0.25))
# Слой преобразования данных из 2D представления в плоское
model.add(Flatten())
# Полносвязный слой для классификации
model.add(Dense(512, activation='relu'))
# Слой регуляризации Dropout
model.add(Dropout(0.5))
# Выходной полносвязный слой
model.add(Dense(nb_classes, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer="SGD",
              metrics=['accuracy'])
# Обучаем модель
model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=nb_epoch,
          validation_split=0.1,
          shuffle=True,
          verbose=2)

# Оцениваем качество обучения модели на тестовых данных
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Точность работы на тестовых данных: %.2f%%" % (scores[1] * 100))

# numpy.random.seed(42)
#
# (x_train_1, y_train), (x_test_1, y_test) = cifar10.load_data()
#
# x_train = numpy.copy(x_train_1)
# x_test = numpy.copy(x_test_1)
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train = numpy.true_divide(x_train, 255)
# x_test = numpy.true_divide(x_test, 255)
#
# y_train = np_utils.to_categorical(y_train, 10)
# y_test = np_utils.to_categorical(y_test, 10)
#
# # x_train = x_train.reshape(50000, 3, 32, 32)
#
# # create model
# model = Sequential()
# # первый сверточный слой
# model.add(Conv2D(32, (3, 3), padding='same', input_shape=(3, 32, 32), activation="relu"))
# # второй сверточный слой
# model.add(Conv2D(32, (3, 3), padding='same', activation="relu"))
# # слой подвыборки
# model.add(MaxPooling2D(pool_size=(2, 2)))
# # слой регуляризации
# model.add(Dropout(0.25))
# # третий сверточный слой
# model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
# # четвертый сверточный слой
# model.add(Conv2D(64, (3, 3), activation="relu"))
# # второй слой подвыборки
# model.add(MaxPooling2D(pool_size=(2, 2)))
# # слой регуляризации
# model.add(Dropout(0.25))
#
# # Классификатор
# # преобразование из двумерного массива в плоский
# model.add(Flatten())
# # полносвязный слой
# model.add(Dense(512, activation="relu"))
# # слой регуляризации
# model.add(Dropout(0.5))
# # выходной слой
# model.add(Dense(10, activation="softmax"))
#
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy',
#               optimizer=sgd,
#               metrics=['accuracy'])
# model.fit(x_train, y_train, batch_size=32, epochs=25, validation_split=0.1, shuffle=True)
