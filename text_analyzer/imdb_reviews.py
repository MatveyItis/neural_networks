import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM
from keras.datasets import imdb

np.random.seed(42)

max_features = 5000
(x_train, y_train), (x_test, y_test) = imdb.load_data(nb_words=max_features)

maxlen = 80
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

# создаем модель
model = Sequential()
# слой для векторного представления слов
model.add(Embedding(max_features, 32, dropout=0.2))
# слойдолго-краткосрочной памяти
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
# полносвязный слой для классификации
model.add(Dense(1, activation="sigmoid"))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=7, validation_data=(x_test, y_test), verbose=1)

# проверяем качество обучения на тестовых данных
scores = model.evaluate(x_test, y_test, batch_size=64)
print("Доля верных ответов на тестовых данных: %.2f%%" % (scores[1] * 100))
