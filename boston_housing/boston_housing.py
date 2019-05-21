import numpy as np
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense

# задаем сид для повторяемости результата
np.random.seed(42)

# load data
(x_train_1, y_train), (x_test_1, y_test) = boston_housing.load_data()

x_train = np.copy(x_train_1)
x_test = np.copy(x_test_1)

mean = x_train.mean(axis=0)
std = x_train.std(axis=0)
x_train -= mean
x_train = np.true_divide(x_train, std)
x_test -= mean
x_test = np.true_divide(x_test, std)

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1)

mse, mae = model.evaluate(x_test, y_test, verbose=0)
print(mae)

# смотрим что выкинет нам наша сеть
pred = model.predict(x_test)

print(pred[1][0], y_test[1])
print(pred[25][0], y_test[25])
print(pred[50][0], y_test[50])
print(pred[75][0], y_test[75])
print(pred[100][0], y_test[100])

