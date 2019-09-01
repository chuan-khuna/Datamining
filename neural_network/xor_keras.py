from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np
from keras.utils import plot_model

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

model = Sequential()
model.add(Dense(2, input_dim=2, activation="tanh"))
model.add(Dense(1, activation="sigmoid"))
# gradient descent
sgd = SGD(lr=0.1)
model.compile(loss="binary_crossentropy", optimizer=sgd)

model.fit(x, y, batch_size=4, verbose=1, epochs=2000)

print(model.predict_proba(x))
