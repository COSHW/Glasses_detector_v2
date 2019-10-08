from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
import pickle
import numpy as np


pickle_in = open(r"saved_data\X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open(r"saved_data\y.pickle", "rb")
y = pickle.load(pickle_in)
y = np.array(y).astype(float)

X = X/255.0

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dropout(0.2))


model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X, y, batch_size=50, epochs=15, validation_split=0.1)
model.save(r"saved_data\model.h5")
