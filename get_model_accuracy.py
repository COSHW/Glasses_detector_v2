import pickle
from tensorflow.python.keras.models import load_model
import numpy as np


pickle_in = open(r"saved_data\X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open(r"saved_data\y.pickle", "rb")
y = pickle.load(pickle_in)

X = X.reshape(-1, 200, 200, 1)
X = X/255.0
y = np.array(y).astype(float)

model = load_model(r"saved_data\model.h5")

model.evaluate(X, y, verbose=2)
