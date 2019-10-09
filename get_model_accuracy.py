import pickle
from tensorflow.python.keras.models import load_model
import numpy as np


pickle_in = open(r"saved_data\X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open(r"saved_data\y.pickle", "rb")
y = pickle.load(pickle_in)
<<<<<<< HEAD

X = X.reshape(-1, 200, 200, 1)
=======
X = X.reshape(-1, 100, 100, 1)
>>>>>>> 68617e521dd6cf27f48edb101401be2540fc2c58
X = X/255.0
y = np.array(y).astype(float)

model = load_model(r"saved_data\model.h5")

model.evaluate(X, y, verbose=2)
