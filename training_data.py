import cv2
import os
from tqdm import tqdm
import random
import numpy as np
import pickle
from face_detecor import detector


DATADIR = r"images"
CATEGORIES = ["FaceWithGlasses", "FaceWithoutGlasses"]

training_data = []
image_size = 100


def create_training_data():
    for category in CATEGORIES:

        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)

        for img in tqdm(os.listdir(path)):
            try:
                img_array = detector(os.path.join(path, img))
                img_array = cv2.resize(img_array, (image_size, image_size))
                training_data.append([img_array, class_num])
            except Exception as e:
                pass


create_training_data()


random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X)

# X = X.reshape(-1, 200, 200, 1)


pickle_out = open(r"saved_data\X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open(r"saved_data\y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

print(len(training_data))
