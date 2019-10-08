import cv2
import os
from tqdm import tqdm
import random
import numpy as np
import pickle
from face_detecor import detector


DATADIR = r"dataset"
CATEGORIES = ["FaceWithGlasses", "FaceWithoutGlasses"]
training_data = []
image_size = 200


def create_training_data(DATADIR, CATEGORIES):
    for category in CATEGORIES:

        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)

        for img in tqdm(os.listdir(path)):
            try:
                img_array = detector(os.path.join(path, img))
                img_array = cv2.resize(img_array, (200, 200))
                training_data.append([img_array, class_num, os.path.join(path, img)])
            except Exception as e:
                pass

    random.shuffle(training_data)

    X = []
    y = []
    z = []

    for features, label, path_to_image in training_data:
        X.append(features)
        y.append(label)
        z.append(path_to_image)

    X = np.array(X)
    X = np.reshape(X, (-1, 200, 200, 1))
    y = np.array(y).astype(float)

    return X, y, z

if __name__ == "__main__":
    X, y, z = create_training_data(DATADIR, CATEGORIES)

    pickle_out = open(r"saved_data\X.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open(r"saved_data\y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()

    print(len(training_data))
