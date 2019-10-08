import tensorflow as tf
from tensorflow.python.keras.models import load_model
import argparse
import os
import numpy as np
from training_data import create_training_data


dataset_folder = False

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_folder", help="folder with two folders of images to be processed")
args = parser.parse_args()

if args.dataset_folder:
    dataset_folder = args.dataset_folder

if dataset_folder is False:
    print("Укажите папку с датасетом. Правильный пример: python .\predict_dataset.py --dataset_folder=<folder path>")
else:
    CATEGORIES = os.listdir(dataset_folder)

    model = load_model(r"saved_data\model.h5")
    training_data = []

    print("----------------------------------------------------------------------")
    print("Начинаю подготовку изображений")
    print("----------------------------------------------------------------------")
    X, y, path_of_files = create_training_data(dataset_folder, CATEGORIES)

    print("----------------------------------------------------------------------")
    print("Начинаю подсчёт точности")
    print("----------------------------------------------------------------------")
    model.evaluate(X, y, verbose=2)

    print("----------------------------------------------------------------------")
    print("Начинаю поиск лиц с очками")
    print("----------------------------------------------------------------------")
    for image, path in zip(X, path_of_files):
        image = np.expand_dims(image, axis=0)
        image = tf.cast(image, tf.float64)
        prediction = model.predict([image])
        if int(prediction[0][0]) == 0:
            print("Guess: {} - {}".format(CATEGORIES[int(prediction[0][0])], path))
