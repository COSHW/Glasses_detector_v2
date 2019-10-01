import cv2
from tensorflow.python.keras.models import load_model
from face_detecor import detector
import argparse


CATEGORIES = ["FaceWithGlasses", "FaceWithoutGlasses"]


def prepare(filepath):
    img_array = detector(filepath)
    img_array = cv2.resize(img_array, (100, 100))
    img_array = img_array.reshape(-1, 100, 100, 1)
    return img_array


file_name = r"images\FaceWithoutGlasses\18.01anne-hathaway.jpg"
file_names = False

parser = argparse.ArgumentParser()
parser.add_argument("--image", help="image to be processed")
args = parser.parse_args()

if args.image:
    file_names = args.image
    file_names = file_names.split(", ")

model = load_model(r"saved_data\model.h5")

if file_names == False:
    prediction = model.predict([prepare(file_name)])
    print("Guess: {}".format(CATEGORIES[int(prediction[0][0])]))
else:
    for image in file_names:
        prediction = model.predict([prepare(image)])
        if int(prediction[0][0]) == 0:
            print("Guess: {} - {}".format(CATEGORIES[int(prediction[0][0])], image))
            print(model.predict_proba(prepare(image)))
