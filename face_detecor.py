import cv2


def detector(img):
    face_cascade = cv2.CascadeClassifier(r'saved_data\haarcascade_frontalface_default.xml')
    gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    faces = face_cascade.detectMultiScale(gray, 1.4, 5)
    crop_img = False
    for (x, y, w, h) in faces:
        cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2)
        crop_img = gray[y:y + h, x:x + w]
    if len(faces) != 0:
        return crop_img
    else:
        return gray
