import os

from typing import List

import cv2

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades
                                        + "haarcascade_frontalface_default.xml"
                                        )


def get_images() -> List[str]:
    return os.listdir("faces")


def create_cropped_faces() -> str:
    cropped_faces_folder = os.path.join(
            os.path.curdir,
            "cropped_faces")
    os.makedirs(cropped_faces_folder, exist_ok=True)

    return cropped_faces_folder


def get_image(base_path: str, face: str, flip=False) -> cv2.UMat:
    img1 = cv2.imread(os.path.join(base_path, face))
    if flip:
        img1 = cv2.flip(img1, cv2.ROTATE_180)
    return img1


def try_crop_face(img: cv2.UMat) -> cv2.UMat:
    detected_face = face_classifier.detectMultiScale(img,
                                                     scaleFactor=1.1,
                                                     minNeighbors=5,
                                                     minSize=(40, 40))
    return cv2.resize(img, (112, 112))
    if len(detected_face) >= 1:
        (x, y, w, h) = detected_face[-1]
        img = img[y:y+h, x:x+w]
