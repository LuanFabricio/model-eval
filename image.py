import os

from typing import List

import cv2

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades
                                        + "haarcascade_frontalface_default.xml"
                                        )


def get_images(path: str) -> List[str]:
    return os.listdir(path)


def create_cropped_faces() -> str:
    cropped_faces_folder = os.path.join(
            os.path.curdir,
            "cropped_faces")
    os.makedirs(cropped_faces_folder, exist_ok=True)

    return cropped_faces_folder


def get_image(base_path: str, face: str, flip=False) -> cv2.UMat:
    print(f"Path: {base_path}/{face}")
    img1 = cv2.imread(os.path.join(base_path, face))
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    if flip:
        img1 = cv2.flip(img1, cv2.ROTATE_180)
    return img1


# TODO: Corrigr nome da função
def try_crop_face(img: cv2.UMat) -> cv2.UMat:
    return cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
