import os

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from image import get_images

base_options = python.BaseOptions(model_asset_path="detector.tflite")
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

BASE_FOLDER = "faces"

faces = get_images(BASE_FOLDER)


def save_labels(
    image: mp.Image,
    detection_result: vision.FaceDetectorResult,
    base_path: str,
    filename: str
):
    os.makedirs(base_path, exist_ok=True)

    LABEL_COLOR = (0xff, 0, 0)
    TEXT_COLOR = (0xff, 0, 0)
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    MARGIN = 10
    ROW_SIZE = 10

    label_img = np.copy(image.numpy_view())
    for i, detection in enumerate(detection_result.detections):
        bbox = detection.bounding_box
        start = bbox.origin_x, bbox.origin_y
        end = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height

        cv2.rectangle(label_img, start, end, LABEL_COLOR, 3)

        text_location = (
            MARGIN + bbox.origin_x,
            MARGIN + bbox.origin_y + ROW_SIZE
        )
        text = f"Label {i+1}"
        cv2.putText(label_img, text, text_location,
                    cv2.FONT_HERSHEY_PLAIN, FONT_SIZE,
                    TEXT_COLOR, FONT_THICKNESS)

    label_img = cv2.cvtColor(label_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(base_path, f"label_{filename}.jpg"), label_img)


def save_crop(
    image: mp.Image,
    detection_result: vision.FaceDetectorResult,
    base_path: str,
    filename: str
):
    os.makedirs(base_path, exist_ok=True)

    crop_img = np.copy(image.numpy_view())
    detection = detection_result.detections[0]

    bbox = detection.bounding_box
    x, y = bbox.origin_x, bbox.origin_y
    xx, yy = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height

    crop_img = crop_img[y:yy, x:xx]

    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(base_path, f"crop_{filename}.jpg"), crop_img)


for face in faces:
    face_path = os.path.join(BASE_FOLDER, face)

    image = mp.Image.create_from_file(face_path)

    detection_result = detector.detect(image)
    LABELS_PATH = "cropped/label"
    save_labels(image, detection_result, LABELS_PATH, face.split(".")[0])

    CROP_PATH = "cropped/crop"
    save_crop(image, detection_result, CROP_PATH, face.split(".")[0])
