import os
from typing import List

import numpy as np
import cv2

from model import load_model_interpreter, get_embeddings
from image import get_image, get_images, create_cropped_faces, try_crop_face
from margin import Margin
from utils import cosine_distance

model_path = "/home/luan/dev/tcc-eval/mobilefacenet.tflite"
# model_path = "/home/luan/dev/tcc-eval/rafael_student.tflite"
model_path = "/home/luan/dev/tcc-eval/triplet_dist_student.tflite"


interpreter = load_model_interpreter(model_path)

print(interpreter.get_input_details())
print(interpreter.get_output_details())

base_path = "faces"
faces = get_images()

cropped_faces_folder = create_cropped_faces()


def print_model_output(embeddings: np.array, top_k: np.array, dt: float):
    print(f"Top K: {top_k}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings [:5]: {embeddings[:5]}")
    print(f"Embeddings [-5:]: {embeddings[-5:]}")


def print_dist_sim(dist: float, sim: float):
    print(f"Dist: {dist}")
    print(f"Sim: {sim}")


def print_margins_dist(margins: List[Margin], dist: float):
    for m in margins:
        print(f"\tMargin {m.threshold}: {m.evaluate(dist)}")


def print_margin_avg(margins: List[Margin]):
    for m in margins:
        print(f"Margin {m.threshold}")
        print(f"\tAccuracy: {m.get_accuracy()}")
        print(f"\tPrecision: {m.get_precision()}")
        print(f"\tRecall: {m.get_recall()}")
        print(f"\tF1 Score: {m.get_f1_score()}")


def test_flipped_faces():
    total_dt1 = 0
    total_dt2 = 0
    total_dist = 0
    total_sim = 0

    margins = [
        Margin(0.025), Margin(0.050), Margin(0.100), Margin(0.150),
        Margin(0.200), Margin(0.300), Margin(0.400), Margin(0.500),
    ]

    for face in faces:
        face_str = f"{'='*20} {face} {'='*20}"
        print(face_str)
        img1 = get_image(base_path, face)

        img1 = try_crop_face(img1)
        cv2.imwrite(os.path.join(cropped_faces_folder, face), img1)

        embeddings1, top_k, dt1 = get_embeddings(interpreter, img1)
        print_model_output(embeddings1, top_k, dt1)
        total_dt1 += dt1

        img2 = cv2.flip(img1, cv2.ROTATE_180)
        print(f"{'='*15} {face} (flipped) {'='*15}")

        embeddings2, top_k, dt2 = get_embeddings(interpreter, img2)
        print_model_output(embeddings1, top_k, dt1)
        total_dt2 += dt2

        dist, sim = cosine_distance(embeddings1, embeddings2)

        total_dist += dist
        total_sim += sim

        dist_msg = f" {face} dist/sim "
        equal_len = (len(face_str) - len(dist_msg)) // 2
        print(f"{'='*equal_len}{dist_msg}{'='*equal_len}")
        print_dist_sim(dist, sim)

        print_margins_dist(margins, dist)

    print("="*30)

    len_faces = len(faces)
    print(f"Avg. dt1: {total_dt1/len_faces}(ms)")
    print(f"Avg. dt2: {total_dt2/len_faces}(ms)")
    print(f"Avg. dist: {total_dist/len_faces}")
    print(f"Avg. sim: {total_sim/len_faces}")

    print_margin_avg(margins)

# # exit(0)
# face_1 = faces[0]
# print(f"Img1: {face_1}")
# f1 = get_image(base_path, face_1)
# f1 = try_crop_face(f1)
#
# # f1 = Image.open(os.path.join(base_path, face))
# # f1 = f1.resize((112, 112))
# embeddings1, top_k, dt1 = get_embeddings(interpreter, f1)
# print(embeddings1)
#
# for face_2 in faces[1:]:
#     print(f"Img2: {face_2}")
#     f2 = get_image(base_path, face_2)
#     f2 = try_crop_face(f2)
#
#     embeddings2, top_k, dt2 = get_embeddings(interpreter, f2)
#
#     dist, sim = cosine_distance(embeddings1, embeddings2)
#     print("="*30)
#     print_dist_sim(dist, sim)
#     for m in margins:
#         print(f"Margin {m.threshold}: {m.evaluate(dist, False)}")
#
# print("="*30)
# for m in margins:
#     print(f"Margin {m.threshold}")
#     print(f"\tAccuracy: {m.get_accuracy()}")
#     print(f"\tPrecision: {m.get_precision()}")
#     print(f"\tRecall: {m.get_recall()}")
#     print(f"\tF1_score: {m.get_f1_score()}")


if __name__ == "__main__":
    test_flipped_faces()
