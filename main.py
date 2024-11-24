import os
import sys

from typing import List

import numpy as np
import cv2

from confusion_matrix import ConfusionMatrix
from model import load_model_interpreter, get_embeddings
from image import get_image, get_images, create_cropped_faces, try_crop_face
from margin import Margin
from utils import cosine_distance, triplet_loss

current_model = sys.argv[1] if len(sys.argv) > 1 else "mobilefacenet.tflite"
model_path = os.path.join(os.curdir, current_model)

# model_path = "/home/luan/dev/tcc-eval/mobilefacenet.tflite"
# model_path = "/home/luan/dev/tcc-eval/rafael_student.tflite"
# model_path = "/home/luan/dev/tcc-eval/triplet_dist_student.tflite"
# model_path = "/home/luan/dev/tcc-eval/pair_model.tflite"

interpreter = load_model_interpreter(model_path)

print(interpreter.get_input_details())
print(interpreter.get_output_details())

base_path = "faces"
faces = get_images(base_path)

cropped_faces_folder = create_cropped_faces()


def print_model_output(embeddings: np.array, top_k: np.array, dt: float):
    print(f"Top K: {top_k}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings [:5]: {embeddings[:5]}")
    print(f"Embeddings [-5:]: {embeddings[-5:]}")


def print_dist_sim(dist: float, sim: float):
    print(f"Dist: {dist}")
    print(f"Sim: {sim}")


def print_margins_dist(margins: List[Margin], dist: float, is_true=True):
    for m in margins:
        res = m.evaluate_with_metrics(dist, is_true)
        print(f"\tMargin {m.threshold}: {res} ({is_true})")


def print_margin_avg(margins: List[Margin]):
    print("="*50)
    for m in margins:
        print(f"Margin {m.threshold}")
        print(f"\tAccuracy: {m.get_accuracy()}")
        print(f"\tPrecision: {m.get_precision()}")
        print(f"\tRecall: {m.get_recall()}")
        print(f"\tF1 Score: {m.get_f1_score()}")


def log_margins_avg(margins: List[Margin], file: str, title: str = ""):
    model_name = current_model.split(".")[0]
    base_folder = f"logs/{model_name}"
    os.makedirs(base_folder, exist_ok=True)

    equals = "="*25
    file_path = os.path.join(base_folder, file)
    print(file_path)
    with open(file_path, "w") as file:
        file.write(f"{equals}{title}{equals}\n")
        for m in margins:
            file.write(f"Margin {m.threshold}\n")
            file.write(f"\tAccuracy: {m.get_accuracy()}\n")
            file.write(f"\tPrecision: {m.get_precision()}\n")
            file.write(f"\tRecall: {m.get_recall()}\n")
            file.write(f"\tF1 Score: {m.get_f1_score()}\n")


def test_flipped_faces(
        margins: List[Margin],
        file: str,
        title: str,
        confusion_matrix_bool: ConfusionMatrix
):
    total_dt1 = 0
    total_dt2 = 0
    total_dist = 0
    total_sim = 0

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

        for m in margins:
            res = m.evaluate_with_metrics(dist, True)
            confusion_matrix_bool.add(1, 1 if res else 0)

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
    log_margins_avg(margins, file, title)

    model_name = current_model.split(".")[0]
    confusion_matrix_bool.log("confusion_matrix.log", model_name)


def test_diff(margins: List[Margin], file: str, title: str):
    faces = get_images(base_path)

    face_1 = faces[0]
    print(f"Img1: {face_1}")
    f1 = get_image(base_path, face_1)
    f1 = try_crop_face(f1)

    f1 = get_image(base_path, face_1)
    f1 = try_crop_face(f1)
    embeddings1, top_k, dt1 = get_embeddings(interpreter, f1)

    for face_2 in faces[1:]:
        print(f"Img2: {face_2}")
        f2 = get_image(base_path, face_2)
        f2 = try_crop_face(f2)

        embeddings2, top_k, dt2 = get_embeddings(interpreter, f2)

        dist, sim = cosine_distance(embeddings1, embeddings2)

        for m in margins:
            m.evaluate_with_metrics(dist, False)
        print("="*30)
        print_dist_sim(dist, sim)
        print_margins_dist(margins, dist, False)

    print_margin_avg(margins)
    log_margins_avg(margins, file, title)


def test_diff_faces(margins: List[Margin]):
    faces = get_images(base_path)

    from random import shuffle
    faces_index = [i for i in range(len(faces))]
    faces_index_shuffle = faces_index.copy
    shuffle(faces_index_shuffle)

    margin_pairs = [
        (faces_index[i], faces_index_shuffle[i]) for i in range(len(faces))
    ]

    print(margin_pairs)
    print("="*30)


def triplet_test(margins: List[Margin], file: str, title: str):
    base_path = "data"

    anchor = get_images(os.path.join(base_path, "anchor"))
    anchor.sort()
    anchor = anchor[:2]

    positive = get_images(os.path.join(base_path, "positive"))
    positive.sort()
    positive = positive[:2]

    negative = get_images(os.path.join(base_path, "negative"))
    negative.sort()
    negative = negative[:2]


    n = len(anchor)

    print(f"Anchor len: {len(anchor)}")
    print(f"Positive len: {len(positive)}")
    print(f"Negative len: {len(negative)}")

    for i in range(n):
        anchor_img = get_image(
                os.path.join(base_path, "anchor"), anchor[i])
        anchor_img = try_crop_face(anchor_img)
        embeddings_anchor_img, top_k, dt = get_embeddings(interpreter, anchor_img)
        print_model_output(embeddings_anchor_img, top_k, dt)

        positive_img = get_image(
                os.path.join(base_path, "positive"), positive[i])
        positive_img = try_crop_face(positive_img)
        embeddings_pos_img, top_k, dt = get_embeddings(interpreter, positive_img)
        print_model_output(embeddings_pos_img, top_k, dt)

        negative_img = get_image(
                os.path.join(base_path, "negative"), negative[i])
        negative_img = try_crop_face(negative_img)
        embeddings_neg_img, top_k, dt = get_embeddings(interpreter, negative_img)
        print_model_output(embeddings_neg_img, top_k, dt)

        print(f"{'=' * 15} Anchor ({anchor[i]}) vs Positive ({positive[i]}) {'=' * 15}")
        dist_ap, sim_ap = cosine_distance(
                embeddings_anchor_img, embeddings_pos_img)
        print_dist_sim(dist_ap, sim_ap)
        print_margins_dist(margins, dist_ap, is_true=True)

        print(f"{'=' * 15} Anchor ({anchor[i]}) vs Negative ({negative[i]}) {'=' * 15}")
        dist_an, sim_an = cosine_distance(
                embeddings_anchor_img, embeddings_neg_img)
        print_dist_sim(dist_an, sim_an)
        print_margins_dist(margins, dist_ap, is_true=False)

        print(f"{'=' * 15} Triplet loss {'=' * 15}")
        for m in margins:
            print(f"Threshold: {m.threshold}")
            loss = triplet_loss(
                        embeddings_anchor_img,
                        embeddings_pos_img,
                        embeddings_neg_img,
                        m.threshold)
            print(f"\tLoss: {loss}")

    print_margin_avg(margins)
    log_margins_avg(margins, file, title)


if __name__ == "__main__":

    base = 0.000
    scale = 0.0001
    start = 1
    end = 6

    def create_margin():
        return Margin.from_range(base, scale, start, end)

    margins = create_margin()

    cm = ConfusionMatrix(2)
    # test_flipped_faces(margins, "flipped.log", "Flipped", cm)
    # test_diff(margins, "flipped+diff.log", "Flipped and Diff")
    # test_diff(create_margin(), "diff.log", "Diff")
    triplet_test(create_margin(), "triplet.log", "Triplet")
