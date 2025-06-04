import sys
import os

from typing import List, Tuple

import cv2
import numpy as np

from confusion_matrix import ConfusionMatrix
from margin import Margin
from model import Model
from image import get_images, get_image, try_crop_face, create_cropped_faces
from utils import cosine_distance, triplet_loss


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


def log_margins_avg(
        margins: List[Margin],
        file: str,
        model_name: str,
        title: str = ""
):
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


class TestModel:

    model: Model

    def __init__(
        self,
        model: Model,
        margins: List[Margin],
        faces_base_path: str
    ):
        self.model = model
        self.faces_base_path = faces_base_path
        self.faces = get_images(faces_base_path)

    def test_flipped_faces2(
        self,
        run: int,
        file: str,
        title: str,
    ) -> Tuple[float, int, float]:
        total_dt = 0
        qnt_inference = 0
        len_faces = len(self.faces)

        cm = ConfusionMatrix(len_faces)
        embeddings_cache = [None for _ in range(len_faces)]
        flipped_embeddings_cache = [None for _ in range(len_faces)]

        cropped_faces_folder = create_cropped_faces()

        for i in range(len(self.faces)):
            lowest_dist = sys.float_info.max
            face_index = -1

            face_anchor = self.faces[i]

            if flipped_embeddings_cache[i] is not None:
                anchor_embeddings = flipped_embeddings_cache[i]
            else:
                img_anchor = get_image(
                        self.faces_base_path, face_anchor, flip=True)
                img_anchor = try_crop_face(img_anchor)
                cv2.imwrite(
                        os.path.join(
                            cropped_faces_folder,
                            f"flipped_{face_anchor}"),
                        cv2.cvtColor(img_anchor, cv2.COLOR_RGB2BGR))
                print(f"Image dtype: {img_anchor.dtype}")
                anchor_embeddings, _, current_dt = self.model.get_embeddings(
                        img_anchor)
                total_dt += current_dt
                qnt_inference += 1
                flipped_embeddings_cache[i] = anchor_embeddings

            for j in range(len(self.faces)):
                face_ = self.faces[j]

                if embeddings_cache[j] is not None:
                    face_embeddings = embeddings_cache[j]
                else:
                    img = get_image(self.faces_base_path, face_)
                    img = try_crop_face(img)

                    cv2.imwrite(
                        os.path.join(
                            cropped_faces_folder,
                            f"{face_}"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                    face_embeddings, _, current_dt = self.model.get_embeddings(
                            img)
                    total_dt += current_dt
                    qnt_inference += 1

                    embeddings_cache[j] = face_embeddings

                dist, sim = cosine_distance(anchor_embeddings, face_embeddings)

                print(f"{'='*15} {face_anchor} vs {face_} {'='*15}")
                print_dist_sim(dist, sim)
                print(f"Lowest dist: {lowest_dist}")
                print(f"Face index: {face_index}")
                if dist < lowest_dist:
                    lowest_dist = dist
                    face_index = j

            cm.add(i, face_index)
            print(f"{'='*15} Target vs Predicted {'='*15}")
            print(f"Target: {face_anchor}")
            print(f"Predicted: {self.faces[face_index]}")
            print(f"Dist: {lowest_dist}")

        cm.log(f"confusion_matrix_{run:02}.log", self.model.name)

        print("="*30)
        avg_dt = total_dt/(len_faces*len_faces)
        print(f"Avg. dt: {avg_dt}(ms)")

        return total_dt, len_faces, avg_dt

    def test_flipped_faces(
        self,
        margins: List[Margin],
        file: str,
        title: str,
        confusion_matrix_bool: ConfusionMatrix
    ):
        cropped_faces_folder = create_cropped_faces()

        total_dt1 = 0
        total_dt2 = 0
        total_dist = 0
        total_sim = 0

        for face in self.faces:
            face_str = f"{'='*20} {face} {'='*20}"
            print(face_str)
            img1 = get_image(self.faces_base_path, face)

            img1 = try_crop_face(img1)
            cv2.imwrite(os.path.join(cropped_faces_folder, face), img1)

            embeddings1, top_k, dt1 = self.model.get_embeddings(img1)
            # print_model_output(embeddings1, top_k, dt1)
            total_dt1 += dt1

            img2 = cv2.flip(img1, cv2.ROTATE_180)
            print(f"{'='*15} {face} (flipped) {'='*15}")

            embeddings2, top_k, dt2 = self.model.get_embeddings(img2)
            # print_model_output(embeddings1, top_k, dt1)
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

        len_faces = len(self.faces)
        print(f"Avg. dt1: {total_dt1/len_faces}(ms)")
        print(f"Avg. dt2: {total_dt2/len_faces}(ms)")
        print(f"Avg. dist: {total_dist/len_faces}")
        print(f"Avg. sim: {total_sim/len_faces}")

        print_margin_avg(margins)
        log_margins_avg(margins, file, title)

        confusion_matrix_bool.log("confusion_matrix.log", self.model.name)

    def test_diff(self, margins: List[Margin], file: str, title: str):
        face_1 = self.faces[0]
        print(f"Img1: {face_1}")
        f1 = get_image(self.faces_base_path, face_1)
        f1 = try_crop_face(f1)

        f1 = get_image(self.faces_base_path, face_1)
        f1 = try_crop_face(f1)
        embeddings1, top_k, dt1 = self.model.get_embeddings(f1)

        for face_2 in self.faces[1:]:
            print(f"Img2: {face_2}")
            f2 = get_image(self.faces_base_path, face_2)
            f2 = try_crop_face(f2)

            embeddings2, top_k, dt2 = self.model.get_embeddings(f2)

            dist, sim = cosine_distance(embeddings1, embeddings2)

            for m in margins:
                m.evaluate_with_metrics(dist, False)
            print("="*30)
            print_dist_sim(dist, sim)
            print_margins_dist(margins, dist, False)

        print_margin_avg(margins)
        log_margins_avg(margins, file, title)

    def triplet_test(self, margins: List[Margin], file: str, title: str):
        base_path = "data"

        anchor = get_images(os.path.join(base_path, "anchor"))
        anchor.sort()
        anchor = anchor

        positive = get_images(os.path.join(base_path, "positive"))
        positive.sort()
        positive = positive

        negative = get_images(os.path.join(base_path, "negative"))
        negative.sort()
        negative = negative

        n = len(anchor)

        print(f"Anchor len: {len(anchor)}")
        print(f"Positive len: {len(positive)}")
        print(f"Negative len: {len(negative)}")

        for i in range(n):
            anchor_img = get_image(
                    os.path.join(base_path, "anchor"), anchor[i])
            anchor_img = try_crop_face(anchor_img)
            embeddings_anchor_img, top_k, dt = self.model.get_embeddings(anchor_img)
            print_model_output(embeddings_anchor_img, top_k, dt)

            positive_img = get_image(
                    os.path.join(base_path, "positive"), positive[i])
            positive_img = try_crop_face(positive_img)
            embeddings_pos_img, top_k, dt = self.model.get_embeddings(positive_img)
            print_model_output(embeddings_pos_img, top_k, dt)

            negative_img = get_image(
                    os.path.join(base_path, "negative"), negative[i])
            negative_img = try_crop_face(negative_img)
            embeddings_neg_img, top_k, dt = self.model.get_embeddings(negative_img)
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
