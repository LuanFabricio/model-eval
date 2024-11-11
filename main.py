import time
import os

from typing import List, Tuple

import tflite_runtime.interpreter as tflite
# from PIL import Image
import numpy as np
import cv2

from margin import Margin

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades
                                        + "haarcascade_frontalface_default.xml"
                                        )

# model_path = "/home/luan/dev/tcc-eval/mobilefacenet.tflite"
# model_path = "/home/luan/dev/tcc-eval/rafael_student.tflite"
model_path = "/home/luan/dev/tcc-eval/triplet_dist_student.tflite"


def load_model_interpreter(model_path: str) -> tflite.Interpreter:
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    return interpreter


interpreter = load_model_interpreter(model_path)

print(interpreter.get_input_details())
print(interpreter.get_output_details())


def test_image(interpreter: tflite.Interpreter, image: np.array) -> Tuple[np.array, np.array, float]:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # NxHxWxC, H:1, W:2
    # height = input_details[0]['shape'][1]
    # width = input_details[0]['shape'][2]

    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32

    # add N dim
    input_data = np.expand_dims(image, axis=0)
    input_mean = 0
    input_std = 1

    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)

    start_time = time.time()
    interpreter.invoke()
    stop_time = time.time()

    dt = (stop_time - start_time) * 1000
    print('time: {:.3f}ms'.format(dt))

    output_data = interpreter.get_tensor(output_details[0]['index'])

    results = np.squeeze(output_data)
    top_k = results.argsort()[-5:][::-1]

    return results, top_k, dt


def get_images() -> List[str]:
    return os.listdir("faces")


def cosine_distance(v1: np.ndarray, v2: np.ndarray) -> List[float]:
    v1_v2 = np.dot(v1, v2)
    v1_len = np.linalg.norm(v1)
    v2_len = np.linalg.norm(v2)

    cos_sim = v1_v2 / (v1_len * v2_len)
    return 1 - cos_sim, cos_sim


base_path = "faces"
faces = get_images()

total_dt1 = 0
total_dt2 = 0
total_dist = 0
total_sim = 0

cropped_faces_folder = os.path.join(
        os.path.curdir,
        "cropped_faces")
os.makedirs(cropped_faces_folder, exist_ok=True)


margins = [
    Margin(0.10),
    Margin(0.15),
    Margin(0.20),
    Margin(0.30),
    Margin(0.40),
    Margin(0.50),
]

for face in faces:
    print(f"Face: {face}")

    img1 = cv2.imread(os.path.join(base_path, face))
    img1 = cv2.flip(img1, 1)

    detected_face = face_classifier.detectMultiScale(img1,
                                                     scaleFactor=1.1,
                                                     minNeighbors=5,
                                                     minSize=(40, 40))

    if len(detected_face) >= 1:
        (x, y, w, h) = detected_face[-1]
        img1 = img1[y:y+h, x:x+w]

    # cv2.imshow(f"debug ({i})", img1)
    # cv2.waitKey(0)

    img1 = cv2.resize(img1, (112, 112))
    cv2.imwrite(os.path.join(cropped_faces_folder, face), img1)

    # img1 = Image.open(os.path.join(base_path, face))
    # img1 = img1.resize((112, 112))
    embeddings1, top_k, dt1 = test_image(interpreter, img1)
    print(top_k)
    print(embeddings1.shape)
    print(embeddings1[:5])
    print(embeddings1[-5:])
    total_dt1 += dt1

    img2 = cv2.flip(img1, cv2.ROTATE_180)
    # img2 = np.flip(img1, axis=2)
    embeddings2, top_k, dt2 = test_image(interpreter, img2)
    total_dt2 += dt2

    dist, sim = cosine_distance(embeddings1, embeddings2)
    print(top_k)
    print(embeddings2.shape)
    print(embeddings2[:5])
    print(embeddings2[-5:])

    total_dist += dist
    total_sim += sim
    print(f"Cos dist: {dist}")
    print(f"Cos sim: {sim}")

    for m in margins:
        print(f"\tMargin {m.threshold}: {m.evaluate(dist)}")

print("="*30)

len_faces = len(faces)
print(f"Avg. dt1: {total_dt1/len_faces}(ms)")
print(f"Avg. dt2: {total_dt2/len_faces}(ms)")
print(f"Avg. dist: {total_dist/len_faces}")
print(f"Avg. sim: {total_sim/len_faces}")

for m in margins:
    print(f"Margin {m.threshold}")
    print(f"\tAccuracy: {m.get_accuracy()}")
    print(f"\tPrecision: {m.get_precision()}")
    print(f"\tRecall: {m.get_recall()}")
    print(f"\tF1 Score: {m.get_f1_score()}")
