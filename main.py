import os
import sys

from confusion_matrix import ConfusionMatrix
from model import Model
# load_model_interpreter, get_embeddings
from margin import Margin
from tests import TestModel

current_model = sys.argv[1] if len(sys.argv) > 1 else "mobilefacenet.tflite"
model_path = os.path.join(os.curdir, current_model)

# model_path = "/home/luan/dev/tcc-eval/mobilefacenet.tflite"
# model_path = "/home/luan/dev/tcc-eval/rafael_student.tflite"
# model_path = "/home/luan/dev/tcc-eval/triplet_dist_student.tflite"
# model_path = "/home/luan/dev/tcc-eval/pair_model.tflite"

# interpreter = load_model_interpreter(model_path)
model_name = model_path.split("/")[-1].split(".")[0]
interpreter = Model(model_path, model_name)

# print(interpreter.get_input_details())
# print(interpreter.get_output_details())

if __name__ == "__main__":
    base = 0.000
    scale = 0.0001
    start = 1
    end = 6

    def create_margin():
        return Margin.from_range(base, scale, start, end)

    margins = create_margin()
    model_test = TestModel(interpreter, margins, faces_base_path="faces")

    cm = ConfusionMatrix(2)
    model_test.test_flipped_faces2("flipped.log", "Flipped 2")
    # model_test.test_flipped_faces(margins, "flipped.log", "Flipped", cm)
    # model_test.test_diff(margins, "flipped+diff.log", "Flipped and Diff")
    # model_test.test_diff(create_margin(), "diff.log", "Diff")
    # model_test.triplet_test(create_margin(), "triplet.log", "Triplet")
