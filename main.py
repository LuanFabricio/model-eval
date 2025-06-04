import os
import sys
from math import sqrt

from model import Model
from margin import Margin
from tests import TestModel

current_model = sys.argv[1] if len(sys.argv) > 1 else "mobilefacenet.tflite"
model_path = os.path.join(os.curdir, current_model)

model_name = model_path.split("/")[-1].split(".")[0]
interpreter = Model(model_path, model_name)

if __name__ == "__main__":
    base = 0.000
    scale = 0.0001
    start = 1
    end = 6

    runs = 10
    samples_inf_time = []
    samples_accuracy = []
    for i in range(1, runs+1):
        print(f"Run: {i}")

        def create_margin():
            return Margin.from_range(base, scale, start, end)

        margins = create_margin()
        model_test = TestModel(
            interpreter, margins, faces_base_path="cropped/crop")

        _, _, avg_dt, cm = model_test.test_flipped_faces2(
                i, f"flipped_{i:02}.log", "Flipped 2")

        samples_inf_time.append(avg_dt)
        samples_accuracy.append(cm.get_accuracy())

    inf_time_avg = sum(samples_inf_time) / runs
    inf_time_standard_deviation = sqrt(
            sum(map(lambda x: (x - inf_time_avg)**2, samples_inf_time)) / runs)
    for i in range(1, runs+1):
        print(f"{'='*32} Run: {i:02} {'='*32}")
        print(f"Inference time: {samples_inf_time[i-1]}")
        print(f"Accuracy: {samples_accuracy[i-1]}")
        print(f"{'='*(64+9)}")
    print(f"\nTotal avg: {inf_time_avg*1000}ms (+/-{inf_time_standard_deviation}*1000)")

    accuracy_avg = sum(samples_accuracy) / runs
    accuracy_standard_deviation = sqrt(
            sum(map(lambda x: (x - accuracy_avg)**2, samples_accuracy)) / runs)
    print(f"Accuracy avg: {accuracy_avg*100}% (+/-{accuracy_standard_deviation*100})")
