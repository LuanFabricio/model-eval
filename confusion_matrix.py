import os

from typing import List

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


class ConfusionMatrix:
    matrix: List[List[int]]
    size: int
    labels: List[int]
    predicted: List[int]

    def __init__(self, size: int):
        self.matrix = [[0] * size for _ in range(size)]
        self.size = size
        self.labels = []
        self.predicted = []

    def add(self, target: int, predicted: int):
        self.matrix[target][predicted] += 1
        self.labels.append(target)
        self.predicted.append(predicted)

    def get_accuracy(self) -> float:
        true_values = 0
        for i in range(self.size):
            true_values += self.matrix[i][i]

        total = 0
        for row in self.matrix:
            total += sum(row)

        return true_values / total

    def show(self):
        for row in self.matrix:
            print(row)

    def log(self, file_path: str, model_name: str):
        base_folder = f"logs/models/{model_name}"

        os.makedirs(base_folder, exist_ok=True)

        cm = confusion_matrix(self.labels, self.predicted)
        # figure = plt.figure()
        plt.matshow(cm)
        plt.title(f"Confusion Matrix - ({self.size}x{self.size})")
        plt.colorbar()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.savefig(
                os.path.join(base_folder, file_path.replace(".log", ".png")))

        equals = "=" * 25
        with open(os.path.join(base_folder, file_path), "w") as file:
            file.write(f"{equals}Confusion matrix ({model_name}){equals}\n")
            file.write("Target x Predicted\n")

            str_matrix = ""
            str_precision = "\n"
            total_precision = 0
            for i, row in enumerate(self.matrix):
                str_matrix += f"{str(row)}\n"
                total = sum(row)
                precision = 0 if total == 0 else row[i] / total
                str_precision += f"[{i}]: {precision}\n"
                total_precision += precision

            file.write(str_matrix)
            file.write(str_precision)

            file.write(f"Precision (avg): {total_precision / self.size}\n")
            file.write(f"\nAccuracy: {self.get_accuracy()}")
