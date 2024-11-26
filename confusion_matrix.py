import os


class ConfusionMatrix:
    def __init__(self, size: int):
        self.matrix = [[0] * size for _ in range(size)]
        self.size = size

    def add(self, target: int, predicted: int):
        self.matrix[target][predicted] += 1

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
        base_folder = f"logs/{model_name}"
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
