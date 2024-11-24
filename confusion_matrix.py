import os


class ConfusionMatrix:
    def __init__(self, size: int):
        self.matrix = [[0] * size for _ in range(size)]
        self.size = size

    def add(self, target: int, predicted: int):
        self.matrix[target][predicted] += 1

    def show(self):
        for row in self.matrix:
            print(row)

    def log(self, file_path: str, model_name: str):
        base_folder = f"logs/{model_name}"
        equals = "=" * 25
        with open(os.path.join(base_folder, file_path), "w") as file:
            file.write(f"{equals}Confusion matrix ({model_name}){equals}\n")
            file.write("Target x Predicted\n")
            for row in self.matrix:
                file.write(str(row))
