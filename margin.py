class Margin:
    threshold: float

    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

    def __init__(self, threshold: float):
        self.threshold = threshold

        self.true_positive = 0
        self.true_negative = 0
        self.false_positive = 0
        self.false_negative = 0

    @staticmethod
    def from_range(base: float, scale: float, start: int = 1, end: int = 11):
        return [Margin(base + i * scale) for i in range(start, end)]

    def evaluate(self, distance: float) -> bool:
        return distance < self.threshold

    def evaluate_with_metrics(self, distance: float, is_true: bool = True):
        result = self.evaluate(distance)

        if result:
            if is_true:
                self.true_positive += 1
            else:
                self.false_positive += 1
        else:
            if is_true:
                self.false_negative += 1
            else:
                self.true_negative += 1

        return result

    def get_precision(self) -> float:
        total_positive = self.true_positive + self.false_positive
        if total_positive == 0:
            return 0

        return self.true_positive / total_positive

    def get_recall(self) -> float:
        total_positive = self.true_positive + self.false_negative
        if total_positive == 0:
            return 0

        return self.true_positive / total_positive

    def get_f1_score(self) -> float:
        recall = self.get_recall()
        precision = self.get_precision()

        recall_precision = recall + precision
        if recall_precision == 0:
            return 0

        return 2 * precision * recall / (precision + recall)

    def get_accuracy(self) -> float:
        positive = self.true_positive + self.true_negative
        total = positive + self.false_negative + self.false_positive
        return positive / total
