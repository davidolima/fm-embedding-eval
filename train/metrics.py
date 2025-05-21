import torch

class Metric:
    def __init__(self, name: str):
        self.name = name
        self.value = None

    def update(self, value: float):
        self.value = value

    def reset(self):
        self.value = None

    def compute(self) -> float:
        return self.value

    def __str__(self) -> str:
        if not self.value:
            self.compute()
        return f"{self.name}: {self.value:.4f}"

class Accuracy(Metric):
    def __init__(self):
        super().__init__("Accuracy")
        self.correct = 0
        self.total = 0

    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        self.correct += (preds == labels).sum().item()
        self.total += labels.size(0)

    def compute(self) -> float:
        self.value = (self.correct / self.total) if self.total > 0 else 0.0
        return self.value

    def reset(self):
        super().reset()
        self.correct = 0
        self.total = 0

class Loss(Metric):
    def __init__(self):
        super().__init__("Loss")
        self.total_loss = 0.0
        self.count = 0

    def update(self, loss: float):
        self.total_loss += loss
        self.count += 1

    def compute(self) -> float:
        self.value = (self.total_loss / self.count) if self.count > 0 else 0.0
        return self.value

    def reset(self):
        super().reset()
        self.total_loss = 0.0
        self.count = 0

class Precision(Metric):
    def __init__(self):
        super().__init__("Precision")
        self.true_positive = 0
        self.false_positive = 0

    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        self.true_positive += ((preds == 1) & (labels == 1)).sum().item()
        self.false_positive += ((preds == 1) & (labels == 0)).sum().item()

    def compute(self) -> float:
        self.value = (self.true_positive / (self.true_positive + self.false_positive)) if (self.true_positive + self.false_positive) > 0 else 0.0
        return self.value

    def reset(self):
        super().reset()
        self.true_positive = 0
        self.false_positive = 0

class Recall(Metric):
    def __init__(self):
        super().__init__("Recall")
        self.true_positive = 0
        self.false_negative = 0

    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        self.true_positive += ((preds == 1) & (labels == 1)).sum().item()
        self.false_negative += ((preds == 0) & (labels == 1)).sum().item()

    def compute(self) -> float:
        self.value = (self.true_positive / (self.true_positive + self.false_negative)) if (self.true_positive + self.false_negative) > 0 else 0.0
        return self.value

    def reset(self):
        super().reset()
        self.true_positive = 0
        self.false_negative = 0

class F1Score(Metric):
    def __init__(self):
        super().__init__("F1 Score")
        self.precision = Precision()
        self.recall = Recall()

    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        self.precision.update(preds, labels)
        self.recall.update(preds, labels)

    def compute(self) -> float:
        precision = self.precision.compute()
        recall = self.recall.compute()
        self.value = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return self.value

    def reset(self):
        super().reset()
        self.precision.reset()
        self.recall.reset()
