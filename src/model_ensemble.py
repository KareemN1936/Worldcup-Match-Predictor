import numpy as np


class ProbabilityEnsemble:
    """A fitted soft-voting ensemble that preserves three-class probability order."""

    def __init__(self, models: list, weights: list[float] | None = None):
        self.models = models
        raw_weights = np.asarray(weights if weights is not None else [1.0] * len(models), dtype=float)
        self.weights = raw_weights / raw_weights.sum()
        self.classes_ = np.asarray([0, 1, 2])

    def predict_proba(self, features):
        combined = np.zeros((len(features), 3), dtype=float)
        for model, weight in zip(self.models, self.weights):
            raw = model.predict_proba(features)
            aligned = np.zeros_like(combined)
            for source_index, class_label in enumerate(model.classes_):
                aligned[:, int(class_label)] = raw[:, source_index]
            combined += weight * aligned
        return combined / combined.sum(axis=1, keepdims=True)

    def predict(self, features):
        return self.classes_[self.predict_proba(features).argmax(axis=1)]
