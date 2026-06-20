import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

from config import MODELS_DIR


DEFAULT_POLICY = {
    "name": "argmax",
    "draw_min_probability": 1.0,
    "draw_margin": 0.0,
    "description": "Choose the class with the highest model probability.",
}


def policy_path() -> Path:
    return MODELS_DIR / "prediction_policy.json"


def apply_prediction_policy(probabilities: pd.DataFrame, policy: dict | None = None) -> pd.Series:
    policy = policy or DEFAULT_POLICY
    draw_min_probability = float(policy.get("draw_min_probability", 1.0))
    draw_margin = float(policy.get("draw_margin", 0.0))

    labels = []
    for _, row in probabilities[[0, 1, 2]].iterrows():
        strongest_non_draw = max(float(row[0]), float(row[2]))
        if float(row[1]) >= draw_min_probability and strongest_non_draw - float(row[1]) <= draw_margin:
            labels.append(1)
        else:
            labels.append(int(row.idxmax()))
    return pd.Series(labels, index=probabilities.index)


def evaluate_policy(probabilities: pd.DataFrame, y_true: pd.Series, policy: dict) -> dict:
    predictions = apply_prediction_policy(probabilities, policy)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "draw_recall": float(recall_score(y_true, predictions, labels=[1], average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1, 2]).tolist(),
    }


def tune_draw_policy(probabilities: pd.DataFrame, y_true: pd.Series) -> dict:
    baseline = evaluate_policy(probabilities, y_true, DEFAULT_POLICY)
    best_policy = {
        **DEFAULT_POLICY,
        "validation_accuracy": baseline["accuracy"],
        "validation_draw_recall": baseline["draw_recall"],
        "validation_confusion_matrix": baseline["confusion_matrix"],
    }
    best_score = baseline["accuracy"] + 0.02 * baseline["draw_recall"]

    for threshold in [round(value / 100, 2) for value in range(18, 36)]:
        for margin in [round(value / 100, 2) for value in range(0, 16)]:
            policy = {
                "name": "draw_threshold",
                "draw_min_probability": threshold,
                "draw_margin": margin,
                "description": "Predict draw when draw probability is high and close to the strongest win/loss probability.",
            }
            metrics = evaluate_policy(probabilities, y_true, policy)
            accuracy_drop = baseline["accuracy"] - metrics["accuracy"]
            if accuracy_drop > 0.015:
                continue

            score = metrics["accuracy"] + 0.02 * metrics["draw_recall"]
            if score > best_score:
                best_score = score
                best_policy = {
                    **policy,
                    "validation_accuracy": metrics["accuracy"],
                    "validation_draw_recall": metrics["draw_recall"],
                    "validation_confusion_matrix": metrics["confusion_matrix"],
                    "baseline_validation_accuracy": baseline["accuracy"],
                    "baseline_validation_draw_recall": baseline["draw_recall"],
                }

    return best_policy


def save_policy(policy: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    policy_path().write_text(json.dumps(policy, indent=2), encoding="utf-8")


def load_policy() -> dict:
    path = policy_path()
    if not path.exists():
        return DEFAULT_POLICY
    return json.loads(path.read_text(encoding="utf-8"))
