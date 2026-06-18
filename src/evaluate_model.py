import json

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss

from config import FEATURE_COLUMNS, MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, TARGET_COLUMN
from train_model import time_based_split


def _probabilities_for_all_classes(model, features: pd.DataFrame) -> pd.DataFrame:
    probabilities = model.predict_proba(features)
    output = pd.DataFrame(0.0, index=features.index, columns=[0, 1, 2])
    for index, class_label in enumerate(model.classes_):
        output[int(class_label)] = probabilities[:, index]
    return output


def _feature_importance(model) -> pd.DataFrame:
    classifier = model.named_steps.get("classifier", model)
    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        values = abs(classifier.coef_).mean(axis=0)
    else:
        values = [0.0] * len(FEATURE_COLUMNS)
    return pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": values}).sort_values("importance", ascending=False)


def main() -> None:
    data = pd.read_csv(PROCESSED_DATA_DIR / "training_dataset.csv")
    if data.empty:
        raise ValueError("training_dataset.csv is empty. Run the pipeline after adding historical data.")

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    _, _, test_data = time_based_split(data.sort_values("date").reset_index(drop=True))

    metadata = json.loads((MODELS_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    model = joblib.load(MODELS_DIR / "best_model.pkl")

    x_test = test_data[FEATURE_COLUMNS]
    y_test = test_data[TARGET_COLUMN].astype(int)
    probabilities = _probabilities_for_all_classes(model, x_test)
    predictions = probabilities.idxmax(axis=1)

    accuracy = accuracy_score(y_test, predictions)
    loss = log_loss(y_test, probabilities[[0, 1, 2]], labels=[0, 1, 2])
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1, 2])
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2],
        target_names=["Team A loss", "Draw", "Team A win"],
        zero_division=0,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "Best Model Test Evaluation",
        "",
        f"Best model: {metadata['best_model']}",
        f"Accuracy: {accuracy:.4f}",
        f"Log loss: {loss:.4f}",
        "",
        "Confusion matrix rows=true, columns=predicted, order=[Team A loss, Draw, Team A win]:",
        str(matrix.tolist()),
        "",
        "Classification report:",
        report,
    ]
    (REPORTS_DIR / "evaluation_report.txt").write_text("\n".join(lines), encoding="utf-8")
    _feature_importance(model).to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    print(f"Evaluation saved: {REPORTS_DIR / 'evaluation_report.txt'}")


if __name__ == "__main__":
    main()
