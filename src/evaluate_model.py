import json

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss, recall_score

from config import FEATURE_COLUMNS, MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, TARGET_COLUMN
from prediction_policy import apply_prediction_policy, load_policy
from train_model import time_based_split


def _probabilities_for_all_classes(model, features: pd.DataFrame) -> pd.DataFrame:
    probabilities = model.predict_proba(features)
    output = pd.DataFrame(0.0, index=features.index, columns=[0, 1, 2])
    for index, class_label in enumerate(model.classes_):
        output[int(class_label)] = probabilities[:, index]
    return output


def _feature_importance(model, feature_columns: list[str]) -> pd.DataFrame:
    classifier = model
    if hasattr(model, "named_steps"):
        classifier = model.named_steps.get("classifier", model)
    elif hasattr(model, "estimator") and hasattr(model.estimator, "named_steps"):
        classifier = model.estimator.named_steps.get("classifier", model.estimator)

    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        values = abs(classifier.coef_).mean(axis=0)
    else:
        values = [0.0] * len(feature_columns)
    return pd.DataFrame({"feature": feature_columns, "importance": values}).sort_values("importance", ascending=False)


def main() -> None:
    data = pd.read_csv(PROCESSED_DATA_DIR / "training_dataset.csv")
    if data.empty:
        raise ValueError("training_dataset.csv is empty. Run the pipeline after adding historical data.")

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    _, _, test_data = time_based_split(data.sort_values("date").reset_index(drop=True))

    metadata = json.loads((MODELS_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    model = joblib.load(MODELS_DIR / "best_model.pkl")
    feature_columns = metadata.get("features", FEATURE_COLUMNS)

    x_test = test_data[feature_columns]
    y_test = test_data[TARGET_COLUMN].astype(int)
    probabilities = _probabilities_for_all_classes(model, x_test)
    predictions = probabilities.idxmax(axis=1)
    policy = load_policy()
    policy_predictions = apply_prediction_policy(probabilities, policy)

    accuracy = accuracy_score(y_test, predictions)
    loss = log_loss(y_test, probabilities[[0, 1, 2]], labels=[0, 1, 2])
    draw_recall = recall_score(y_test, predictions, labels=[1], average="macro", zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1, 2])
    policy_accuracy = accuracy_score(y_test, policy_predictions)
    policy_draw_recall = recall_score(y_test, policy_predictions, labels=[1], average="macro", zero_division=0)
    policy_matrix = confusion_matrix(y_test, policy_predictions, labels=[0, 1, 2])
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2],
        target_names=["Team A loss", "Draw", "Team A win"],
        zero_division=0,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "Best Model Final Test Evaluation",
        "",
        f"Best model: {metadata['best_model']}",
        f"Accuracy: {accuracy:.4f}",
        f"Log loss: {loss:.4f}",
        f"Draw recall: {draw_recall:.4f}",
        f"Number of features used: {len(feature_columns)}",
        f"PyFotMob features included: {metadata.get('fotmob_features_included', [])}",
        "",
        "Confusion matrix rows=true, columns=predicted, order=[Team A loss, Draw, Team A win]:",
        str(matrix.tolist()),
        "",
        "Decision policy:",
        json.dumps(policy, indent=2),
        f"Policy accuracy: {policy_accuracy:.4f}",
        f"Policy draw recall: {policy_draw_recall:.4f}",
        "Policy confusion matrix rows=true, columns=predicted, order=[Team A loss, Draw, Team A win]:",
        str(policy_matrix.tolist()),
        "",
        "Classification report:",
        report,
    ]
    report_path = REPORTS_DIR / "evaluation_report.txt"
    if report_path.exists():
        existing = report_path.read_text(encoding="utf-8").rstrip()
        report_path.write_text(existing + "\n\n" + "\n".join(lines), encoding="utf-8")
    else:
        report_path.write_text("\n".join(lines), encoding="utf-8")
    _feature_importance(model, feature_columns).to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    print(f"Evaluation saved: {REPORTS_DIR / 'evaluation_report.txt'}")


if __name__ == "__main__":
    main()
