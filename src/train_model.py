import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import FEATURE_COLUMNS, MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_STATE, REPORTS_DIR, TARGET_COLUMN


MODEL_SPECS = {
    "logistic_regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ]),
    "random_forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1)),
    ]),
    "xgboost": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
        )),
    ]),
}


def time_based_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sort_columns = ["date"]
    if "match_id" in data.columns:
        sort_columns.append("match_id")
    data = data.sort_values(sort_columns).reset_index(drop=True)
    train_end = int(len(data) * 0.70)
    validation_end = int(len(data) * 0.85)
    return data.iloc[:train_end], data.iloc[train_end:validation_end], data.iloc[validation_end:]


def _probabilities_for_all_classes(model, features: pd.DataFrame) -> pd.DataFrame:
    probabilities = model.predict_proba(features)
    classes = list(model.classes_)
    output = pd.DataFrame(0.0, index=features.index, columns=[0, 1, 2])
    for index, class_label in enumerate(classes):
        output[int(class_label)] = probabilities[:, index]
    return output


def evaluate_split(model, split: pd.DataFrame) -> dict:
    x = split[FEATURE_COLUMNS]
    y = split[TARGET_COLUMN].astype(int)
    probabilities = _probabilities_for_all_classes(model, x)
    predictions = probabilities.idxmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "log_loss": float(log_loss(y, probabilities[[0, 1, 2]], labels=[0, 1, 2])),
        "confusion_matrix": confusion_matrix(y, predictions, labels=[0, 1, 2]).tolist(),
    }


def train() -> dict:
    data_path = PROCESSED_DATA_DIR / "training_dataset.csv"
    data = pd.read_csv(data_path)
    if data.empty:
        raise ValueError("training_dataset.csv is empty. Add historical matches before training.")

    missing_columns = [column for column in [TARGET_COLUMN, "date", *FEATURE_COLUMNS] if column not in data.columns]
    if missing_columns:
        raise ValueError(f"training_dataset.csv is missing columns: {missing_columns}")

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date", TARGET_COLUMN])
    if len(data) < 30:
        raise ValueError("Need at least 30 historical matches for a useful time-based split.")

    train_data, validation_data, test_data = time_based_split(data)
    if train_data[TARGET_COLUMN].nunique() < 3:
        raise ValueError("The training split must include wins, draws, and losses. Add more historical data.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    best_name = None
    best_validation_log_loss = float("inf")

    for name, model in MODEL_SPECS.items():
        model.fit(train_data[FEATURE_COLUMNS], train_data[TARGET_COLUMN].astype(int))
        joblib.dump(model, MODELS_DIR / f"{name}.pkl")

        validation_metrics = evaluate_split(model, validation_data)
        test_metrics = evaluate_split(model, test_data)
        results[name] = {
            "validation": validation_metrics,
            "test": test_metrics,
        }

        if validation_metrics["log_loss"] < best_validation_log_loss:
            best_validation_log_loss = validation_metrics["log_loss"]
            best_name = name

    best_model_path = MODELS_DIR / f"{best_name}.pkl"
    joblib.dump(joblib.load(best_model_path), MODELS_DIR / "best_model.pkl")

    metadata = {
        "best_model": best_name,
        "best_model_path": str(Path("models") / "best_model.pkl"),
        "selection_metric": "validation_log_loss",
        "features": FEATURE_COLUMNS,
        "target": "0 = Team A loss, 1 = Draw, 2 = Team A win",
        "split": {
            "type": "time_based",
            "train_rows": len(train_data),
            "validation_rows": len(validation_data),
            "test_rows": len(test_data),
            "train_date_range": [str(train_data["date"].min().date()), str(train_data["date"].max().date())],
            "validation_date_range": [str(validation_data["date"].min().date()), str(validation_data["date"].max().date())],
            "test_date_range": [str(test_data["date"].min().date()), str(test_data["date"].max().date())],
        },
        "models": results,
    }
    (MODELS_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def write_training_report(metadata: dict) -> None:
    lines = [
        "Phase 1 Historical Match Prediction Report",
        "",
        f"Best model: {metadata['best_model']}",
        f"Selection metric: {metadata['selection_metric']}",
        "",
        "Features used:",
        *[f"- {feature}" for feature in metadata["features"]],
        "",
        "Time-based split:",
        json.dumps(metadata["split"], indent=2),
        "",
        "Model metrics:",
    ]
    for name, metrics in metadata["models"].items():
        lines.extend([
            "",
            name,
            f"Validation accuracy: {metrics['validation']['accuracy']:.4f}",
            f"Validation log loss: {metrics['validation']['log_loss']:.4f}",
            f"Validation confusion matrix [[loss, draw, win], ...]: {metrics['validation']['confusion_matrix']}",
            f"Test accuracy: {metrics['test']['accuracy']:.4f}",
            f"Test log loss: {metrics['test']['log_loss']:.4f}",
            f"Test confusion matrix [[loss, draw, win], ...]: {metrics['test']['confusion_matrix']}",
        ])
    (REPORTS_DIR / "evaluation_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    metadata = train()
    write_training_report(metadata)
    print(f"Models trained. Best model: {metadata['best_model']}")


if __name__ == "__main__":
    main()
