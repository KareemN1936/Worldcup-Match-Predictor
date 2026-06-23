import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from config import FEATURE_COLUMNS, FOTMOB_DIFF_FEATURE_COLUMNS, MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_STATE, RAW_DATA_DIR, REPORTS_DIR, TARGET_COLUMN
from model_ensemble import ProbabilityEnsemble
from prediction_policy import save_policy, tune_draw_policy


def _logistic_regression(class_weight=None) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=RANDOM_STATE)),
    ])


def _random_forest(class_weight=None) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def _xgboost(**overrides) -> Pipeline:
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "n_estimators": 250,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "eval_metric": "mlogloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    params.update(overrides)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", XGBClassifier(**params)),
    ])


MODEL_SPECS = {
    "logistic_regression_baseline": {"factory": lambda: _logistic_regression(), "sample_weight": False, "type": "baseline"},
    "logistic_regression_balanced": {"factory": lambda: _logistic_regression(class_weight="balanced"), "sample_weight": False, "type": "balanced"},
    "random_forest_baseline": {"factory": lambda: _random_forest(), "sample_weight": False, "type": "baseline"},
    "random_forest_balanced": {"factory": lambda: _random_forest(class_weight="balanced"), "sample_weight": False, "type": "balanced"},
    "xgboost_baseline": {"factory": lambda: _xgboost(), "sample_weight": False, "type": "baseline"},
    "xgboost_balanced": {"factory": lambda: _xgboost(), "sample_weight": True, "type": "balanced"},
    "xgboost_regularized": {
        "factory": lambda: _xgboost(n_estimators=450, learning_rate=0.035, max_depth=2, min_child_weight=4, reg_lambda=4.0, reg_alpha=0.1),
        "sample_weight": False,
        "type": "tuned_baseline",
    },
    "xgboost_regularized_balanced": {
        "factory": lambda: _xgboost(n_estimators=450, learning_rate=0.035, max_depth=2, min_child_weight=4, reg_lambda=4.0, reg_alpha=0.1),
        "sample_weight": True,
        "type": "tuned_balanced",
    },
    "xgboost_slow_learning": {
        "factory": lambda: _xgboost(n_estimators=550, learning_rate=0.025, max_depth=3, subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0),
        "sample_weight": False,
        "type": "tuned_baseline",
    },
}


def time_based_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sort_columns = ["date"]
    if "match_id" in data.columns:
        sort_columns.append("match_id")
    data = data.sort_values(sort_columns).reset_index(drop=True)
    train_end = int(len(data) * 0.70)
    validation_end = int(len(data) * 0.85)
    return data.iloc[:train_end], data.iloc[train_end:validation_end], data.iloc[validation_end:]


def calibration_split(train_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_data = train_data.sort_values("date").reset_index(drop=True)
    calibration_start = int(len(train_data) * 0.85)
    base_train_data = train_data.iloc[:calibration_start]
    calibration_data = train_data.iloc[calibration_start:]
    return base_train_data, calibration_data


def _probabilities_for_all_classes(model, features: pd.DataFrame) -> pd.DataFrame:
    probabilities = model.predict_proba(features)
    output = pd.DataFrame(0.0, index=features.index, columns=[0, 1, 2])
    for index, class_label in enumerate(model.classes_):
        output[int(class_label)] = probabilities[:, index]
    return output


def select_feature_columns(data: pd.DataFrame) -> list[str]:
    selected = [column for column in FEATURE_COLUMNS if column in data.columns]
    for column in FOTMOB_DIFF_FEATURE_COLUMNS:
        if column in data.columns:
            non_null = data[column].notna().sum()
            if non_null >= max(30, int(len(data) * 0.1)):
                selected.append(column)
    return selected


def evaluate_split(model, split: pd.DataFrame, feature_columns: list[str]) -> dict:
    x = split[feature_columns]
    y = split[TARGET_COLUMN].astype(int)
    probabilities = _probabilities_for_all_classes(model, x)
    predictions = probabilities.idxmax(axis=1)
    matrix = confusion_matrix(y, predictions, labels=[0, 1, 2])
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "log_loss": float(log_loss(y, probabilities[[0, 1, 2]], labels=[0, 1, 2])),
        "draw_recall": float(recall_score(y, predictions, labels=[1], average="macro", zero_division=0)),
        "confusion_matrix": matrix.tolist(),
    }


def evaluate_time_windows(model, split: pd.DataFrame, feature_columns: list[str], windows: int = 3) -> dict:
    ordered = split.sort_values("date").reset_index(drop=True)
    window_metrics = [
        evaluate_split(model, ordered.iloc[indexes], feature_columns)
        for indexes in np.array_split(np.arange(len(ordered)), windows)
        if len(indexes)
    ]
    losses = [item["log_loss"] for item in window_metrics]
    return {
        "windows": window_metrics,
        "mean_log_loss": float(np.mean(losses)),
        "log_loss_std": float(np.std(losses)),
        "selection_score": float(np.mean(losses) + 0.10 * np.std(losses)),
    }


def recency_weights(data: pd.DataFrame, half_life_years: float = 8.0) -> pd.Series:
    reference_date = data["date"].max()
    age_years = (reference_date - data["date"]).dt.days.clip(lower=0) / 365.25
    weights = 0.5 ** (age_years / half_life_years)
    weights = weights.clip(lower=0.05)
    return weights / weights.mean()


def _fit_model(
    model,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: pd.Series,
    balance_classes: bool = False,
):
    weights = sample_weight.to_numpy(dtype=float)
    if balance_classes:
        weights = weights * compute_sample_weight(class_weight="balanced", y=y_train)

    model.fit(x_train, y_train, classifier__sample_weight=weights)
    return model


def _calibrate_model(base_model, x_calibration: pd.DataFrame, y_calibration: pd.Series):
    calibrated = CalibratedClassifierCV(FrozenEstimator(base_model), method="sigmoid")
    calibrated.fit(x_calibration, y_calibration)
    return calibrated


def _safe_model_filename(name: str) -> str:
    return f"{name}.pkl"


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
    feature_columns = select_feature_columns(data)
    if len(data) < 30:
        raise ValueError("Need at least 30 historical matches for a useful time-based split.")

    train_data, validation_data, test_data = time_based_split(data)
    if train_data[TARGET_COLUMN].nunique() < 3:
        raise ValueError("The training split must include wins, draws, and losses. Add more historical data.")

    base_train_data, calibration_data = calibration_split(train_data)
    if base_train_data[TARGET_COLUMN].nunique() < 3 or calibration_data[TARGET_COLUMN].nunique() < 3:
        raise ValueError("The train/calibration split must include wins, draws, and losses. Add more historical data.")

    x_full_train = train_data[feature_columns]
    y_full_train = train_data[TARGET_COLUMN].astype(int)
    x_base_train = base_train_data[feature_columns]
    y_base_train = base_train_data[TARGET_COLUMN].astype(int)
    x_calibration = calibration_data[feature_columns]
    y_calibration = calibration_data[TARGET_COLUMN].astype(int)
    x_validation = validation_data[feature_columns]
    y_validation = validation_data[TARGET_COLUMN].astype(int)
    full_train_weights = recency_weights(train_data)
    base_train_weights = recency_weights(base_train_data)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    best_name = None
    best_validation_log_loss = float("inf")

    for name, spec in MODEL_SPECS.items():
        model = _fit_model(
            spec["factory"](),
            x_full_train,
            y_full_train,
            full_train_weights,
            balance_classes=spec["sample_weight"],
        )
        model_path = MODELS_DIR / _safe_model_filename(name)
        joblib.dump(model, model_path)

        validation_metrics = evaluate_split(model, validation_data, feature_columns)
        test_metrics = evaluate_split(model, test_data, feature_columns)
        rolling_metrics = evaluate_time_windows(model, validation_data, feature_columns)
        results[name] = {
            "model_path": str(model_path.relative_to(MODELS_DIR.parent)),
            "type": spec["type"],
            "validation": validation_metrics,
            "test": test_metrics,
            "rolling_validation": rolling_metrics,
        }

        if rolling_metrics["selection_score"] < best_validation_log_loss:
            best_validation_log_loss = rolling_metrics["selection_score"]
            best_name = name

        calibrated_name = f"{name}_calibrated"
        calibration_base_model = _fit_model(
            spec["factory"](),
            x_base_train,
            y_base_train,
            base_train_weights,
            balance_classes=spec["sample_weight"],
        )
        calibrated_model = _calibrate_model(calibration_base_model, x_calibration, y_calibration)
        calibrated_path = MODELS_DIR / _safe_model_filename(calibrated_name)
        joblib.dump(calibrated_model, calibrated_path)

        calibrated_validation_metrics = evaluate_split(calibrated_model, validation_data, feature_columns)
        calibrated_test_metrics = evaluate_split(calibrated_model, test_data, feature_columns)
        calibrated_rolling_metrics = evaluate_time_windows(calibrated_model, validation_data, feature_columns)
        results[calibrated_name] = {
            "model_path": str(calibrated_path.relative_to(MODELS_DIR.parent)),
            "type": "calibrated_sigmoid_on_calibration_split",
            "validation": calibrated_validation_metrics,
            "test": calibrated_test_metrics,
            "rolling_validation": calibrated_rolling_metrics,
        }

        if calibrated_rolling_metrics["selection_score"] < best_validation_log_loss:
            best_validation_log_loss = calibrated_rolling_metrics["selection_score"]
            best_name = calibrated_name

    top_names = sorted(
        results,
        key=lambda model_name: results[model_name]["rolling_validation"]["selection_score"],
    )[:3]
    top_scores = np.asarray(
        [results[model_name]["rolling_validation"]["selection_score"] for model_name in top_names],
        dtype=float,
    )
    ensemble_weights = (1.0 / top_scores).tolist()
    ensemble = ProbabilityEnsemble(
        [joblib.load(MODELS_DIR / _safe_model_filename(model_name)) for model_name in top_names],
        ensemble_weights,
    )
    ensemble_name = "probability_ensemble"
    ensemble_path = MODELS_DIR / _safe_model_filename(ensemble_name)
    joblib.dump(ensemble, ensemble_path)
    ensemble_validation_metrics = evaluate_split(ensemble, validation_data, feature_columns)
    ensemble_test_metrics = evaluate_split(ensemble, test_data, feature_columns)
    ensemble_rolling_metrics = evaluate_time_windows(ensemble, validation_data, feature_columns)
    results[ensemble_name] = {
        "model_path": str(ensemble_path.relative_to(MODELS_DIR.parent)),
        "type": "soft_voting_recent_log_loss_weighted",
        "members": top_names,
        "weights": (np.asarray(ensemble_weights) / np.sum(ensemble_weights)).tolist(),
        "validation": ensemble_validation_metrics,
        "test": ensemble_test_metrics,
        "rolling_validation": ensemble_rolling_metrics,
    }
    if ensemble_rolling_metrics["selection_score"] < best_validation_log_loss:
        best_validation_log_loss = ensemble_rolling_metrics["selection_score"]
        best_name = ensemble_name

    best_model_path = MODELS_DIR / _safe_model_filename(best_name)
    best_model = joblib.load(best_model_path)
    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
    validation_probabilities = _probabilities_for_all_classes(best_model, x_validation)
    prediction_policy = tune_draw_policy(validation_probabilities, y_validation)
    save_policy(prediction_policy)

    metadata = {
        "best_model": best_name,
        "best_model_path": str(Path("models") / "best_model.pkl"),
        "prediction_policy_path": str(Path("models") / "prediction_policy.json"),
        "selection_metric": "rolling_validation_log_loss_with_stability_penalty",
        "recency_weighting": {"method": "exponential_half_life", "half_life_years": 8.0, "minimum_weight": 0.05},
        "features": feature_columns,
        "fotmob_features_included": [column for column in feature_columns if column in FOTMOB_DIFF_FEATURE_COLUMNS],
        "target": "0 = Team A loss, 1 = Draw, 2 = Team A win",
        "split": {
            "type": "time_based",
            "base_train_rows": len(base_train_data),
            "calibration_rows": len(calibration_data),
            "validation_rows": len(validation_data),
            "test_rows": len(test_data),
            "base_train_date_range": [str(base_train_data["date"].min().date()), str(base_train_data["date"].max().date())],
            "calibration_date_range": [str(calibration_data["date"].min().date()), str(calibration_data["date"].max().date())],
            "validation_date_range": [str(validation_data["date"].min().date()), str(validation_data["date"].max().date())],
            "test_date_range": [str(test_data["date"].min().date()), str(test_data["date"].max().date())],
        },
        "prediction_policy": prediction_policy,
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
        "Prediction policy:",
        json.dumps(metadata["prediction_policy"], indent=2),
        "",
        "Model comparison:",
    ]
    for name, metrics in metadata["models"].items():
        lines.extend([
            "",
            name,
            f"Type: {metrics['type']}",
            f"Validation accuracy: {metrics['validation']['accuracy']:.4f}",
            f"Validation log loss: {metrics['validation']['log_loss']:.4f}",
            f"Validation draw recall: {metrics['validation']['draw_recall']:.4f}",
            f"Validation confusion matrix [[loss, draw, win], ...]: {metrics['validation']['confusion_matrix']}",
            f"Test accuracy: {metrics['test']['accuracy']:.4f}",
            f"Test log loss: {metrics['test']['log_loss']:.4f}",
            f"Test draw recall: {metrics['test']['draw_recall']:.4f}",
            f"Test confusion matrix [[loss, draw, win], ...]: {metrics['test']['confusion_matrix']}",
        ])
    (REPORTS_DIR / "evaluation_report.txt").write_text("\n".join(lines), encoding="utf-8")

    rows = []
    for name, metrics in metadata["models"].items():
        rows.append({
            "model": name,
            "type": metrics["type"],
            "validation_accuracy": metrics["validation"]["accuracy"],
            "validation_log_loss": metrics["validation"]["log_loss"],
            "validation_draw_recall": metrics["validation"]["draw_recall"],
            "test_accuracy": metrics["test"]["accuracy"],
            "test_log_loss": metrics["test"]["log_loss"],
            "test_draw_recall": metrics["test"]["draw_recall"],
        })
    pd.DataFrame(rows).sort_values("validation_log_loss").to_csv(REPORTS_DIR / "model_comparison.csv", index=False)


def refit_production_model() -> dict:
    """Refit the selected architecture on all data available before the tournament."""
    metadata_path = MODELS_DIR / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    selected_name = metadata["best_model"]
    if selected_name not in MODEL_SPECS:
        members = metadata["models"].get(selected_name, {}).get("members", [])
        selected_name = members[0] if members else "xgboost_slow_learning"
    if selected_name.endswith("_calibrated"):
        selected_name = selected_name.removesuffix("_calibrated")

    data = pd.read_csv(PROCESSED_DATA_DIR / "training_dataset.csv")
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    fixtures = pd.read_csv(RAW_DATA_DIR / "fixtures.csv")
    tournament_start = pd.to_datetime(fixtures["date"], errors="coerce", utc=True).min().tz_localize(None)
    production_data = data[data["date"] < tournament_start].dropna(subset=["date", TARGET_COLUMN]).copy()
    feature_columns = metadata["features"]
    selection_point = int(len(production_data) * 0.85)
    development = production_data.iloc[:selection_point]
    pre_tournament_holdout = production_data.iloc[selection_point:]
    calibration_point = int(len(development) * 0.85)
    base_development = development.iloc[:calibration_point]
    calibration_development = development.iloc[calibration_point:]

    holdout_base = _fit_model(
        MODEL_SPECS[selected_name]["factory"](),
        development[feature_columns], development[TARGET_COLUMN].astype(int), recency_weights(development),
        balance_classes=MODEL_SPECS[selected_name]["sample_weight"],
    )
    calibration_base = _fit_model(
        MODEL_SPECS[selected_name]["factory"](),
        base_development[feature_columns], base_development[TARGET_COLUMN].astype(int), recency_weights(base_development),
        balance_classes=MODEL_SPECS[selected_name]["sample_weight"],
    )
    holdout_calibrated = _calibrate_model(
        calibration_base,
        calibration_development[feature_columns],
        calibration_development[TARGET_COLUMN].astype(int),
    )
    base_loss = evaluate_split(holdout_base, pre_tournament_holdout, feature_columns)["log_loss"]
    calibrated_loss = evaluate_split(holdout_calibrated, pre_tournament_holdout, feature_columns)["log_loss"]

    if calibrated_loss < base_loss:
        final_calibration_point = int(len(production_data) * 0.90)
        final_base_data = production_data.iloc[:final_calibration_point]
        final_calibration_data = production_data.iloc[final_calibration_point:]
        final_base = _fit_model(
            MODEL_SPECS[selected_name]["factory"](),
            final_base_data[feature_columns], final_base_data[TARGET_COLUMN].astype(int), recency_weights(final_base_data),
            balance_classes=MODEL_SPECS[selected_name]["sample_weight"],
        )
        model = _calibrate_model(
            final_base,
            final_calibration_data[feature_columns],
            final_calibration_data[TARGET_COLUMN].astype(int),
        )
        production_variant = "recent_sigmoid_calibrated"
    else:
        model = _fit_model(
            MODEL_SPECS[selected_name]["factory"](),
            production_data[feature_columns], production_data[TARGET_COLUMN].astype(int), recency_weights(production_data),
            balance_classes=MODEL_SPECS[selected_name]["sample_weight"],
        )
        production_variant = "uncalibrated"
    joblib.dump(model, MODELS_DIR / "best_model.pkl")
    metadata["production_model"] = {
        "architecture": selected_name,
        "training_rows": len(production_data),
        "training_date_range": [
            str(production_data["date"].min().date()),
            str(production_data["date"].max().date()),
        ],
        "strict_cutoff": tournament_start.isoformat(),
        "uses_tournament_results": False,
        "variant": production_variant,
        "pre_tournament_holdout_log_loss": {"uncalibrated": base_loss, "calibrated": calibrated_loss},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refit-production", action="store_true")
    args = parser.parse_args()
    if args.refit_production:
        metadata = refit_production_model()
        print(f"Production model refit: {metadata['production_model']}")
        return
    metadata = train()
    write_training_report(metadata)
    print(f"Models trained. Best model: {metadata['best_model']}")


if __name__ == "__main__":
    main()
