"""
Indian Food Delivery ETA Prediction - Model Training Pipeline

This script:
1. Loads the engineered training and held-out test datasets.
2. Creates an internal validation split from the training data.
3. Compares:
   - Random Forest
   - XGBoost
   - Mean baseline
   - Median baseline
   - Distance-only linear regression baseline
4. Selects the best ML model using validation MAE.
5. Retrains the selected ML model on all training data.
6. Evaluates once on the untouched held-out test set.
7. Calculates regression and delay-risk metrics.
8. Logs experiments and metrics to MLflow.
9. Saves the final model and feature list.
10. Generates a model comparison report.

IMPORTANT:
- The held-out test set is NOT used for model selection.
- Only Random Forest and XGBoost are eligible for final model selection.
- Baselines are comparison references only.
"""

from pathlib import Path
import json
import sys

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import evaluate_predictions


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "train_features.csv"
TEST_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "test_features.csv"

MODEL_DIR = PROJECT_ROOT / "models" / "artifacts"
MODEL_PATH = MODEL_DIR / "best_eta_model.joblib"
FEATURE_LIST_PATH = MODEL_DIR / "feature_columns.json"

REPORT_PATH = PROJECT_ROOT / "docs" / "model_comparison.md"

TARGET_COLUMN = "Time_taken_min"

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB}"

EXPERIMENT_NAME = "indian-food-delivery-eta"

# Only these models are allowed to win model selection.
ML_MODELS = {
    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),

    "XGBoost": XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    ),
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_separator():
    print("-" * 70)


def print_metrics(name, metrics):
    """Print regression and delay-risk metrics."""

    print(f"{name}:")
    print(f"  MAE              = {metrics['mae']:.4f} minutes")
    print(f"  RMSE             = {metrics['rmse']:.4f} minutes")
    print(f"  R²               = {metrics['r2']:.4f}")
    print(f"  Variance Explained= {metrics['r2'] * 100:.2f}%")
    print(f"  Delay Threshold  = {metrics['threshold']:.4f} minutes")
    print(f"  Delay Accuracy   = {metrics['accuracy']:.4f}")
    print(f"  Delay Error Rate = {metrics['error_rate']:.4f}")
    print(f"  Delay Precision  = {metrics['precision']:.4f}")
    print(f"  Delay Recall     = {metrics['recall']:.4f}")
    print(f"  Delay F1 Score   = {metrics['f1_score']:.4f}")


def log_metrics_to_mlflow(metrics, prefix=""):
    """
    Log regression and delay-risk metrics to MLflow.

    Example:
        prefix="val_"
        -> val_mae
        -> val_rmse
        -> val_r2
        -> val_accuracy
        ...
    """

    mlflow.log_metric(f"{prefix}mae", metrics["mae"])
    mlflow.log_metric(f"{prefix}mse", metrics["mse"])
    mlflow.log_metric(f"{prefix}rmse", metrics["rmse"])
    mlflow.log_metric(f"{prefix}r2", metrics["r2"])

    mlflow.log_metric(
        f"{prefix}delay_threshold",
        metrics["threshold"],
    )

    mlflow.log_metric(
        f"{prefix}accuracy",
        metrics["accuracy"],
    )

    mlflow.log_metric(
        f"{prefix}error_rate",
        metrics["error_rate"],
    )

    mlflow.log_metric(
        f"{prefix}precision",
        metrics["precision"],
    )

    mlflow.log_metric(
        f"{prefix}recall",
        metrics["recall"],
    )

    mlflow.log_metric(
        f"{prefix}f1_score",
        metrics["f1_score"],
    )


def make_baseline_predictions(
    X_train,
    y_train,
    X_validation,
    feature_columns,
):
    """
    Generate predictions for comparison baselines.

    Baselines:
    1. Mean prediction
    2. Median prediction
    3. Distance-only linear regression

    These baselines are NOT eligible for final model selection.
    """

    baseline_predictions = {}

    # --------------------------------------------------------
    # Mean baseline
    # --------------------------------------------------------

    mean_value = float(y_train.mean())

    baseline_predictions["MeanBaseline"] = np.full(
        len(X_validation),
        mean_value,
    )

    # --------------------------------------------------------
    # Median baseline
    # --------------------------------------------------------

    median_value = float(y_train.median())

    baseline_predictions["MedianBaseline"] = np.full(
        len(X_validation),
        median_value,
    )

    # --------------------------------------------------------
    # Distance-only linear regression
    # --------------------------------------------------------

    if "dist_km" not in feature_columns:
        raise ValueError(
            "dist_km is required for the Distance-only baseline."
        )

    distance_model = LinearRegression()

    distance_train = X_train[["dist_km"]]
    distance_validation = X_validation[["dist_km"]]

    distance_model.fit(
        distance_train,
        y_train,
    )

    baseline_predictions["DistanceOnlyLinear"] = (
        distance_model.predict(distance_validation)
    )

    return baseline_predictions
# ============================================================
# LIGHTGBM OPTIMIZATION
# ============================================================

def optimize_lightgbm(
    X_train,
    y_train,
    X_validation,
    y_validation,
    delay_threshold,
    n_trials=30,
):
    """
    Optimize LightGBM hyperparameters using the existing
    internal validation set.

    The held-out test set is NOT used here.
    Objective: minimize validation MAE.
    """

    print("\n")
    print("=" * 70)
    print("OPTIMIZING LIGHTGBM WITH OPTUNA")
    print("=" * 70)

    def objective(trial):

        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                300,
                1200,
            ),

            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.10,
                log=True,
            ),

            "num_leaves": trial.suggest_int(
                "num_leaves",
                15,
                127,
            ),

            "max_depth": trial.suggest_int(
                "max_depth",
                3,
                12,
            ),

            "min_child_samples": trial.suggest_int(
                "min_child_samples",
                5,
                50,
            ),

            "subsample": trial.suggest_float(
                "subsample",
                0.70,
                1.00,
            ),

            "subsample_freq": 1,

            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                0.70,
                1.00,
            ),

            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                0.0,
                1.0,
            ),

            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                0.0,
                5.0,
            ),

            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": -1,
        }

        model = LGBMRegressor(**params)

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_validation
        )

        metrics = evaluate_predictions(
            y_validation,
            predictions,
            threshold=delay_threshold,
        )

        return metrics["mae"]

    # --------------------------------------------------------
    # Create Optuna study
    # --------------------------------------------------------

    study = optuna.create_study(
        direction="minimize",
        study_name="lightgbm_eta_optimization",
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print("\n")
    print("=" * 70)
    print("OPTUNA OPTIMIZATION COMPLETED")
    print("=" * 70)

    print(f"Number of trials: {n_trials}")
    print(
        f"Best validation MAE: "
        f"{study.best_value:.4f} minutes"
    )

    print("\nBest LightGBM parameters:")

    for parameter, value in study.best_params.items():
        print(f"  {parameter}: {value}")

    return study

# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():

    print("=" * 70)
    print("INDIAN FOOD DELIVERY ETA PREDICTION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    print("\n[1/7] Loading feature datasets...")

    if not TRAIN_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Training features not found:\n{TRAIN_FEATURES_PATH}"
        )

    if not TEST_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Held-out test features not found:\n{TEST_FEATURES_PATH}"
        )

    train_df = pd.read_csv(TRAIN_FEATURES_PATH)
    test_df = pd.read_csv(TEST_FEATURES_PATH)

    print(f"Training dataset: {train_df.shape}")
    print(f"Held-out test dataset: {test_df.shape}")

    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in training data."
        )

    if TARGET_COLUMN not in test_df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in test data."
        )

    # --------------------------------------------------------
    # FEATURE COLUMNS
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in train_df.columns
        if column != TARGET_COLUMN
    ]

    print(f"\nNumber of features: {len(feature_columns)}")
    print(f"X shape: {train_df[feature_columns].shape}")
    print(f"y shape: {train_df[TARGET_COLUMN].shape}")

    X = train_df[feature_columns]
    y = train_df[TARGET_COLUMN]

    X_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN]
    # Delay-risk threshold is derived only from the full training target.
    # The held-out test set must never influence this threshold.
    delay_threshold = float(np.percentile(y, 75))
    print(
        f"Delay-risk threshold from training targets: "
        f"{delay_threshold:.4f} minutes"
    )
    print(f"Final test X shape: {X_test.shape}")
    print(f"Final test y shape: {y_test.shape}")

    # --------------------------------------------------------
    # 2. INTERNAL VALIDATION SPLIT
    # --------------------------------------------------------

    print("\n[2/7] Creating internal validation split...")

    X_train, X_validation, y_train, y_validation = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
    )

    print(f"Internal training rows: {len(X_train)}")
    print(f"Validation rows: {len(X_validation)}")
    print(f"Held-out test rows: {len(X_test)}")

    # --------------------------------------------------------
    # 3. PREPARE MODELS AND BASELINES
    # --------------------------------------------------------

    print("\n[3/7] Preparing baseline and ML models...")

    validation_results = {}

    # --------------------------------------------------------
    # 4. TRAIN CANDIDATE ML MODELS
    # --------------------------------------------------------

    print("\n[4/7] Training candidate models...")
    print_separator()

    best_model_name = None
    best_validation_mae = float("inf")
    best_model = None

    for model_name, model in ML_MODELS.items():

        print(f"\nTraining {model_name}...")

        model.fit(
            X_train,
            y_train,
        )

        validation_predictions = model.predict(
            X_validation
        )

        metrics = evaluate_predictions(
            y_validation,
            validation_predictions,
            threshold=delay_threshold,
        )

        validation_results[model_name] = metrics

        print_metrics(
            model_name,
            metrics,
        )

        # ----------------------------------------------------
        # MLflow validation run
        # ----------------------------------------------------

        with mlflow.start_run(
            run_name=f"{model_name}_validation"
        ):

            mlflow.log_param(
                "model",
                model_name,
            )

            mlflow.log_param(
                "dataset",
                "indian_food_delivery",
            )

            mlflow.log_param(
                "features",
                len(feature_columns),
            )

            mlflow.log_param(
                "validation_size",
                VALIDATION_SIZE,
            )

            mlflow.log_param(
                "selection_metric",
                "validation_mae",
            )

            # Log model-specific parameters
            if model_name == "RandomForest":
                mlflow.log_param(
                    "n_estimators",
                    model.n_estimators,
                )

                mlflow.log_param(
                    "max_depth",
                    model.max_depth,
                )

            elif model_name == "XGBoost":
                mlflow.log_param(
                    "n_estimators",
                    model.n_estimators,
                )

                mlflow.log_param(
                    "max_depth",
                    model.max_depth,
                )

                mlflow.log_param(
                    "learning_rate",
                    model.learning_rate,
                )

            elif model_name == "LightGBM":
                mlflow.log_param(
                    "n_estimators",
                    model.n_estimators,
                )

                mlflow.log_param(
                    "num_leaves",
                    model.num_leaves,
                )

                mlflow.log_param(
                    "max_depth",
                    model.max_depth,
                )

                mlflow.log_param(
                    "learning_rate",
                    model.learning_rate,
                )

            log_metrics_to_mlflow(
                metrics,
                prefix="val_",
            )

        # ----------------------------------------------------
        # MODEL SELECTION
        # ----------------------------------------------------
        #
        # IMPORTANT:
        # Only actual ML models can become the final model.
        # Baselines are comparison references only.
        # ----------------------------------------------------

        if metrics["mae"] < best_validation_mae:

            best_validation_mae = metrics["mae"]
            best_model_name = model_name
            best_model = model

    # --------------------------------------------------------
    # BASELINE EVALUATION
    # --------------------------------------------------------

    print("\n")
    print("Evaluating comparison baselines...")

    baseline_predictions = make_baseline_predictions(
        X_train,
        y_train,
        X_validation,
        feature_columns,
    )

    for baseline_name, predictions in baseline_predictions.items():

        metrics = evaluate_predictions(
            y_validation,
            predictions,
            threshold=delay_threshold,
        )

        validation_results[baseline_name] = metrics

        print(f"\n{baseline_name}:")
        print(f"  MAE              = {metrics['mae']:.4f} minutes")
        print(f"  RMSE             = {metrics['rmse']:.4f} minutes")
        print(f"  R²               = {metrics['r2']:.4f}")
        print(
            f"  Variance Explained= "
            f"{metrics['r2'] * 100:.2f}%"
        )
        print(
            f"  Delay Accuracy   = "
            f"{metrics['accuracy']:.4f}"
        )
        print(
            f"  Delay F1 Score   = "
            f"{metrics['f1_score']:.4f}"
        )

    print_separator()

    print("\nBest ML model based on validation MAE:")
    print(f"  {best_model_name}")
    print(f"  Validation MAE: {best_validation_mae:.4f} minutes")
      # ========================================================
    # LIGHTGBM OPTIMIZATION
    # ========================================================

    if best_model_name != "LightGBM":
        raise RuntimeError(
            f"Expected LightGBM to be selected for optimization, "
            f"but {best_model_name} won."
        )

    # --------------------------------------------------------
    # BEFORE optimization
    # --------------------------------------------------------

    lightgbm_before = validation_results["LightGBM"]

    # --------------------------------------------------------
    # Run Optuna optimization
    # --------------------------------------------------------

    study = optimize_lightgbm(
        X_train,
        y_train,
        X_validation,
        y_validation,
        delay_threshold,
        n_trials=30,
    )

    # --------------------------------------------------------
    # Build optimized LightGBM
    # --------------------------------------------------------

    optimized_params = study.best_params.copy()

    optimized_params.update({
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
        "subsample_freq": 1,
    })

    optimized_lightgbm = LGBMRegressor(
        **optimized_params
    )

    optimized_lightgbm.fit(
        X_train,
        y_train
    )

    optimized_predictions = optimized_lightgbm.predict(
        X_validation
    )

    lightgbm_after = evaluate_predictions(
        y_validation,
        optimized_predictions,
        threshold=delay_threshold,
    )

    # --------------------------------------------------------
    # Calculate improvement
    # --------------------------------------------------------

    mae_improvement = (
        (lightgbm_before["mae"] - lightgbm_after["mae"])
        / lightgbm_before["mae"]
    ) * 100

    rmse_improvement = (
        (lightgbm_before["rmse"] - lightgbm_after["rmse"])
        / lightgbm_before["rmse"]
    ) * 100

    r2_improvement = (
        lightgbm_after["r2"] - lightgbm_before["r2"]
    ) * 100

    print("\n")
    print("=" * 70)
    print("LIGHTGBM BEFORE vs AFTER OPTIMIZATION")
    print("=" * 70)

    print(
        f"{'Metric':<25}"
        f"{'Before':>12}"
        f"{'After':>12}"
        f"{'Change':>12}"
    )

    print("-" * 70)

    print(
        f"{'MAE (minutes)':<25}"
        f"{lightgbm_before['mae']:>12.4f}"
        f"{lightgbm_after['mae']:>12.4f}"
        f"{mae_improvement:>11.2f}%"
    )

    print(
        f"{'RMSE (minutes)':<25}"
        f"{lightgbm_before['rmse']:>12.4f}"
        f"{lightgbm_after['rmse']:>12.4f}"
        f"{rmse_improvement:>11.2f}%"
    )

    print(
        f"{'R² (%)':<25}"
        f"{lightgbm_before['r2'] * 100:>12.2f}"
        f"{lightgbm_after['r2'] * 100:>12.2f}"
        f"{r2_improvement:>11.2f} pp"
    )

    print(
        f"{'Delay Accuracy (%)':<25}"
        f"{lightgbm_before['accuracy'] * 100:>12.2f}"
        f"{lightgbm_after['accuracy'] * 100:>12.2f}"
    )

    print(
        f"{'Delay F1 (%)':<25}"
        f"{lightgbm_before['f1_score'] * 100:>12.2f}"
        f"{lightgbm_after['f1_score'] * 100:>12.2f}"
    )

    print("-" * 70)

    print("\nBest optimized parameters:")

    for parameter, value in study.best_params.items():
        print(f"  {parameter}: {value}")

    # --------------------------------------------------------
    # 5. RETRAIN SELECTED MODEL ON ALL TRAINING DATA
    # --------------------------------------------------------

    print("\n[5/7] Retraining the selected model on all training data...")

    if best_model is None:
        raise RuntimeError(
            "No ML model was selected."
        )

    final_model = LGBMRegressor(
        **optimized_params
    )

    final_model.fit(
        X,
        y,
    )
    best_model_name = "Optimized LightGBM"
    # --------------------------------------------------------
    # 6. FINAL HELD-OUT TEST EVALUATION
    # --------------------------------------------------------

    print("\n[6/7] Evaluating on the held-out test dataset...")

    final_predictions = final_model.predict(
        X_test
    )

    test_metrics = evaluate_predictions(
        y_test,
        final_predictions,
        threshold=delay_threshold,
    )

    print("\n")
    print("=" * 70)
    print("FINAL HELD-OUT TEST RESULTS")
    print("=" * 70)

    print(f"Model              : {best_model_name}")
    print(
        f"MAE                : "
        f"{test_metrics['mae']:.4f} minutes"
    )
    print(
        f"MSE                : "
        f"{test_metrics['mse']:.4f}"
    )
    print(
        f"RMSE               : "
        f"{test_metrics['rmse']:.4f} minutes"
    )
    print(
        f"R²                 : "
        f"{test_metrics['r2']:.4f}"
    )
    print(
        f"Variance Explained : "
        f"{test_metrics['r2'] * 100:.2f}%"
    )

    print("\nDelay-Risk Classification")
    print(
        f"Threshold          : "
        f"{test_metrics['threshold']:.4f} minutes"
    )
    print(
        f"Accuracy           : "
        f"{test_metrics['accuracy']:.4f}"
    )
    print(
        f"Error Rate         : "
        f"{test_metrics['error_rate']:.4f}"
    )
    print(
        f"Precision          : "
        f"{test_metrics['precision']:.4f}"
    )
    print(
        f"Recall             : "
        f"{test_metrics['recall']:.4f}"
    )
    print(
        f"F1 Score           : "
        f"{test_metrics['f1_score']:.4f}"
    )

    # --------------------------------------------------------
    # MLflow FINAL RUN
    # --------------------------------------------------------

    print("\nLogging final experiment to MLflow...")

    with mlflow.start_run(
        run_name=f"{best_model_name}_final"
    ):

        mlflow.log_param(
            "model",
            best_model_name,
        )

        mlflow.log_param(
            "dataset",
            "indian_food_delivery",
        )

        mlflow.log_param(
            "training_rows",
            len(X),
        )

        mlflow.log_param(
            "validation_rows",
            len(X_validation),
        )

        mlflow.log_param(
            "held_out_test_rows",
            len(X_test),
        )

        mlflow.log_param(
            "number_of_features",
            len(feature_columns),
        )

        mlflow.log_param(
            "selection_metric",
            "validation_mae",
        )

        mlflow.log_param(
            "held_out_test_used_for_selection",
            False,
        )

        mlflow.log_param(
            "delay_risk_threshold_method",
            "75th_percentile_of_full_training_targets",
        )

        mlflow.log_param(
            "delay_risk_threshold_source_rows",
            len(y),
        )

        log_metrics_to_mlflow(
            test_metrics,
            prefix="test_",
        )

        # ----------------------------------------------------
        # Save model to MLflow
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            final_model,
            artifact_path="model",
            skops_trusted_types=[
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBRegressor",
                "sklearn.ensemble._forest.RandomForestRegressor",
                "lightgbm.sklearn.LGBMRegressor",
                "collections.OrderedDict",
                "lightgbm.basic.Booster",
            ],
        )

    # --------------------------------------------------------
    # SAVE MODEL LOCALLY
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        final_model,
        MODEL_PATH,
    )

    with open(
        FEATURE_LIST_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            feature_columns,
            f,
            indent=2,
        )

    print("\nSaved final model:")
    print(f"  {MODEL_PATH}")

    print("\nSaved feature list:")
    print(f"  {FEATURE_LIST_PATH}")

    # --------------------------------------------------------
    # 7. GENERATE MODEL COMPARISON REPORT
    # --------------------------------------------------------

    print("\n[7/7] Generating model comparison report...")

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Sort by validation MAE
    sorted_results = sorted(
        validation_results.items(),
        key=lambda item: item[1]["mae"],
    )

    report_lines = []

    report_lines.append(
        "# Indian Food Delivery ETA Model Comparison\n"
    )

    report_lines.append(
        "## Dataset\n"
    )

    report_lines.append(
        f"- Training rows: **{len(X)}**"
    )

    report_lines.append(
        f"- Internal validation rows: **{len(X_validation)}**"
    )

    report_lines.append(
        f"- Held-out test rows: **{len(X_test)}**"
    )

    report_lines.append(
        f"- Input features: **{len(feature_columns)} features + 1 target = {len(feature_columns) + 1} columns**"
    )

    report_lines.append(
        f"- Target: **{TARGET_COLUMN}**"
    )

    report_lines.append(
        "- The held-out test set was kept separate from model selection."
    )

    report_lines.append(
        "\n## Model Selection\n"
    )

    report_lines.append(
       f"All three ML models — **Random Forest**, **XGBoost**, "
    f"and **LightGBM** — were eligible for final model selection."
    )

    report_lines.append(
        f"The selected model was **{best_model_name}**, "
        f"with the lowest validation MAE of "
        f"**{best_validation_mae:.4f} minutes**."
    )

    report_lines.append(
        "\nBaselines were evaluated only to provide reference performance "
        "and were not eligible to become the final model."
    )

    report_lines.append(
        "\n## Validation Results\n"
    )

    report_lines.append(
        "| Model | MAE (min) | RMSE (min) | R² | Variance Explained | Delay Accuracy | Delay F1 |"
    )

    report_lines.append(
        "|---|---:|---:|---:|---:|---:|---:|"
    )

    for model_name, metrics in sorted_results:

        report_lines.append(
            f"| {model_name} "
            f"| {metrics['mae']:.4f} "
            f"| {metrics['rmse']:.4f} "
            f"| {metrics['r2']:.4f} "
            f"| {metrics['r2'] * 100:.2f}% "
            f"| {metrics['accuracy']:.4f} "
            f"| {metrics['f1_score']:.4f} |"
        )

    report_lines.append(
        "\n## Final Held-Out Test Performance\n"
    )

    report_lines.append(
        f"- Model: **{best_model_name}**"
    )

    report_lines.append(
        f"- MAE: **{test_metrics['mae']:.4f} minutes**"
    )

    report_lines.append(
        f"- MSE: **{test_metrics['mse']:.4f}**"
    )

    report_lines.append(
        f"- RMSE: **{test_metrics['rmse']:.4f} minutes**"
    )

    report_lines.append(
        f"- R²: **{test_metrics['r2']:.4f}**"
    )

    report_lines.append(
        f"- Variance explained: "
        f"**{test_metrics['r2'] * 100:.2f}%**"
    )

    report_lines.append(
        "\n### Delay-Risk Performance\n"
    )

    report_lines.append(
        f"- Delay threshold: "
        f"**{test_metrics['threshold']:.4f} minutes**"
    )

    report_lines.append(
        f"- Accuracy: **{test_metrics['accuracy']:.4f}**"
    )

    report_lines.append(
        f"- Error rate: **{test_metrics['error_rate']:.4f}**"
    )

    report_lines.append(
        f"- Precision: **{test_metrics['precision']:.4f}**"
    )

    report_lines.append(
        f"- Recall: **{test_metrics['recall']:.4f}**"
    )

    report_lines.append(
        f"- F1 score: **{test_metrics['f1_score']:.4f}**"
    )

    report_lines.append(
        "\n## Metric Interpretation\n"
    )

    report_lines.append(
        "- **MAE** represents the average absolute ETA prediction error in minutes."
    )

    report_lines.append(
        "- **RMSE** penalizes larger prediction errors more strongly than MAE."
    )

    report_lines.append(
        "- **R²** measures the proportion of variance in delivery time explained by the model."
    )

    report_lines.append(
        "- **R² is not classification accuracy.** For example, R² = 0.8271 means the model explains approximately 82.71% of the variance in delivery time."
    )

    report_lines.append(
        "- **Delay accuracy, precision, recall and F1** belong to the separate delay-risk classification layer."
    )

    report_lines.append(
        "\n## Baseline Purpose\n"
    )

    report_lines.append(
        "The baseline models establish simple reference points against which "
        "the machine-learning models can be evaluated."
    )

    report_lines.append(
        "- Mean baseline: predicts the average delivery time."
    )

    report_lines.append(
        "- Median baseline: predicts the median delivery time."
    )

    report_lines.append(
        "- Distance-only linear regression: estimates delivery time using only distance."
    )

    report_lines.append(
        "\n## Final Model\n"
    )

    report_lines.append(
        f"The final model is **{best_model_name}**."
    )

    report_lines.append(
        "It was selected using internal validation data and then retrained "
        "on the complete training dataset before the final held-out evaluation."
    )

    REPORT_PATH.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print(f"Report saved to:")
    print(f"  {REPORT_PATH}")

    # --------------------------------------------------------
    # COMPLETION
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(f"\nSelected model: {best_model_name}")
    print(
        f"Final test MAE: "
        f"{test_metrics['mae']:.4f} minutes"
    )

    print(
        f"Final test RMSE: "
        f"{test_metrics['rmse']:.4f} minutes"
    )

    print(
        f"Final test R²: "
        f"{test_metrics['r2']:.4f}"
    )

    print(
        f"Variance explained: "
        f"{test_metrics['r2'] * 100:.2f}%"
    )

    print(
        f"Delay-risk F1: "
        f"{test_metrics['f1_score']:.4f}"
    )

    print("\nOutput files:")
    print(f"  Model:   {MODEL_PATH}")
    print(f"  Features:{FEATURE_LIST_PATH}")
    print(f"  Report:  {REPORT_PATH}")
    print(f"  MLflow:  {MLFLOW_DB}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    # Configure MLflow
    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    main()