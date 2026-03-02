"""
Module huấn luyện và đánh giá mô hình phân loại văn bản.
"""

import time
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC


# Mapping tên model -> class sklearn
MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression,
    "svm": LinearSVC,
    "multinomial_nb": MultinomialNB,
    "complement_nb": ComplementNB,
    "random_forest": RandomForestClassifier,
}

# Params mặc định cho từng model - đây là điểm config của pipeline
DEFAULT_PARAMS = {
    "logistic_regression": {
        "max_iter": 1000,
        "C": 1.0,
        "solver": "saga",
        "n_jobs": -1,
        "random_state": 42,
    },
    "svm": {
        "C": 1.0,
        "max_iter": 2000,
        "random_state": 42,
    },
    "multinomial_nb": {
        "alpha": 1.0,
    },
    "complement_nb": {
        "alpha": 1.0,
    },
    "random_forest": {
        "n_estimators": 50,
        "max_features": "sqrt",
        "max_depth": None,
        "min_samples_leaf": 2,
        "n_jobs": -1,
        "random_state": 42,
    },
}


def get_model(model_name: str, **override_params):
    """
    Tạo instance model theo tên. override_params để thay đổi hyperparams mặc định.
    Ví dụ: get_model('logistic_regression', C=0.5)
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_name}' không hỗ trợ. Chọn trong: {list(MODEL_REGISTRY.keys())}"
        )

    params = DEFAULT_PARAMS.get(model_name, {}).copy()
    params.update(override_params)
    return MODEL_REGISTRY[model_name](**params)


def train_and_evaluate(
    model_name: str,
    X_train,
    y_train,
    X_test,
    y_test,
    label_names: list[str] | None = None,
    **model_params,
) -> dict[str, Any]:
    """
    Train một model và trả về dict kết quả đầy đủ.
    """
    model = get_model(model_name, **model_params)

    print(f"\n  Training {model_name}...")
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  Done in {train_time:.1f}s")

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    results = {
        "model_name": model_name,
        "model": model,
        "model_params": model_params,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "train_time_s": round(train_time, 2),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=label_names,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
    }

    print(
        f"  Accuracy: {acc:.4f} | F1-macro: {f1_macro:.4f} | F1-weighted: {f1_weighted:.4f}"
    )
    return results


def run_experiment(
    feature_name: str,
    X_train,
    y_train,
    X_test,
    y_test,
    models: list[str] | None = None,
    label_names: list[str] | None = None,
    model_configs: dict | None = None,
) -> list[dict]:
    """
    Chạy nhiều models trên cùng một bộ features, trả về list kết quả.

    model_configs: dict model_name -> dict params,
      ví dụ {'logistic_regression': {'C': 0.5}, 'svm': {'C': 2.0}}
    """
    if models is None:
        models = list(MODEL_REGISTRY.keys())
    if model_configs is None:
        model_configs = {}

    print(f"\n{'='*60}")
    print(f"  Feature: {feature_name}")
    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"{'='*60}")

    all_results = []
    for model_name in models:
        params = model_configs.get(model_name, {})
        try:
            result = train_and_evaluate(
                model_name,
                X_train,
                y_train,
                X_test,
                y_test,
                label_names=label_names,
                **params,
            )
            result["feature_name"] = feature_name
            all_results.append(result)
        except Exception as e:
            print(f"  ERROR [{model_name}]: {e}")

    return all_results


def summarize_results(all_results: list[dict]) -> "pd.DataFrame":
    """
    Tổng hợp kết quả thành DataFrame để so sánh dễ hơn.
    """
    import pandas as pd

    rows = []
    for r in all_results:
        rows.append(
            {
                "Feature": r.get("feature_name", ""),
                "Model": r["model_name"],
                "Accuracy": round(r["accuracy"], 4),
                "F1 Macro": round(r["f1_macro"], 4),
                "F1 Weighted": round(r["f1_weighted"], 4),
                "Train Time (s)": r["train_time_s"],
            }
        )
    df = pd.DataFrame(rows)
    df = df.sort_values("F1 Weighted", ascending=False).reset_index(drop=True)
    return df
