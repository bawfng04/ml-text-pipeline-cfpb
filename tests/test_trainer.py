import numpy as np
import pytest

from modules.trainer import get_model, run_experiment, summarize_results, train_and_evaluate


def _tiny_binary_dataset():
    x_train = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    y_train = np.array([0, 0, 1, 1], dtype=np.int64)
    x_test = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    y_test = np.array([0, 1], dtype=np.int64)
    return x_train, y_train, x_test, y_test


def test_get_model_unknown_name_raises():
    with pytest.raises(ValueError):
        get_model("does_not_exist")


def test_train_and_evaluate_returns_expected_keys():
    x_train, y_train, x_test, y_test = _tiny_binary_dataset()
    result = train_and_evaluate(
        "logistic_regression", x_train, y_train, x_test, y_test, max_iter=200
    )

    for key in [
        "model_name",
        "accuracy",
        "f1_macro",
        "f1_weighted",
        "train_time_s",
        "classification_report",
        "confusion_matrix",
        "y_pred",
    ]:
        assert key in result


def test_run_experiment_runs_selected_models():
    x_train, y_train, x_test, y_test = _tiny_binary_dataset()
    results = run_experiment(
        "toy",
        x_train,
        y_train,
        x_test,
        y_test,
        models=["logistic_regression", "svm"],
    )

    assert len(results) == 2
    assert {r["model_name"] for r in results} == {"logistic_regression", "svm"}
    assert all(r["feature_name"] == "toy" for r in results)


def test_summarize_results_sorted_desc_by_f1_weighted():
    rows = [
        {
            "feature_name": "A",
            "model_name": "m1",
            "accuracy": 0.5,
            "f1_macro": 0.5,
            "f1_weighted": 0.5,
            "train_time_s": 1.0,
        },
        {
            "feature_name": "B",
            "model_name": "m2",
            "accuracy": 0.8,
            "f1_macro": 0.8,
            "f1_weighted": 0.8,
            "train_time_s": 2.0,
        },
    ]
    df = summarize_results(rows)
    assert df.iloc[0]["Feature"] == "B"
    assert df.iloc[0]["F1 Weighted"] >= df.iloc[1]["F1 Weighted"]
