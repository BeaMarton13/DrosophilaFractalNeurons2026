"""Compare several classifier models (cross-validated) for central neuron classification."""
import os
import sys
import pickle
import joblib
import numpy as np
import pandas as pd

from axon_dendrite_tree import AxonDendriteForest

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, make_scorer, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def build_df():
    """Build (or load a cached) AxonDendriteForest and collect a labeled feature DataFrame for central neurons."""
    tree_type = "undirected"
    rows = []

    if sys.platform == "darwin":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tree_dir = os.path.abspath(os.path.join(script_dir, f"../../data/{tree_type}/tree_properties/"))
        c_dir = os.path.abspath(os.path.join(script_dir, f"../../data/{tree_type}/c_values/"))
        class_fname_w_path = os.path.abspath(os.path.join(script_dir, "../../data/classification.csv"))
    else:
        res_dir = "/data/RESULTS/USERS/bea/drosophila/"
        class_dir = "/data/RESULTS/PROJECTS/drosophila/"
        tree_dir = os.path.abspath(os.path.join(res_dir, f"data_navis/{tree_type}/tree_properties/"))
        c_dir = os.path.abspath(os.path.join(res_dir, f"data_navis/{tree_type}/c_values/"))
        class_fname_w_path = os.path.abspath(os.path.join(class_dir, "classification.csv"))

    forest_path = "intrinsic_forest.pkl"

    if os.path.exists(forest_path):
        with open(forest_path, "rb") as f:
            forest = pickle.load(f)
    else:
        forest = AxonDendriteForest.build_forest_from_directory(
            tree_dir,
            c_dir,
            undirected=(tree_type == "undirected"),
            class_fname_w_path=class_fname_w_path,
        )
        with open(forest_path, "wb") as f:
            pickle.dump(forest, f)

    for tree in forest.full_tree:
        class_name = str(tree.class_).lower()
        if (
            tree.super_class.lower() == "central"
            and class_name != "nan"
            and "unknown" not in class_name
        ):
            rows.append(
                {
                    "id": tree.id,
                    "fractal_dimension": tree.tree_properties.get("fractal_dimension"),
                    "cable_length": tree.tree_properties.get("cable_length"),
                    "class": tree.class_,
                }
            )

    return pd.DataFrame(rows)


def main():
    """Cross-validate several classifiers, pick the best by macro F1, and report holdout performance."""
    df = build_df().dropna().copy()

    feature_columns = ["fractal_dimension", "cable_length"]
    X = df[feature_columns]
    y = df["class"]

    print("Original class counts:")
    print(y.value_counts())

    # Remove ultra-rare classes
    min_samples_per_class = 10
    keep_classes = y.value_counts()[y.value_counts() >= min_samples_per_class].index
    df = df[df["class"].isin(keep_classes)].copy()

    X = df[feature_columns]
    y = df["class"]

    print("\nFiltered class counts:")
    print(y.value_counts())

    print(f"\nUsing features: {feature_columns}")
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {y.nunique()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    min_train_class_count = y_train.value_counts().min()
    n_splits = min(5, min_train_class_count)
    if n_splits < 2:
        raise ValueError("Not enough samples in smallest class for stratified CV.")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scoring = {
        "accuracy": "accuracy",
        "f1_macro": make_scorer(f1_score, average="macro"),
        "f1_weighted": make_scorer(f1_score, average="weighted"),
    }

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            multi_class="multinomial",
        ),
        "SVC": SVC(
            class_weight="balanced",
            kernel="rbf",
        ),
    }

    results = []

    print("\n=== Cross-validation results ===")
    for name, model in models.items():
        cv_results = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        row = {
            "model": name,
            "cv_accuracy_mean": np.mean(cv_results["test_accuracy"]),
            "cv_accuracy_std": np.std(cv_results["test_accuracy"]),
            "cv_f1_macro_mean": np.mean(cv_results["test_f1_macro"]),
            "cv_f1_macro_std": np.std(cv_results["test_f1_macro"]),
            "cv_f1_weighted_mean": np.mean(cv_results["test_f1_weighted"]),
            "cv_f1_weighted_std": np.std(cv_results["test_f1_weighted"]),
        }
        results.append(row)

        print(
            f"{name:22s} "
            f"accuracy={row['cv_accuracy_mean']:.4f}±{row['cv_accuracy_std']:.4f}   "
            f"macro_f1={row['cv_f1_macro_mean']:.4f}±{row['cv_f1_macro_std']:.4f}   "
            f"weighted_f1={row['cv_f1_weighted_mean']:.4f}±{row['cv_f1_weighted_std']:.4f}"
        )

    results_df = pd.DataFrame(results).sort_values(by="cv_f1_macro_mean", ascending=False)

    print("\n=== Ranked by CV macro F1 ===")
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["model"]
    best_model = models[best_model_name]

    print(f"\nBest model by CV macro F1: {best_model_name}")

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    print("\n=== Holdout test results ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    output = {
        "model_name": best_model_name,
        "model": best_model,
        "features": feature_columns,
        "classes": sorted(y.unique()),
        "cv_results": results_df,
        "min_samples_per_class": min_samples_per_class,
    }

    joblib.dump(output, "intrinsic_central_neuron_classifier_2features.pkl")
    print("\nSaved model bundle to intrinsic_central_neuron_classifier_2features.pkl")


if __name__ == "__main__":
    main()