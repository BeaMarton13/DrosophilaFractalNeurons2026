"""Train a random forest classifier for central neuron classes from tree properties."""
import os
import sys
import pickle
import joblib
import pandas as pd

from axon_dendrite_tree import AxonDendriteForest

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


MODEL_PATH = "intrinsic_central_neuron_classifier.pkl"
FOREST_PATH = "intrinsic_forest.pkl"


def build_df():
    """Build (or load a cached) AxonDendriteForest and collect a labeled feature DataFrame for central neurons."""
    tree_type = "undirected"
    rows = []

    if sys.platform == "darwin":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tree_dir = os.path.abspath(
            os.path.join(script_dir, f"../../data/{tree_type}/tree_properties/")
        )
        c_dir = os.path.abspath(
            os.path.join(script_dir, f"../../data/{tree_type}/c_values/")
        )
        class_fname_w_path = os.path.abspath(
            os.path.join(script_dir, "../../data/classification.csv")
        )
    else:
        res_dir = "/data/RESULTS/USERS/bea/drosophila/"
        class_dir = "/data/RESULTS/PROJECTS/drosophila/"
        tree_dir = os.path.abspath(
            os.path.join(res_dir, f"data_navis/{tree_type}/tree_properties/")
        )
        c_dir = os.path.abspath(
            os.path.join(res_dir, f"data_navis/{tree_type}/c_values/")
        )
        class_fname_w_path = os.path.abspath(
            os.path.join(class_dir, "classification.csv")
        )

    if os.path.exists(FOREST_PATH):
        with open(FOREST_PATH, "rb") as f:
            forest = pickle.load(f)
    else:
        forest = AxonDendriteForest.build_forest_from_directory(
            tree_dir,
            c_dir,
            undirected=(tree_type == "undirected"),
            class_fname_w_path=class_fname_w_path,
        )
        with open(FOREST_PATH, "wb") as f:
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
                    "height": tree.tree_properties.get("height"),
                    "max_width": tree.tree_properties.get("max_width"),
                    "cable_length": tree.tree_properties.get("cable_length"),
                    "number_of_nodes": tree.tree_properties.get("num_nodes"),
                    "num_filtered_pre_synapses": tree.tree_properties.get("num_filtered_pre_synapses"),
                    "num_filtered_post_synapses": tree.tree_properties.get("num_filtered_post_synapses"),
                    "num_leaf_nodes": tree.tree_properties.get("num_leaf_nodes"),
                    "class": tree.class_,
                }
            )

    return pd.DataFrame(rows)


def get_model():
    """Construct the RandomForestClassifier used for training."""
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def train_and_save_model(X_train, y_train, feature_columns):
    """Fit the model, bundle it with its feature list, and save it to MODEL_PATH."""
    model = get_model()
    model.fit(X_train, y_train)

    bundle = {
        "model_name": "RandomForest",
        "model": model,
        "features": feature_columns,
    }

    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")
    return bundle


def load_model_bundle():
    """Load a previously saved model bundle from MODEL_PATH."""
    bundle = joblib.load(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")
    return bundle


def main():
    """Build the dataset, train the classifier, and report holdout performance."""
    df = build_df().dropna().copy()

    feature_columns = [col for col in df.columns if col in ["fractal_dimension", "cable_length"]]
    X = df[feature_columns]
    y = df["class"]

    print(f"Using features: {feature_columns}")
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {y.nunique()}")
    print("\nClass counts:")
    print(y.value_counts())

    min_class_count = y.value_counts().min()
    if min_class_count < 2:
        raise ValueError(
            f"At least one class has only {min_class_count} sample(s). "
            "Cannot do stratified split safely."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    RETRAIN = True
    if os.path.exists(MODEL_PATH) and not RETRAIN:
        bundle = load_model_bundle()
    else:
        bundle = train_and_save_model(X_train, y_train, feature_columns)

    model = bundle["model"]
    saved_features = bundle["features"]

    # Ensure test columns match saved training columns
    X_test = X_test[saved_features]

    y_pred = model.predict(X_test)

    print("\n=== Holdout test results ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    if hasattr(model, "feature_importances_"):
        importances = pd.Series(
            model.feature_importances_, index=saved_features
        ).sort_values(ascending=False)
        print("\nFeature importances:")
        print(importances)


if __name__ == "__main__":
    main()