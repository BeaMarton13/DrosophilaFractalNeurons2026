"""Classify previously unlabeled (NaN/unknown) central neurons using a trained model."""
import os
import pickle
import joblib
import pandas as pd

from axon_dendrite_tree import AxonDendriteForest


MODEL_PATH = "intrinsic_central_neuron_classifier.pkl"
FOREST_PATH = "intrinsic_forest.pkl"
OUTPUT_FOREST_PATH = "intrinsic_forest_nan_classified.pkl"


def load_forest():
    """Load the cached AxonDendriteForest pickle produced by classify_data.py."""
    if not os.path.exists(FOREST_PATH):
        raise FileNotFoundError("Forest pickle not found. Build it first.")

    with open(FOREST_PATH, "rb") as f:
        forest = pickle.load(f)

    print("Loaded forest")
    return forest


def load_model():
    """Load the trained model bundle and return (model, feature_columns)."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Train it first.")

    bundle = joblib.load(MODEL_PATH)
    print("Loaded model")

    return bundle["model"], bundle["features"]


def extract_features(tree):
    """Collect the model's input features from a single tree's properties."""
    return {
        "fractal_dimension": tree.tree_properties.get("fractal_dimension"),
        "height": tree.tree_properties.get("height"),
        "max_width": tree.tree_properties.get("max_width"),
        "cable_length": tree.tree_properties.get("cable_length"),
        "number_of_nodes": tree.tree_properties.get("num_nodes"),
        "num_filtered_pre_synapses": tree.tree_properties.get("num_filtered_pre_synapses"),
        "num_filtered_post_synapses": tree.tree_properties.get("num_filtered_post_synapses"),
        "num_leaf_nodes": tree.tree_properties.get("num_leaf_nodes"),
    }


def main():
    """Classify all NaN/unknown central neurons in the forest and save the updated forest."""
    forest = load_forest()
    model, feature_columns = load_model()

    nan_trees = []
    rows = []

    for tree in forest.full_tree:
        class_name = str(tree.class_).lower()

        # Identify trees to classify
        if (
            tree.super_class.lower() == "central"
            and (class_name == "nan" or "unknown" in class_name)
        ):
            features = extract_features(tree)

            # Skip if any feature missing
            if any(v is None for v in features.values()):
                continue

            nan_trees.append(tree)
            rows.append(features)

    print(f"Found {len(nan_trees)} trees to classify")

    if len(rows) == 0:
        print("No valid trees to classify")
        return

    X = pd.DataFrame(rows)

    # Ensure feature order matches training
    X = X[feature_columns]

    predictions = model.predict(X)

    # Assign predictions back to trees
    for tree, pred in zip(nan_trees, predictions):
        tree.predicted_class = pred  # optional: keep original intact
        tree.class_ = pred  # overwrite class

    print(f"Classified {len(predictions)} trees")

    # Save updated forest
    with open(OUTPUT_FOREST_PATH, "wb") as f:
        pickle.dump(forest, f)

    print(f"Saved updated forest to {OUTPUT_FOREST_PATH}")


if __name__ == "__main__":
    main()