"""Filter skeletons by their maximum intrinsic/extrinsic distance, tagging those with max distance <= 0.25."""
import os

# Disable internal parallelism to prevent subprocess spawning
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import navis

navis.set_pbars(hide=True)  # Disable progress bars
navis.config.parallel = False  # Disable navis parallelism if available

# Clear navis cache if it exists
if hasattr(navis, 'cache'):
    navis.cache.clear()

import sys
import gc
import pandas as pd
import argparse

from src.utils.skeleton_tree import SkeletonTree
from main import process_single_skeleton


if sys.platform == "darwin":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    connector_filename = os.path.abspath(os.path.join(script_dir, f"./filtered_connectors/{skeleton_id}.csv"))
    split_filename = os.path.abspath(os.path.join(script_dir, f"./data/calculate_dendrite_axonend_distances-MIN.csv"))
else:
    # We will call this script_dir for easier usage
    script_dir = "/data/RESULTS/USERS/bea/drosophila/"
    in_script_dir = "/data/RESULTS/PROJECTS/drosophila/filtered_connectors/"
    split_filename = "/data/RESULTS/PROJECTS/drosophila/calculate_dendrite_axonend_distances-MIN/calculate_dendrite_axonend_distances-MIN.csv"

def filter_by_max_dist(skeleton_tree, connector_filename, extrinsic=False):
    """Compute the skeleton's max distance and append its ID to the appropriate filtered-ID file(s) if <= 0.25."""
    if os.path.isfile(connector_filename):
        filtered_connectors = pd.read_csv(connector_filename)
        skeleton_tree.skeleton._set_connectors(filtered_connectors)
        # Full skeleton
        max_dist = max(skeleton_tree.calculate_distances())
        del filtered_connectors

    with open("filtered_skeleton_ids(025)_full.txt", "a") as f_full:
        if max_dist <= 0.25:
            f_full.write(f"{skeleton_tree.skeleton_id}\n")
    if extrinsic:
        with open("filtered_skeleton_ids(025)_extrinsic.txt", "a") as f_extrinsic:
            if max_dist <= 0.25:
                f_extrinsic.write(f"{skeleton_tree.skeleton_id}\n")
    else:
        with open("filtered_skeleton_ids(025)_intrinsic.txt", "a") as f_intrinsic:
            if max_dist <= 0.25:
                f_intrinsic.write(f"{skeleton_tree.skeleton_id}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add a new property to the saved properties.")
    parser.add_argument("start", type=str, help="Start index")
    parser.add_argument("length", type=str, help="Length of the batch")
    args = parser.parse_args()
    start = int(args.start)
    length = int(args.length)

    if sys.platform == "darwin":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        connector_filename = os.path.abspath(os.path.join(script_dir, f"./filtered_connectors/{skeleton_id}.csv"))
        SKELETONS_DIR = None
    else:
        # We will call this script_dir for easier usage
        script_dir = "/data/RESULTS/USERS/bea/drosophila/"
        class_dir = "/data/RESULTS/PROJECTS/drosophila/"
        in_script_dir = "/data/RESULTS/PROJECTS/drosophila/filtered_connectors/"
        SKELETONS_DIR = "/data/RESULTS/PROJECTS/drosophila/input_data/full"
        filtered_connectors_ext_filelist = "/data/RESULTS/PROJECTS/drosophila/filtered_connectors/filtered_connectors-GOOD_EXTRINSIC.filelist"
        filtered_connectors_all = "/data/RESULTS/PROJECTS/drosophila/filtered_connectors/"
        class_fname_w_path = os.path.abspath(os.path.join(class_dir, f"classification.csv"))

    filtered_connectors_ext = set(x for x in pd.read_csv(filtered_connectors_ext_filelist, header=None).iloc[:,0].tolist() if ".csv" in x)
    process_only_not_oks = False

    df = pd.read_csv(class_fname_w_path)
    super_classes_dict = {int(row['root_id']): row['super_class'] for _, row in df.iterrows()}
    del df  # Free memory

    # Use sets for O(1) lookup
    processed_filtering_all = set()
    processed_filtering_ok = set()
    if os.path.isfile("processed_filtering_all.txt"):
        with open("processed_filtering_all.txt", "r") as f_processed:
            processed_filtering_all = {int(line.strip()) for line in f_processed if line.strip()}
    if os.path.isfile("processed_filtering_ok.txt"):
        with open("processed_filtering_ok.txt", "r") as f_processed:
            processed_filtering_ok = {int(line.strip()) for line in f_processed if line.strip()}

    if not process_only_not_oks:
        ids = os.listdir(filtered_connectors_all)
        remaining_ids = set(ids) - {f"{s_id}.csv" for s_id in processed_filtering_all}
        ids = list(remaining_ids)
    else:
        ids = set(processed_filtering_all) - set(processed_filtering_ok)
        ids = [f"{s_id}.csv" for s_id in ids]

    print(len(ids), "skeletons to process after filtering processed IDs.")

    if start + length <= len(ids):
        ids = ids[start:start+length]
    else:
        ids = ids[start:]

    for i, connector in enumerate(ids):
        if not connector.endswith(".csv"):
            continue

        skeleton_id = int(connector.split(".csv")[0])
        print(f"[{i}/{len(ids)}] {connector}", super_classes_dict.get(skeleton_id, "Unknown"))
        connector_filename = os.path.abspath(os.path.join(in_script_dir, connector))

        skeleton_tree = None

        if skeleton_id not in processed_filtering_all:
            try:
                skeleton_tree = SkeletonTree.from_properties(SKELETONS_DIR, skeleton_id, skeleton_type="full", undirected=True, lightweight=True)
                filter_by_max_dist(skeleton_tree, connector_filename, extrinsic=connector in filtered_connectors_ext)
                with open("processed_filtering_ok.txt", "a") as f_processed:
                    f_processed.write(f"{skeleton_id}\n")
            except Exception as e:
                print(f"Error processing skeleton ID {skeleton_id}: {e}")
                ok = process_single_skeleton(skeleton_id, True, SKELETONS_DIR, None, 'navis')

                if ok:
                    skeleton_tree = SkeletonTree.from_properties(SKELETONS_DIR, skeleton_id, skeleton_type="full", undirected=True, lightweight=True)
                    filter_by_max_dist(skeleton_tree, connector_filename, extrinsic=connector in filtered_connectors_ext)
                    with open("processed_filtering_ok.txt", "a") as f_processed:
                        f_processed.write(f"{skeleton_id}\n")

        elif skeleton_id not in processed_filtering_ok:
            ok = process_single_skeleton(skeleton_id, True, SKELETONS_DIR, None, 'navis')
            if ok:
                skeleton_tree = SkeletonTree.from_properties(SKELETONS_DIR, skeleton_id, skeleton_type="full", undirected=True, lightweight=True)
                filter_by_max_dist(skeleton_tree, connector_filename, extrinsic=connector in filtered_connectors_ext)
                with open("processed_filtering_ok.txt", "a") as f_processed:
                    f_processed.write(f"{skeleton_id}\n")
            else:
                print(f"Skipping skeleton ID {skeleton_id}, error while processing.")

        with open("processed_filtering_all.txt", "a") as f_processed:
            f_processed.write(f"{skeleton_id}\n")

        # Clean up memory aggressively
        if skeleton_tree is not None:
            if hasattr(skeleton_tree, 'skeleton'):
                del skeleton_tree.skeleton
            del skeleton_tree

        # Clear navis cache every iteration
        if hasattr(navis, 'cache') and hasattr(navis.cache, 'clear'):
            navis.cache.clear()

        # Force garbage collection every 10 iterations
        if i % 10 == 0:
            gc.collect()
            print(f"Memory cleaned at iteration {i}")