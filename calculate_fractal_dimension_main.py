"""Recompute the fractal dimension (full/axon/dendrite) for a batch of skeletons and update their saved properties."""
import os
import sys
import gc
import signal

# Disable internal parallelism to prevent subprocess spawning
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import navis

navis.set_pbars(hide=True)
navis.config.parallel = False

import numpy as np
import json
import argparse

from src.utils.skeleton_tree import SkeletonTree
from src.utils.fractal_dimension import fractal_dimension_sparse


def select_skeleton(skeleton, skeleton_type="full", undirected=True):
    """Break a skeleton into fragments and return the one with the most nodes as a SkeletonTree."""
    fragments = navis.break_fragments(skeleton)
    number_of_nodes_in_fragments = []
    print(len(fragments), "fragments found for skeleton type:", skeleton_type, skeleton)
    if len(fragments) > 1:
        for f in fragments:
            skeleton_tree = SkeletonTree.from_skeleton(skeleton=f, skeleton_type=skeleton_type, undirected=undirected, scaled_coords=None, lightweight=True)
            number_of_nodes_in_fragments.append(skeleton_tree.num_nodes)
            del skeleton_tree  # Clean up immediately
    selected_fragment_idx = np.argmax(number_of_nodes_in_fragments) if len(number_of_nodes_in_fragments) > 0 else 0
    skeleton_tree = SkeletonTree.from_skeleton(skeleton=fragments[selected_fragment_idx], skeleton_type=skeleton_type, undirected=undirected, scaled_coords=None, lightweight=True)
    print(f"For skeleton {skeleton_type} with {len(fragments)} fragments, selected fragment index: {selected_fragment_idx} with {number_of_nodes_in_fragments[selected_fragment_idx] if len(number_of_nodes_in_fragments) > 0 else skeleton_tree.num_nodes} nodes.")
    
    # Clean up fragments
    del fragments
    del number_of_nodes_in_fragments
    
    return skeleton_tree

def calculate_fractal_dimension(skeleton_id, skeleton_tree_full, split, dist, skeleton_type="full", undirected=True):
    """Compute (and persist) the fractal dimension for one compartment (full/axon/dendrite) of a skeleton."""
    skeleton_navis = None
    skeleton = None
    additional_points = None
    
    try:
        if skeleton_type == "dendrite":
            skeleton_navis = split[(split.compartment == 'dendrite')][0]
        elif skeleton_type == "axon":
            skeleton_navis = split[(split.compartment == 'axon')][0]
        else:
            skeleton_navis = skeleton_tree_full.skeleton
        
        skeleton = select_skeleton(skeleton_navis, skeleton_type=skeleton_type, undirected=True)
        fd = skeleton._calculate_fractal_dimension(skeleton_tree_full.scaled_coords, recalculate=True)
        
        if fd < 1:
            additional_points = get_additional_points(skeleton, distance=dist, scaled_coords=skeleton_tree_full.scaled_coords)
            coeffs, sizes, counts = fractal_dimension_sparse(additional_points, max_dist=dist, recalculate=True)
            fd = coeffs[0]
            del coeffs, sizes, counts
        
        update_property(skeleton_id, fd, undirected=True, skeleton_type=skeleton_type)
    
    finally:
        # Cleanup
        del skeleton_navis
        del skeleton
        del additional_points
        gc.collect()


def calculate_fractal_dimension_all(skeleton_tree, filtered_connectors, dist, undirected=True, skeleton_id=None):
    """Split a skeleton into axon/dendrite and compute the fractal dimension for full/axon/dendrite compartments."""
    split = None
    
    try:
        skeleton_tree.skeleton._set_connectors(filtered_connectors)
        split = navis.split_axon_dendrite(skeleton_tree.skeleton, metric='synapse_flow_centrality', reroot_soma=True, cellbodyfiber="soma")
        fd = skeleton_tree._calculate_fractal_dimension(None, recalculate=True)
        update_property(skeleton_id, fd, undirected=True, skeleton_type="full")

        calculate_fractal_dimension(skeleton_tree.id, skeleton_tree, split, dist, skeleton_type="axon", undirected=undirected)
        calculate_fractal_dimension(skeleton_tree.id, skeleton_tree, split, dist, skeleton_type="dendrite", undirected=undirected)
    
    finally:
        # Cleanup
        del split
        gc.collect()


def get_additional_points(skeleton, distance, scaled_coords, num_segments=None):
    """Interpolate extra 3D points along each node-parent edge of the skeleton, spaced by `distance`."""
    additional_points = []
    for idx, parent in zip(skeleton.skeleton.nodes['node_id'], skeleton.skeleton.nodes['parent_id']):
        if parent != -1:
            additional_points.extend(skeleton.interpolate_3d(scaled_coords[parent - 1], scaled_coords[idx - 1], distance=distance, num_segments=num_segments))
    return np.array(additional_points)

def update_property(skeleton_id, fractal_dimension, undirected=True, skeleton_type="full"):
    """Write the computed fractal dimension into the skeleton's saved properties JSON file."""
    if undirected:
        properties_dir = "/data/RESULTS/USERS/bea/drosophila/data_navis/undirected/tree_properties/"
    else:
        properties_dir = "/data/RESULTS/USERS/bea/drosophila/data_navis/directed/tree_properties/"
    properties_path = os.path.join(properties_dir, f"{skeleton_id}_{skeleton_type}.json")
    if not os.path.exists(properties_path):
        with open("missings_fd_rerun.txt", "a") as f:
            f.write(f"{skeleton_id}\n")
        raise FileNotFoundError(f"Properties file not found: {properties_path}")
        
    properties = json.load(open(properties_path, 'r'))
    properties['fractal_dimension'] = fractal_dimension
    with open(properties_path, 'w') as f:
        json.dump(properties, f, indent=4)

def get_ids_processed():
    """Read the list of skeleton IDs still to be (re)processed."""
    with open("remaining_fd_rerun_ids.txt", "r") as f:
        processed = f.read().splitlines()
    return [int(p) for p in processed]

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Process exceeded 120 seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add a new property to the saved properties.")
    parser.add_argument("start", type=str, help="Start index")
    parser.add_argument("length", type=str, help="Length of the batch")
    args = parser.parse_args()
    start = int(args.start)
    length = int(args.length)
    
    # NOTE: Change the directed/undirected cases, while experimenting
    tree_type = "undirected"
    undirected = True if tree_type == "undirected" else False

    with open("axon.txt", "r") as f:
        axon_ids = [int(line.strip()) for line in f if line.strip()]
    with open("dendrite.txt", "r") as f:
        dendrite_ids = [int(line.strip()) for line in f if line.strip()]
    with open("full.txt", "r") as f:
        full_ids = [int(line.strip()) for line in f if line.strip()]
    
    if not os.path.isfile("processed_fd_rerun.txt"):
        processed_ids = []
    else:
        with open("processed_fd_rerun.txt", "r") as f:
            processed_ids = [int(line.strip()) for line in f if line.strip()]

    if not os.path.isfile("missings_fd_rerun.txt"):
        missing_ids = []
    else:
        with open("missings_fd_rerun.txt", "r") as f:
            missing_ids = [int(line.strip()) for line in f if line.strip()]
    
    ids = get_ids_processed()

    if start + length <= len(ids):
        ids = ids[start:start+length]
    else:
        ids = ids[start:]

    if sys.platform == "darwin":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path_to_skeletons = "./input_data/full"
        SKELETONS_DIR = os.path.join(script_dir, path_to_skeletons)
    else:
        script_dir = "/data/RESULTS/USERS/bea/drosophila/"
        in_script_dir = "/data/RESULTS/PROJECTS/drosophila/input_data/full"
        SKELETONS_DIR = in_script_dir

    for skeleton_id in ids:
        skeleton = None
        skeleton_tree = None
        filtered_connectors = None


        # signal.signal(signal.SIGALRM, timeout_handler)
        # signal.alarm(120)  # 120 seconds timeout
        
        try:
            skeleton = SkeletonTree(SKELETONS_DIR, skeleton_id, skeleton_type="full", undirected=undirected, lightweight=True)
            
            skeleton_tree = select_skeleton(skeleton.skeleton, skeleton_type="full", undirected=undirected)
            
            if sys.platform == "darwin":
                script_dir = os.path.dirname(os.path.abspath(__file__))
                connector_filename = os.path.abspath(os.path.join(script_dir, f"../filtered_connectors/{skeleton_id}.csv"))
            else:
                script_dir = "/data/RESULTS/USERS/bea/drosophila/"
                in_script_dir = "/data/RESULTS/PROJECTS/drosophila/filtered_connectors/"
                connector_filename = os.path.abspath(os.path.join(in_script_dir, f"{skeleton_id}.csv"))
            
            if os.path.isfile(connector_filename):
                filtered_connectors = pd.read_csv(connector_filename)
                dist = 0.000002
                calculate_fractal_dimension_all(skeleton_tree, filtered_connectors, dist, undirected=undirected, skeleton_id=skeleton_id)
                print(f"Processed skeleton ID: {skeleton_id}")
                with open("processed_fd_rerun.txt", "a") as f:
                    f.write(f"{skeleton_id}\n")
            else:
                print(f"Connector file not found for skeleton ID: {skeleton_id}")
                with open("missings_fd_rerun.txt", "a") as f:
                    f.write(f"{skeleton_id}\n")
        
        except TimeoutException:
            with open("timeouts_fd_rerun.txt", "a") as f:
                f.write(f"{skeleton_id}\n")
            print("Process killed due to 120 second timeout")
        except Exception as e:
            print(f"Error processing skeleton ID {skeleton_id}: {e}")
            with open("missings_fd_rerun.txt", "a") as f:
                f.write(f"{skeleton_id}\n")
        
        finally:
            # Explicit cleanup after each skeleton
            del skeleton
            del skeleton_tree
            del filtered_connectors
            gc.collect()  # Force garbage collection
            
            # Optionally clear navis cache if it exists
            if hasattr(navis, 'cache'):
                navis.cache.clear()