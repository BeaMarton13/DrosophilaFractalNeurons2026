import os
import sys
from pathlib import Path

# Disable internal parallelism to prevent subprocess spawning
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import navis
import numpy as np
import json
import argparse

import signal
import gc

navis.set_pbars(hide=True)  # Disable progress bars
navis.config.parallel = False  # Disable navis parallelism if available

from src.utils.skeleton_tree import SkeletonTree
from src.utils.fractal_dimension import fractal_dimension_sparse, plot
from src.utils.common import NpEncoder, add_synapse_properties


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Process exceeded 120 seconds")

def calculate_fractal_dimension(skeleton_id, skeleton_tree_full, split, dist, skeleton_type="full"):
    if skeleton_type == "dendrite":
        skeleton_navis = split[(split.compartment == 'dendrite')][0]
    elif skeleton_type == "axon":
        skeleton_navis = split[(split.compartment == 'axon')][0]
    else:
        skeleton_navis = skeleton_tree_full.skeleton
    
    try:
        skeleton = SkeletonTree.from_properties(SKELETONS_DIR, skeleton_id, skeleton_type=skeleton_type, undirected=True, skeleton=skeleton_navis)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading properties for {skeleton_id}_{skeleton_type}: {e}")
        # If properties file is corrupt, recreate from skeleton
        skeleton = SkeletonTree.from_skeleton(skeleton=skeleton_navis, skeleton_type=skeleton_type, undirected=True, scaled_coords=skeleton_tree_full.scaled_coords)
    
    fd = skeleton._calculate_fractal_dimension(skeleton_tree_full.scaled_coords, recalculate=True)
    if fd < 1:
        additional_points = get_additional_points(skeleton, distance=dist, scaled_coords=skeleton_tree_full.scaled_coords)
        coeffs, sizes, counts = fractal_dimension_sparse(additional_points, max_dist=dist, recalculate=True)
        fd = coeffs[0]
    if hasattr(skeleton, 'max_dist'):
        update_property(skeleton_id, fd, undirected=True, skeleton_type=skeleton_type, max_dist=skeleton.max_dist)
    else:
        update_property(skeleton_id, fd, undirected=True, skeleton_type=skeleton_type, max_dist=None)


def calculate_fractal_dimension_all(skeleton_tree, filtered_connectors, dist):
    skeleton_tree.skeleton._set_connectors(filtered_connectors)
    split = navis.split_axon_dendrite(skeleton_tree.skeleton, metric='synapse_flow_centrality', reroot_soma=True, cellbodyfiber="soma")
    fd = skeleton_tree._calculate_fractal_dimension(None, recalculate=True)
    if hasattr(skeleton_tree, 'max_dist'):
        update_property(skeleton_tree.id, fd, undirected=True, skeleton_type="full", max_dist=skeleton_tree.max_dist)
    else:
        update_property(skeleton_id, fd, undirected=True, skeleton_type="full", max_dist=None)

    calculate_fractal_dimension(skeleton_tree.id, skeleton_tree, split, dist, skeleton_type="axon")
    calculate_fractal_dimension(skeleton_tree.id, skeleton_tree, split, dist, skeleton_type="dendrite")


def get_additional_points(skeleton, distance, scaled_coords, num_segments=None):
    additional_points = []
    for idx, parent in zip(skeleton.skeleton.nodes['node_id'], skeleton.skeleton.nodes['parent_id']):
        if parent != -1:
            additional_points.extend(skeleton.interpolate_3d(scaled_coords[parent - 1], scaled_coords[idx - 1], distance=distance, num_segments=num_segments))
    return np.array(additional_points)

def update_property(skeleton_id, fractal_dimension, undirected=True, skeleton_type="full", max_dist=None):
    if undirected:
        properties_dir = "/data/RESULTS/USERS/bea/drosophila/data_navis/undirected/tree_properties/"
    else:
        properties_dir = "/data/RESULTS/USERS/bea/drosophila/data_navis/directed/tree_properties/"
    properties_path = os.path.join(properties_dir, f"{skeleton_id}_{skeleton_type}.json")
    if not os.path.exists(properties_path):
        with open("missings_all.txt", "a") as f:
            f.write(f"{skeleton_id}\n")
        raise FileNotFoundError(f"Properties file not found: {properties_path}")
        
    properties = json.load(open(properties_path, 'r'))
    properties['fractal_dimension'] = fractal_dimension
    if max_dist is not None:
        properties['max_dist'] = max_dist
    # print(properties_path, properties)
    with open(properties_path, 'w') as f:
        json.dump(properties, f, cls=NpEncoder, indent=4)

def properties(skeleton_tree: SkeletonTree, undirected, filename=None, split_type="navis"):
    if filename is not None:
        if undirected:
            dir_val = "undirected"
        else:
            dir_val = "directed"

        if sys.platform == "darwin":
            folder = os.path.abspath(os.path.join(script_dir, f"../data_{split_type}/{dir_val}/tree_properties/"))
            file_with_path = os.path.abspath(os.path.join(script_dir, f"../data_{split_type}/{dir_val}/tree_properties/{filename}"))
        else:
            folder = os.path.abspath(os.path.join(script_dir, f"data_{split_type}/{dir_val}/tree_properties/"))
            file_with_path = os.path.abspath(os.path.join(script_dir, f"data_{split_type}/{dir_val}/tree_properties/{filename}"))
        Path(folder).mkdir(parents=True, exist_ok=True)
        try:
            with open(file_with_path, 'w') as fp:
                json.dump(skeleton_tree.tree_properties, fp, cls=NpEncoder, indent=4)
                print("Saved properties to", file_with_path)
            # print("Data successfully written to output.json.")
        except TypeError as e:
            print(f"Error: {e}")
    else:
        print("--- Tree Properties ---")
        for key, value in skeleton_tree.tree_properties.items():
            print(f"{key}: {value}")

    if filename is not None:

        if undirected:
            dir_val = "undirected"
        else:
            dir_val = "directed"
        if sys.platform == "darwin":
            folder = os.path.abspath(os.path.join(script_dir, f"../data_{split_type}/{dir_val}/c_values/"))
            file_with_path = os.path.abspath(os.path.join(script_dir, f"../data_{split_type}/{dir_val}/c_values/{filename}"))
        else:
            folder = os.path.abspath(os.path.join(script_dir, f"data_{split_type}/{dir_val}/c_values/"))
            file_with_path = os.path.abspath(os.path.join(script_dir, f"data_{split_type}/{dir_val}/c_values/{filename}"))
        Path(folder).mkdir(parents=True, exist_ok=True)
        
        # Save properties to a JSON file
        with open(file_with_path, 'w') as fp:
            json.dump(skeleton_tree.c_value_properties, fp)
            # print(f"Properties saved to {file_with_path}")
    else:
        print("\n--- C Values ---")
        for key, value in skeleton_tree.c_value_properties.items():
            print(f"{key}: {value}")


def process_skeleton(skeleton_id, undirected, skeletons_dir, axon_dendrite_node_ids, split_type, with_timeout=False):
    if with_timeout:
        # Set up the timeout signal
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(120)  # 120 seconds timeout

    try:
        skeleton_tree = SkeletonTree(skeletons_dir, skeleton_id, skeleton_type="full", undirected=undirected)

        # NOTE we need this function to get the synapses and to navis.split_axon_dendrite work
        # flywire.get_synapses(skeleton_tree.skeleton, attach=True, neuropils=True, materialization=783)



        if sys.platform == "darwin":
            script_dir = os.path.dirname(os.path.abspath(__file__))
            connector_filename = os.path.abspath(os.path.join(script_dir, f"../filtered_connectors/{skeleton_id}.csv"))
        else:
            # We will call this script_dir for easier usage
            script_dir = "/data/RESULTS/USERS/bea/drosophila/"
            in_script_dir = "/data/RESULTS/PROJECTS/drosophila/filtered_connectors/"
            connector_filename = os.path.abspath(os.path.join(in_script_dir, f"{skeleton_id}.csv"))

        if os.path.isfile(connector_filename):
            filtered_connectors = pd.read_csv(connector_filename)
            skeleton_tree.skeleton._set_connectors(filtered_connectors)

            fragments = navis.break_fragments(skeleton_tree.skeleton)
            number_of_nodes_in_fragments = []
            if len(fragments) > 1:
                for f in fragments:
                    skeleton_tree_axon = SkeletonTree.from_skeleton(skeleton=f, skeleton_type="full", undirected=undirected, scaled_coords=None)
                    number_of_nodes_in_fragments.append(skeleton_tree_axon.num_nodes)

            selected_fragment_idx = np.argmax(number_of_nodes_in_fragments) if len(number_of_nodes_in_fragments) > 0 else 0
            skeleton_tree = SkeletonTree.from_skeleton(skeleton=fragments[selected_fragment_idx], skeleton_type="full", undirected=undirected, scaled_coords=None)
            
            if axon_dendrite_node_ids is None:
                split = navis.split_axon_dendrite(skeleton_tree.skeleton, metric='synapse_flow_centrality', reroot_soma=True, cellbodyfiber="soma")
                # Full skeleton
                skeleton_tree.add_property("num_axons", len(split[(split.compartment == 'axon')]))
                skeleton_tree.add_property("num_dendrites", len(split[(split.compartment == 'dendrite')]))
                # Split skeleton
                dendrite_skeleton = split[(split.compartment == 'dendrite')][0]
                axon_skeleton = split[(split.compartment == 'axon')][0]
            else:
                from src.utils.split_axon_dendrite import split
                dendrite_skeleton = split(skeleton_tree.skeleton, axon_dendrite_node_ids.loc[axon_dendrite_node_ids['root_id'] == skeleton_tree.skeleton_id, 'dendrite_idx'].values[0])
                axon_skeleton = split(skeleton_tree.skeleton, axon_dendrite_node_ids.loc[axon_dendrite_node_ids['root_id'] == skeleton_tree.skeleton_id, 'axonend_idx'].values[0])

            # if not isinstance(axon_skeleton, navis.TreeNeuron):
            #     axon_skeleton = axon_skeleton[0]
            # if not isinstance(dendrite_skeleton, navis.TreeNeuron):
            #     dendrite_skeleton = dendrite_skeleton[0]


            add_synapse_properties(skeleton_tree)
            properties(skeleton_tree, undirected, filename=f"{skeleton_id}_full.json", split_type=split_type)


            
            fragments = navis.break_fragments(axon_skeleton)
            number_of_nodes_in_fragments = []
            if len(fragments) > 1:
                for f in fragments:
                    skeleton_tree_axon = SkeletonTree.from_skeleton(skeleton=f, skeleton_type="axon", undirected=undirected, scaled_coords=skeleton_tree.scaled_coords)
                    number_of_nodes_in_fragments.append(skeleton_tree_axon.num_nodes)

            selected_fragment_idx = np.argmax(number_of_nodes_in_fragments) if len(number_of_nodes_in_fragments) > 0 else 0
            print(f"[axon] fragment node counts: {number_of_nodes_in_fragments}, selected: {selected_fragment_idx}")
            skeleton_tree_axon = SkeletonTree.from_skeleton(skeleton=fragments[selected_fragment_idx], skeleton_type="axon", undirected=undirected, scaled_coords=skeleton_tree.scaled_coords)
            add_synapse_properties(skeleton_tree_axon)
            properties(skeleton_tree_axon, undirected, filename=f"{skeleton_id}_axon.json", split_type=split_type)

            # Dendrite skeleton
            fragments = navis.break_fragments(dendrite_skeleton)
            number_of_nodes_in_fragments = []
            if len(fragments) > 1:
                for f in fragments:
                    skeleton_tree_dendrite = SkeletonTree.from_skeleton(skeleton=f, skeleton_type="dendrite", undirected=undirected, scaled_coords=skeleton_tree.scaled_coords)
                    number_of_nodes_in_fragments.append(skeleton_tree_dendrite.num_nodes)

            selected_fragment_idx = np.argmax(number_of_nodes_in_fragments) if len(number_of_nodes_in_fragments) > 0 else 0
            print(f"[dendrite] fragment node counts: {number_of_nodes_in_fragments}, selected: {selected_fragment_idx}")
            skeleton_tree_dendrite = SkeletonTree.from_skeleton(skeleton=fragments[selected_fragment_idx], skeleton_type="dendrite", undirected=undirected, scaled_coords=skeleton_tree.scaled_coords)

            add_synapse_properties(skeleton_tree_dendrite)
            properties(skeleton_tree_dendrite, undirected, filename=f"{skeleton_id}_dendrite.json", split_type=split_type)

            del skeleton_tree_axon
            del skeleton_tree_dendrite
            del skeleton_tree
            del filtered_connectors
            gc.collect()  # Force garbage collection
            return True
        
    except TimeoutException:
        with open("timeouts_all.txt", "a") as f:
            f.write(f"{skeleton_id}\n")
        print("Process killed due to 120 second timeout")
        return None
    except Exception as e:
        signal.alarm(0)  # Cancel alarm on error
        raise e

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
    if sys.platform == "darwin":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path_to_skeletons = "./input_data/full"
        SKELETONS_DIR = os.path.join(script_dir, path_to_skeletons)
    else:
        # We will call this script_dir for easier usage
        script_dir = "/data/RESULTS/USERS/bea/drosophila/"
        in_script_dir = "/data/RESULTS/PROJECTS/drosophila/input_data/full"
        SKELETONS_DIR = in_script_dir


    if not os.path.isfile("processed_all.txt"):
        processed_ids = []
    else:
        with open("processed_all.txt", "r") as f:
            processed_ids = [int(line.strip()) for line in f if line.strip()]

    if not os.path.isfile("missings_all.txt"):
        missing_ids = []
    else:
        with open("missings_all.txt", "r") as f:
            missing_ids = [int(line.strip()) for line in f if line.strip()]

    if not os.path.isfile("timeouts_all.txt"):
        timeout_ids = []
    else:
        with open("timeouts_all.txt", "r") as f:
            timeout_ids = [int(line.strip()) for line in f if line.strip()]
    
    # NOTE: currently reprocesses whatever previously timed out (see timeout_ids above)
    ids = timeout_ids
    with open("unprocessed_all.txt", "w") as f:
        for s_id in ids:
            f.write(f"{s_id}\n")

    if start + length <= len(ids):
        ids = ids[start:start+length]
    else:
        ids = ids[start:]

    for skeleton_id in ids:
        if sys.platform == "darwin":
            script_dir = os.path.dirname(os.path.abspath(__file__))
            connector_filename = os.path.abspath(os.path.join(script_dir, f"../filtered_connectors/{skeleton_id}.csv"))
        else:
            script_dir = "/data/RESULTS/USERS/bea/drosophila/"
            in_script_dir = "/data/RESULTS/PROJECTS/drosophila/filtered_connectors/"
            connector_filename = os.path.abspath(os.path.join(in_script_dir, f"{skeleton_id}.csv"))
        
        if os.path.isfile(connector_filename):
            filtered_connectors = None
            skeleton_tree = None
            
            try:
                filtered_connectors = pd.read_csv(connector_filename)
                
                # Process skeleton with timeout
                result = process_skeleton(skeleton_id, True, SKELETONS_DIR, None, split_type="navis")
                
                if result is None:  # Timeout occurred
                    continue
                
                # Calculate fractal dimension
                dist = 0.000002
                skeleton_tree = SkeletonTree.from_properties(SKELETONS_DIR, skeleton_id, skeleton_type="full", undirected=undirected)
                calculate_fractal_dimension_all(skeleton_tree, filtered_connectors, dist)
                
                print(f"Processed skeleton ID: {skeleton_id}")
                with open("processed_all.txt", "a") as f:
                    f.write(f"{skeleton_id}\n")
                    
            except Exception as e:
                print(f"Error processing skeleton ID {skeleton_id}: {e}")
                with open("missings_all.txt", "a") as f:
                    f.write(f"{skeleton_id}\n")
            finally:
                # Explicit cleanup
                del filtered_connectors
                del skeleton_tree
                gc.collect()  # Force garbage collection