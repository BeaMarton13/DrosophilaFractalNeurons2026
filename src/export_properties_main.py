import navis
import json
import os
import pandas as pd
import sys
from pathlib import Path

from src.utils.skeleton_tree import SkeletonTree
from src.utils.common import NpEncoder, add_synapse_properties


if sys.platform == "darwin":
    script_dir = os.path.dirname(os.path.abspath(__file__))
else:
    # We will call this script_dir for easier usage
    script_dir = "/data/RESULTS/USERS/bea/drosophila/"


def generate_directory_structure(split_type="navis"):
    """Create the output directory tree for tree properties, c-values and input skeletons."""
    if sys.platform == "darwin":
        os.makedirs(os.path.abspath(os.path.join(script_dir, f"../data_{split_type}/undirected/tree_properties")), exist_ok=True)
        os.makedirs(os.path.abspath(os.path.join(script_dir, f"../data_{split_type}/undirected/c_values")), exist_ok=True)
        os.makedirs(os.path.abspath(os.path.join(script_dir, f"../data_{split_type}/directed/tree_properties")), exist_ok=True)
        os.makedirs(os.path.abspath(os.path.join(script_dir, f"../data_{split_type}/directed/c_values")), exist_ok=True)
        os.makedirs(os.path.abspath(os.path.join(script_dir, "../input_data")), exist_ok=True)
        os.makedirs(os.path.abspath(os.path.join(script_dir, "../input_data/axon")), exist_ok=True)
        os.makedirs(os.path.abspath(os.path.join(script_dir, "../input_data/dendrite")), exist_ok=True)
    else:
        os.makedirs(os.path.abspath(os.path.join("/data/RESULTS/USERS/bea/drosophila/", f"data_{split_type}/undirected/tree_properties")), exist_ok=True)
        os.makedirs(os.path.abspath(os.path.join("/data/RESULTS/USERS/bea/drosophila/", f"data_{split_type}/undirected/c_values")), exist_ok=True)
        os.makedirs(os.path.abspath(os.path.join("/data/RESULTS/USERS/bea/drosophila/", f"data_{split_type}/directed/tree_properties")), exist_ok=True)
        os.makedirs(os.path.abspath(os.path.join("/data/RESULTS/USERS/bea/drosophila/", f"data_{split_type}/directed/c_values")), exist_ok=True)
        os.makedirs(os.path.abspath(os.path.join("/data/RESULTS/PROJECTS/drosophila/", "input_data")), exist_ok=True)
        os.makedirs(os.path.abspath(os.path.join("/data/RESULTS/PROJECTS/drosophila/", "input_data/axon")), exist_ok=True)
        os.makedirs(os.path.abspath(os.path.join("/data/RESULTS/PROJECTS/drosophila/", "input_data/dendrite")), exist_ok=True)

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


def process_skeleton(skeleton_id, undirected, skeletons_dir, axon_dendrite_node_ids, split_type):
    """Load a skeleton, split it into axon/dendrite, and export properties for all three parts."""
    skeleton_tree = SkeletonTree(skeletons_dir, skeleton_id, skeleton_type="full", undirected=undirected)

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

        if not isinstance(axon_skeleton, navis.TreeNeuron):
            axon_skeleton = axon_skeleton[0]
        if not isinstance(dendrite_skeleton, navis.TreeNeuron):
            dendrite_skeleton = dendrite_skeleton[0]


        add_synapse_properties(skeleton_tree)
        properties(skeleton_tree, undirected, filename=f"{skeleton_id}_full.json", split_type=split_type)


        # Axon skeleton
        skeleton_tree_axon = SkeletonTree.from_skeleton(skeleton=axon_skeleton, skeleton_type="axon", undirected=undirected, scaled_coords=skeleton_tree.scaled_coords)
        add_synapse_properties(skeleton_tree_axon)
        properties(skeleton_tree_axon, undirected, filename=f"{skeleton_id}_axon.json", split_type=split_type)

        # Dendrite skeleton
        skeleton_tree_dendrite = SkeletonTree.from_skeleton(skeleton=dendrite_skeleton, skeleton_type="dendrite", undirected=undirected, scaled_coords=skeleton_tree.scaled_coords)
        add_synapse_properties(skeleton_tree_dendrite)
        properties(skeleton_tree_dendrite, undirected, filename=f"{skeleton_id}_dendrite.json", split_type=split_type)