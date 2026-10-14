"""Build the axon/dendrite/full skeleton forest and pull out the trees whose
fractal dimension is below 1, saving their IDs to axon.txt/dendrite.txt/full.txt."""

import os
import sys
import pandas as pd
import navis
import argparse

from src.utils.axon_dendrite_tree import AxonDendriteForest
from src.utils.skeleton_tree import SkeletonTree


if sys.platform == "darwin":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path_to_skeletons = "./input_data/full"
    SKELETONS_DIR = os.path.join(script_dir, path_to_skeletons)
else:
    # We will call this script_dir for easier usage
    script_dir = "/data/RESULTS/USERS/bea/drosophila/"
    in_script_dir = "/data/RESULTS/PROJECTS/drosophila/input_data/full"
    SKELETONS_DIR = in_script_dir


if sys.platform == "darwin":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    PLOT_DIR = os.path.abspath(os.path.join(script_dir, f"../../data/fractal_dimensions/"))
    class_fname_w_path = os.path.abspath(os.path.join(script_dir, f"../../data/classification.csv"))
else:
    res_dir = "/data/RESULTS/USERS/bea/drosophila/"
    class_dir = "/data/RESULTS/PROJECTS/drosophila/"
    PLOT_DIR = os.path.abspath(os.path.join(res_dir, f"data/fractal_dimensions/"))
    class_fname_w_path = os.path.abspath(os.path.join(class_dir, f"classification.csv"))

def generate_savefig_path(tree_type, save_dir, html=False):
    """Build the output path for a tree-type's fractal-dimension plot, creating save_dir if needed."""
    os.makedirs(save_dir, exist_ok=True)
    if html:
        filename = f"{tree_type}.html"
    else:
        filename = f"{tree_type}.png"
    return os.path.join(save_dir, filename)


def filter_out_high_fractal_dimension(trees, tree_type):
    """Return the trees (of the given type) with a saved connector file and fractal_dimension < 1."""
    trees_ret = []
    for tree in trees:

        if sys.platform == "darwin":
            script_dir = os.path.dirname(os.path.abspath(__file__))
            connector_filename = os.path.abspath(os.path.join(script_dir, f"../filtered_connectors/{tree.id}.csv"))
        else:
            # We will call this script_dir for easier usage
            script_dir = "/data/RESULTS/USERS/bea/drosophila/"
            in_script_dir = "/data/RESULTS/PROJECTS/drosophila/filtered_connectors/"
            connector_filename = os.path.abspath(os.path.join(in_script_dir, f"{tree.id}.csv"))
        if os.path.isfile(connector_filename):
            if tree.fractal_dimension is not None and tree.fractal_dimension < 1:
                print(tree_type, tree.fractal_dimension, tree.id, tree.tree_type)
                trees_ret.append(tree)
    print(len(trees_ret))
    return trees_ret


def handle_skeleton(skeleton_id, axon_dendrite_node_ids):
    """Load a skeleton, split it into axon/dendrite, and plot the fractal dimension curve for each part."""
    # NOTE: currently unreachable from __main__ below (only called after an unconditional exit()).
    undirected = True
    
    skeleton_tree = SkeletonTree(SKELETONS_DIR, skeleton_id, skeleton_type="full", undirected=undirected, with_exit=False)
    skeleton_tree._calculate_fractal_dimension(scaled_coords=skeleton_tree.scaled_coords, fname=generate_savefig_path("full", f"{PLOT_DIR}{skeleton_id}/", html=False))

    
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

        # Force single process for navis
        with navis.config.set_temp({'max_cores': 1}):
            if axon_dendrite_node_ids is None:
                split = navis.split_axon_dendrite(skeleton_tree.skeleton, 
                                                metric='synapse_flow_centrality', 
                                                reroot_soma=True, 
                                                cellbodyfiber="soma")
                dendrite_skeleton = split[(split.compartment == 'dendrite')][0]
                axon_skeleton = split[(split.compartment == 'axon')][0]
            else:
                from src.utils.split_axon_dendrite import split
                dendrite_skeleton = split(skeleton_tree.skeleton, axon_dendrite_node_ids.loc[axon_dendrite_node_ids['root_id'] == skeleton_tree.skeleton_id, 'dendrite_idx'].values)
                axon_skeleton = split(skeleton_tree.skeleton, axon_dendrite_node_ids.loc[axon_dendrite_node_ids['root_id'] == skeleton_tree.skeleton_id, 'axonend_idx'].values)

        if not isinstance(axon_skeleton, navis.TreeNeuron):
            axon_skeleton = axon_skeleton[0]
        if not isinstance(dendrite_skeleton, navis.TreeNeuron):
            dendrite_skeleton = dendrite_skeleton[0]

        # Axon skeleton
        skeleton_tree_axon = SkeletonTree.from_skeleton(skeleton=axon_skeleton, skeleton_type="axon", undirected=undirected, scaled_coords=skeleton_tree.scaled_coords, with_exit=False)
        skeleton_tree_axon._calculate_fractal_dimension(skeleton_tree.scaled_coords, fname=generate_savefig_path("axon", f"{PLOT_DIR}{skeleton_id}/", html=False))

        # Dendrite skeleton
        skeleton_tree_dendrite = SkeletonTree.from_skeleton(skeleton=dendrite_skeleton, skeleton_type="dendrite", undirected=undirected, scaled_coords=skeleton_tree.scaled_coords, with_exit=False)
        skeleton_tree_dendrite._calculate_fractal_dimension(skeleton_tree.scaled_coords, fname=generate_savefig_path("dendrite", f"{PLOT_DIR}{skeleton_id}/", html=False))

        print(f"Full w id {skeleton_tree.skeleton_id} fractal dimension: {skeleton_tree.fractal_dimension}, max dist: {skeleton_tree.max_dist}")
        print(f"Axon w id {skeleton_tree_axon.skeleton_id} fractal dimension: {skeleton_tree_axon.fractal_dimension}, max dist: {skeleton_tree_axon.max_dist}")
        print(f"Dendrite w id {skeleton_tree_dendrite.skeleton_id} fractal dimension: {skeleton_tree_dendrite.fractal_dimension}, max dist: {skeleton_tree_dendrite.max_dist}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add a new property to the saved properties.")
    parser.add_argument("split_type", type=str, help="Split type (navis or mb)")
    args = parser.parse_args()
    if sys.platform == "darwin":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        split_filename = os.path.abspath(os.path.join(script_dir, f"./data/calculate_dendrite_axonend_distances-MIN.csv"))
    else:
        # We will call this script_dir for easier usage
        script_dir = "/data/RESULTS/USERS/bea/drosophila/"
        class_dir = "/data/RESULTS/PROJECTS/drosophila/"
        split_filename = "/data/RESULTS/PROJECTS/drosophila/calculate_dendrite_axonend_distances-MIN/calculate_dendrite_axonend_distances-MIN.csv"
        class_fname_w_path = os.path.abspath(os.path.join(class_dir, f"classification.csv"))
    # Read graph
    split_type = args.split_type
    axon_dendrite_node_ids = None
    if split_type != "navis":
        axon_dendrite_node_ids = pd.read_csv(split_filename)


    # NOTE: Change the directed/undirected cases, while experimenting
    tree_type = "undirected"  # or "undirected"


    if sys.platform == "darwin":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tree_dir = os.path.abspath(os.path.join(script_dir, f"../../data/{tree_type}/tree_properties/"))
        c_dir = os.path.abspath(os.path.join(script_dir, f"../../data/{tree_type}/c_values/"))
        class_fname_w_path = os.path.abspath(os.path.join(script_dir, f"../../data/classification.csv"))
    else:
        res_dir = "/data/RESULTS/USERS/bea/drosophila/"
        tree_dir = os.path.abspath(os.path.join(res_dir, f"data_{split_type}/{tree_type}/tree_properties/"))
        c_dir = os.path.abspath(os.path.join(res_dir, f"data_{split_type}/{tree_type}/c_values/"))
    
    print(tree_dir)
    forest = AxonDendriteForest.build_forest_from_directory(tree_dir, c_dir, undirected=tree_type=='undirected', class_fname_w_path=class_fname_w_path)

    axons = filter_out_high_fractal_dimension(forest.axon_tree, 'axon')
    dendrites = filter_out_high_fractal_dimension(forest.dendrite_tree, 'dendrite')
    all = filter_out_high_fractal_dimension(forest.full_tree, 'full')
    print("ALL FRACTAL DIMENSION < 1 IDs: ", [t.id for t in all if t.fractal_dimension < 1.0])
    print("AXON & DENDRITE FRACTAL DIMENSION < 1 IDs: ", set([t.id for t in axons if t.fractal_dimension < 1.0]).intersection(set([t.id for t in dendrites if t.fractal_dimension < 1.0])))
    print("ALL & AXON & DENDRITE FRACTAL DIMENSION < 1 IDs: ", set(set([t.id for t in axons if t.fractal_dimension < 1.0]).intersection(set([t.id for t in dendrites if t.fractal_dimension < 1.0]))).intersection(set([t.id for t in all if t.fractal_dimension < 1.0])))
    
    # Save IDs to files
    with open("axon.txt", "w") as f:
        f.write("\n".join(str(t.id) for t in axons))
    
    with open("dendrite.txt", "w") as f:
        f.write("\n".join(str(t.id) for t in dendrites))
    
    with open("full.txt", "w") as f:
        f.write("\n".join(str(t.id) for t in all))
        
    exit()
