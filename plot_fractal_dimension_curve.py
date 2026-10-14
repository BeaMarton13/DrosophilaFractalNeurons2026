"""Plot fractal-dimension curves for a fixed set of skeletons, splitting each
into axon/dendrite and saving one plot per part."""

import sys
import os
import pandas as pd
import navis
import argparse

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
    PLOT_DIR = os.path.abspath(os.path.join(script_dir, f"data/fractal_dimensions/"))
else:
    res_dir = "/data/RESULTS/USERS/bea/drosophila/"
    PLOT_DIR = os.path.abspath(os.path.join(res_dir, f"data/fractal_dimensions/"))

def generate_savefig_path(tree_type, save_dir, html=False):
    """Build the output path for a tree-type's fractal-dimension plot, creating save_dir if needed."""
    os.makedirs(save_dir, exist_ok=True)
    if html:
        filename = f"{tree_type}.html"
    else:
        filename = f"{tree_type}.png"
    return os.path.join(save_dir, filename)

def handle_skeleton(skeleton_id, axon_dendrite_node_ids):
    """Load a skeleton, split it into axon/dendrite, and plot the fractal dimension curve for each part."""
    undirected = True

    skeleton_tree = SkeletonTree(SKELETONS_DIR, skeleton_id, skeleton_type="full", undirected=undirected, with_exit=False)
    skeleton_tree._calculate_fractal_dimension(scaled_coords=skeleton_tree.scaled_coords, fname=generate_savefig_path("full", f"{PLOT_DIR}/{skeleton_id}/", html=False))

    
    if sys.platform == "darwin":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        connector_filename = os.path.abspath(os.path.join(script_dir, f"./filtered_connectors/{skeleton_id}.csv"))
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
        skeleton_tree_axon._calculate_fractal_dimension(skeleton_tree.scaled_coords, fname=generate_savefig_path("axon", f"{PLOT_DIR}/{skeleton_id}/", html=False))

        # Dendrite skeleton
        skeleton_tree_dendrite = SkeletonTree.from_skeleton(skeleton=dendrite_skeleton, skeleton_type="dendrite", undirected=undirected, scaled_coords=skeleton_tree.scaled_coords, with_exit=False)
        skeleton_tree_dendrite._calculate_fractal_dimension(skeleton_tree.scaled_coords, fname=generate_savefig_path("dendrite", f"{PLOT_DIR}/{skeleton_id}/", html=False))

        print(f"Full w id {skeleton_tree.skeleton_id} fractal dimension: {skeleton_tree.fractal_dimension}, max dist: {skeleton_tree.max_dist}")
        print(f"Axon w id {skeleton_tree_axon.skeleton_id} fractal dimension: {skeleton_tree_axon.fractal_dimension}, max dist: {skeleton_tree_axon.max_dist}")
        print(f"Dendrite w id {skeleton_tree_dendrite.skeleton_id} fractal dimension: {skeleton_tree_dendrite.fractal_dimension}, max dist: {skeleton_tree_dendrite.max_dist}")
    return skeleton_tree.skeleton


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
        split_filename = "/data/RESULTS/PROJECTS/drosophila/calculate_dendrite_axonend_distances-MIN/calculate_dendrite_axonend_distances-MIN.csv"
    # Read graph
    split_type = args.split_type
    axon_dendrite_node_ids = None
    if split_type != "navis":
        axon_dendrite_node_ids = pd.read_csv(split_filename)

    # axons
    ids = [720575940604195360, 720575940604654048, 720575940606327072, 720575940607557554, 720575940608370610, 720575940608994652]
    for id in ids:
        skeleton = handle_skeleton(id, axon_dendrite_node_ids)

        split = navis.split_axon_dendrite(skeleton, metric='synapse_flow_centrality', reroot_soma=True, cellbodyfiber="soma")
        split.plot3d(color=split.color).show()
