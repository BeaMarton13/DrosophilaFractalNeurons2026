import os
import sys
import pandas as pd
# from add_new_property_main import properties
import navis

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
else:
    res_dir = "/data/RESULTS/USERS/bea/drosophila/"
    PLOT_DIR = os.path.abspath(os.path.join(res_dir, f"data/fractal_dimensions/"))

def generate_savefig_path(tree_type, save_dir, html=False):
    os.makedirs(save_dir, exist_ok=True)
    if html:
        filename = f"{tree_type}.html"
    else:
        filename = f"{tree_type}.png"
    return os.path.join(save_dir, filename)


def filter_out_high_fractal_dimension(trees, tree_type):
    trees_ret = []
    for tree in trees:
        if tree.fractal_dimension is not None and tree.fractal_dimension > 2.5:
            print(tree_type, tree.fractal_dimension, tree.id)
            trees_ret.append(tree)
    print(len(trees_ret))
    return trees_ret


def handle_skeleton(skeleton_id):
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
        split = navis.split_axon_dendrite(skeleton_tree.skeleton, metric='synapse_flow_centrality', reroot_soma=True, cellbodyfiber="soma")

        dendrite_skeleton = split[(split.compartment == 'dendrite')][0]
        axon_skeleton = split[(split.compartment == 'axon')][0]

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
    # NOTE: Change the directed/undirected cases, while experimenting
    tree_type = "undirected"  # or "undirected"


    if sys.platform == "darwin":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tree_dir = os.path.abspath(os.path.join(script_dir, f"../../data/{tree_type}/tree_properties/"))
        c_dir = os.path.abspath(os.path.join(script_dir, f"../../data/{tree_type}/c_values/"))
    else:
        res_dir = "/data/RESULTS/USERS/bea/drosophila/"
        tree_dir = os.path.abspath(os.path.join(res_dir, f"data/{tree_type}/tree_properties/"))
        c_dir = os.path.abspath(os.path.join(res_dir, f"data/{tree_type}/c_values/"))

    forest = AxonDendriteForest.build_forest_from_directory(tree_dir, c_dir, undirected=tree_type=='undirected')

    axons = filter_out_high_fractal_dimension(forest.axon_tree, 'axon')
    dendrites = filter_out_high_fractal_dimension(forest.dendrite_tree, 'dendrite')

    for tree in axons:
        handle_skeleton(tree.id)
        print(f"Tree fractal dimension: {tree.fractal_dimension}, id: {tree.id}")

    for tree in dendrites:
        handle_skeleton(tree.id)


    
