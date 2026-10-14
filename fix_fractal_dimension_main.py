import os
import sys
import pandas as pd
import navis
import numpy as np
import json

from src.utils.skeleton_tree import SkeletonTree
from src.utils.fractal_dimension import fractal_dimension_sparse, plot

"""Recompute fractal dimension for a fixed list of skeletons using an
interpolated-point sampling of each tree's edges (see get_additional_points),
and write the corrected value back into the saved properties JSON.

Sample log output from an earlier run (kept for reference):
    DENDRITE Tree ID: 720575940629694330, Fractal Dimension: 0.5832--- Old Fractal Dimension: 0.2782
    DENDRITE Tree ID: 720575940659237761, Fractal Dimension: 0.6740--- Old Fractal Dimension: 0.6527
    DENDRITE Tree ID: 720575940659743105, Fractal Dimension: 0.6109--- Old Fractal Dimension: 0.5719
    DENDRITE Tree ID: 720575940650901881, Fractal Dimension: 0.6528--- Old Fractal Dimension: 0.5471
    DENDRITE Tree ID: 720575940643548468, Fractal Dimension: 0.6517--- Old Fractal Dimension: 0.6782
    DENDRITE Tree ID: 720575940639872061, Fractal Dimension: 0.6849--- Old Fractal Dimension: 0.4754
    DENDRITE Tree ID: 720575940639252799, Fractal Dimension: 0.4710--- Old Fractal Dimension: 0.3785
    DENDRITE Tree ID: 720575940638987864, Fractal Dimension: 0.5576--- Old Fractal Dimension: 0.4233
    DENDRITE Tree ID: 720575940638329151, Fractal Dimension: nan--- Old Fractal Dimension: 0.0000
"""




def get_additional_points(skeleton, distance, scaled_coords, num_segments=None):
    """Interpolate extra points along each parent-child edge of the skeleton."""
    additional_points = []
    for idx, parent in zip(skeleton.skeleton.nodes['node_id'], skeleton.skeleton.nodes['parent_id']):
        if parent != -1:
            additional_points.extend(skeleton.interpolate_3d(scaled_coords[parent - 1], scaled_coords[idx - 1], distance=distance, num_segments=num_segments))
    return np.array(additional_points)

def update_property(skeleton_id, fractal_dimension, undirected=True, skeleton_type="full"):
    if undirected:
        properties_dir = "/data/RESULTS/USERS/bea/drosophila/data_navis/undirected/tree_properties/"
    else:
        properties_dir = "/data/RESULTS/USERS/bea/drosophila/data_navis/directed/tree_properties/"
    properties_path = os.path.join(properties_dir, f"{skeleton_id}_{skeleton_type}.json")
    if not os.path.exists(properties_path):
        raise FileNotFoundError(f"Properties file not found: {properties_path}")
    properties = json.load(open(properties_path, 'r'))
    properties['fractal_dimension'] = fractal_dimension
    with open(properties_path, 'w') as f:
        json.dump(properties, f, indent=4)

if __name__ == "__main__":
    # NOTE: Change the directed/undirected cases, while experimenting
    tree_type = "undirected"  # or "undirected"


    if sys.platform == "darwin":
        tree_dir = os.path.abspath(os.path.join(script_dir, f"../../data/{tree_type}/tree_properties/"))
        c_dir = os.path.abspath(os.path.join(script_dir, f"../../data/{tree_type}/c_values/"))
    else:
        res_dir = "/data/RESULTS/USERS/bea/drosophila/"
        tree_dir = os.path.abspath(os.path.join(res_dir, f"data_navis/{tree_type}/tree_properties/"))
        c_dir = os.path.abspath(os.path.join(res_dir, f"data_navis/{tree_type}/c_values/"))

    undirected = True if tree_type == "undirected" else False

    with open("axon.txt", "r") as f:
        axon_ids = [int(line.strip()) for line in f if line.strip()]
    with open("dendrite.txt", "r") as f:
        dendrite_ids = [int(line.strip()) for line in f if line.strip()]
    with open("full.txt", "r") as f:
        full_ids = [int(line.strip()) for line in f if line.strip()]
    
    with open("processed_fd.txt", "r") as f:
        processed_ids = [int(line.strip()) for line in f if line.strip()]

    ids = axon_ids + dendrite_ids + full_ids
    print("Total skeletons:", len(set(ids)))
    print("Processed skeletons:", len(set(processed_ids)))
    ids = [s_id for s_id in ids if s_id not in processed_ids]
    print(len(set(ids)), "skeletons to process.")

    # NOTE: this hardcoded list overrides the computed `ids` above.
    ids = [720575940612319125, 720575940620500273, 720575940626098705, 720575940624162328]

    if sys.platform == "darwin":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path_to_skeletons = "./input_data/full"
        SKELETONS_DIR = os.path.join(script_dir, path_to_skeletons)
    else:
        # We will call this script_dir for easier usage
        script_dir = "/data/RESULTS/USERS/bea/drosophila/"
        in_script_dir = "/data/RESULTS/PROJECTS/drosophila/input_data/full"
        SKELETONS_DIR = in_script_dir

    for skeleton_id in ids:
        skeleton_tree = SkeletonTree.from_properties(SKELETONS_DIR, skeleton_id, skeleton_type="full", undirected=undirected)
        if sys.platform == "darwin":
            script_dir = os.path.dirname(os.path.abspath(__file__))
            connector_filename = os.path.abspath(os.path.join(script_dir, f"../filtered_connectors/{skeleton_id}.csv"))
        else:
            # We will call this script_dir for easier usage
            script_dir = "/data/RESULTS/USERS/bea/drosophila/"
            in_script_dir = "/data/RESULTS/PROJECTS/drosophila/filtered_connectors/"
            connector_filename = os.path.abspath(os.path.join(in_script_dir, f"{skeleton_id}.csv"))
        
        if os.path.isfile(connector_filename):
            dist = 0.000002
            filtered_connectors = pd.read_csv(connector_filename)
            skeleton_tree.skeleton._set_connectors(filtered_connectors)
            split = navis.split_axon_dendrite(skeleton_tree.skeleton, metric='synapse_flow_centrality', reroot_soma=True, cellbodyfiber="soma")
            skeleton_tree._calculate_fractal_dimension(None)
            if skeleton_id in dendrite_ids:
                dendrite_skeleton_navis = split[(split.compartment == 'dendrite')][0]
                dendrite_skeleton = SkeletonTree.from_properties(SKELETONS_DIR, skeleton_id, skeleton_type="dendrite", undirected=undirected, skeleton=dendrite_skeleton_navis)
                fd_orig = dendrite_skeleton.fractal_dimension
                additional_points = get_additional_points(dendrite_skeleton, distance=dist, scaled_coords=skeleton_tree.scaled_coords)
                coeffs, sizes, counts = fractal_dimension_sparse(additional_points, max_dist=dist, recalculate=True)
                fd = coeffs[0]
                plot(counts, sizes, coeffs, fname=f'./temp_new{skeleton_id}_dendrite.png')
                fd = dendrite_skeleton._calculate_fractal_dimension(skeleton_tree.scaled_coords, recalculate=True)
                # fd = dendrite_skeleton._calculate_fractal_dimension(skeleton_tree.scaled_coords, recalculate=True)
                # fd = dendrite_skeleton._calculate_fractal_dimension(skeleton_tree.scaled_coords, fname=f'./temp{skeleton_id}_dendrite.png')
                with open("log_fd.txt", "a") as log_f:
                    log_f.write(f"DENDRITE Tree ID: {skeleton_id}, Fractal Dimension: {fd:.4f}--- Old Fractal Dimension: {fd_orig:.4f}\n")
                update_property(skeleton_id, fd, undirected=undirected, skeleton_type="dendrite")
            elif skeleton_id in axon_ids:
                axon_skeleton_navis = split[(split.compartment == 'axon')][0]
                axon_skeleton = SkeletonTree.from_properties(SKELETONS_DIR, skeleton_id, skeleton_type="axon", undirected=undirected, skeleton=axon_skeleton_navis)
                fd_orig = axon_skeleton.fractal_dimension
                additional_points = get_additional_points(axon_skeleton, distance=dist, scaled_coords=skeleton_tree.scaled_coords)
                coeffs, sizes, counts = fractal_dimension_sparse(additional_points, max_dist=dist, recalculate=True)
                fd = coeffs[0]
                plot(counts, sizes, coeffs, fname=f'./temp_new{skeleton_id}_axon.png')
                # fd = axon_skeleton._calculate_fractal_dimension(skeleton_tree.scaled_coords, recalculate=True)
                # fd = axon_skeleton._calculate_fractal_dimension(skeleton_tree.scaled_coords, fname=f'./temp{skeleton_id}_axon.png')
                with open("log_fd.txt", "a") as log_f:
                    log_f.write(f"AXON Tree ID: {skeleton_id}, Fractal Dimension: {fd:.4f}--- Old Fractal Dimension: {fd_orig:.4f}\n")
                update_property(skeleton_id, fd, undirected=undirected, skeleton_type="axon")
            elif skeleton_id in full_ids:
                fd_orig = skeleton_tree.fractal_dimension
                additional_points = get_additional_points(skeleton_tree, distance=dist, scaled_coords=skeleton_tree.scaled_coords)
                coeffs, sizes, counts = fractal_dimension_sparse(additional_points, max_dist=dist, recalculate=True)
                fd = coeffs[0]
                plot(counts, sizes, coeffs, fname=f'./temp_new{skeleton_id}_full.png')
                # fd = skeleton_tree._calculate_fractal_dimension(skeleton_tree.scaled_coords, recalculate=True)
                # fd = skeleton_tree._calculate_fractal_dimension(skeleton_tree.scaled_coords, fname=f'./temp{skeleton_id}_full.png')
                with open("log_fd.txt", "a") as log_f:
                    log_f.write(f"FULL Tree ID: {skeleton_id}, Fractal Dimension: {fd:.4f}--- Old Fractal Dimension: {fd_orig:.4f}\n")
                update_property(skeleton_id, fd, undirected=undirected, skeleton_type="full")
   
            with open("processed_fd.txt", "a") as f:
                f.write(f"{skeleton_id}\n")