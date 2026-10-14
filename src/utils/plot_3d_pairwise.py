"""Plot a 3D pairwise comparison of three tree properties across the whole forest."""
import sys
import argparse
import os

from compare_properties import CompareProperties
from axon_dendrite_tree import AxonDendriteForest


if __name__ == "__main__":
    # NOTE: Change the directed/undirected cases, while experimenting
    tree_type = "undirected"
    parser = argparse.ArgumentParser(description="Plot pairwise properties.")
    parser.add_argument("split_type", type=str, help="Split type (navis or mb)")
    parser.add_argument("property_x", type=str, help="Property to plot on X-axis")
    parser.add_argument("property_y", type=str, help="Property to plot on Y-axis")
    parser.add_argument("property_z", type=str, help="Property to plot on Z-axis")
    parser.add_argument("--axon-or-dendrite", type=str, default="both", help="Axon, Dendrite, or both (default: both)")

    args = parser.parse_args()

    split_type = args.split_type
    property_x = args.property_x
    property_y = args.property_y
    property_z = args.property_z
    axon_or_dendrite = args.axon_or_dendrite.lower()

    if sys.platform == "darwin":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tree_dir = os.path.abspath(os.path.join(script_dir, f"../../data_{split_type}/{tree_type}/tree_properties/"))
        c_dir = os.path.abspath(os.path.join(script_dir, f"../../data_{split_type}/{tree_type}/c_values/"))
        class_fname_w_path = os.path.abspath(os.path.join(script_dir, f"../../data/classification.csv"))
    else:
        class_dir = "/data/RESULTS/PROJECTS/drosophila/"
        res_dir = "/data/RESULTS/USERS/bea/drosophila/"
        tree_dir = os.path.abspath(os.path.join(res_dir, f"data_{split_type}/{tree_type}/tree_properties/"))
        c_dir = os.path.abspath(os.path.join(res_dir, f"data_{split_type}/{tree_type}/c_values/"))
        # tree_dir =  os.path.abspath(os.path.join(res_dir, f"data-INTRINSIC/{tree_type}/tree_properties/"))
        # c_dir = os.path.abspath(os.path.join(res_dir, f"data-INTRINSIC/{tree_type}/c_values/"))
        class_fname_w_path = os.path.abspath(os.path.join(class_dir, f"classification.csv"))

    forest = AxonDendriteForest.build_forest_from_directory(tree_dir, c_dir, undirected=tree_type=='undirected', class_fname_w_path=class_fname_w_path)

    comparator = CompareProperties(forest)

    property_to_name = {
        "fractal_dimension": "Fractal Dimension",
        "num_nodes": "Number of Nodes",
        "num_edges": "Number of Edges",
        "num_leaf_nodes": "Number of Leaf Nodes",
        "height": "Height",
        "max_width": "Maximum Width",
        "cable_length": "Cable Length (nm)",
        "num_filtered_pre_synapses": "Number of Pre-synapses",
        "num_filtered_post_synapses": "Number of Post-synapses",
        "c_value": "C-value"
    }

    property_to_log_scale = {
        "fractal_dimension": (False, False, False),
        "height": (False, False, False),
        "max_width": (False, True, True),
        "cable_length": (False, True, True),
        "num_edges": (False, True, True),
    }

    print(f"Plotting {property_x} vs {property_y} vs {property_z}")

    super_classes = {}
    for tree in forest.axon_tree:
        if tree.super_class not in super_classes.keys():
            super_classes[tree.super_class] = {'axon_props': [], 'dendrite_props': [], 'full_props': []}
        super_classes[tree.super_class]['axon_props'].append(tree.tree_properties[property_y])

    for tree in forest.dendrite_tree:
        if tree.super_class not in super_classes.keys():
            super_classes[tree.super_class] = {'axon_props': [], 'dendrite_props': [], 'full_props': []}
        super_classes[tree.super_class]['dendrite_props'].append(tree.tree_properties[property_y])

    for tree in forest.full_tree:
        if tree.super_class not in super_classes.keys():
            super_classes[tree.super_class] = {'axon_props': [], 'dendrite_props': [], 'full_props': []}
        super_classes[tree.super_class]['full_props'].append(tree.tree_properties[property_y])

    log_scale_x = property_to_log_scale.get(property_x, (False, False))[0]
    log_scale_y = property_to_log_scale.get(property_y, (False, False))[1]
    log_scale_z = property_to_log_scale.get(property_z, (False, False))[2]
    foldername = "figures_pairwise_different_properties3d"

    if sys.platform == "darwin":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.abspath(os.path.join(script_dir, f"../../{foldername}"))
    else:
        res_dir = "/data/RESULTS/USERS/bea/drosophila/"
        save_dir = os.path.abspath(os.path.join(res_dir, f"{foldername}"))

    os.makedirs(save_dir, exist_ok=True)
    comparator.plot_pairwise_3d_properties(
        property_x,
        property_y,
        property_z,
        property_to_name[property_x],
        property_to_name[property_y],
        property_to_name[property_z],
        save_path=os.path.join(
            save_dir,
            f"pairwise_3d_{property_x}_vs_{property_y}_vs_{property_z}_logx_{log_scale_x}_logy_{log_scale_y}_axonordendrite_{axon_or_dendrite}.html"
        ),
        log_scale_x=log_scale_x,
        log_scale_y=log_scale_y,
        log_scale_z=log_scale_z,
        axon_or_dendrite=axon_or_dendrite,
        super_classes=super_classes,
        forest=forest
    )
