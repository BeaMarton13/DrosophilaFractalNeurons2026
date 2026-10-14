"""Plot one property against another (2D pairwise scatter) across the whole forest."""
import sys
import argparse
import os

from compare_properties import CompareProperties, generate_savefig_path
from axon_dendrite_tree import AxonDendriteForest


if __name__ == "__main__":
    # NOTE: Change the directed/undirected cases, while experimenting
    tree_type = "undirected"
    parser = argparse.ArgumentParser(description="Plot pairwise properties.")
    parser.add_argument("split_type", type=str, help="Split type (navis or mb)")
    parser.add_argument("property_x", type=str, help="Property to plot on X-axis")
    parser.add_argument("property_y", type=str, nargs='+', help="Property/properties to plot on Y-axis")
    parser.add_argument("--axon-or-dendrite", type=str, default="both", help="Axon, Dendrite, or both (default: both)")

    args = parser.parse_args()

    split_type = args.split_type
    property_x = args.property_x
    property_y_list = args.property_y  # now a list
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

    property_to_start_end = {
        "fractal_dimension": (1.0, None),
        "num_nodes": (None, None),
        "num_edges": (None, None),
        "num_leaf_nodes": (None, None),
        # "num_leaf_nodes": (0, 5000),
        "height": (None, None),
        # "height": (0, None),
        "max_width": (None, None),
        # "max_width": (10, 5000),
        "cable_length": (None, None),
        "num_filtered_pre_synapses": (None, None),
        "num_filtered_post_synapses": (None, None),
        "c_value": (None, None)
    }

    property_to_log_scale = {
        "fractal_dimension": (False, False),
        "height": (False, True),
        "max_width": (False, True),
        "cable_length": (False, True),
        "num_edges": (False, True),
        "num_nodes": (False, True),
        "num_leaf_nodes": (False, True),
    }

    # Loop through each property_y
    for property_y in property_y_list:
        print(f"Plotting {property_x} vs {property_y}")

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
        foldername = "figures_pairwise_different_properties"

        if sys.platform == "darwin":
            script_dir = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.abspath(os.path.join(script_dir, f"../../{foldername}"))
        else:
            res_dir = "/data/RESULTS/USERS/bea/drosophila/"
            save_dir = os.path.abspath(os.path.join(res_dir, f"{foldername}"))

        comparator.plot_pairwise_different_properties(
            property_x,
            property_y,
            property_to_name[property_x],
            property_to_name[property_y],
            save_path=generate_savefig_path(property_x, tree_type, 'pairwise_cut_new', save_dir, html=False, property_name2=property_y, axon_or_dendrite=axon_or_dendrite),
            log_scale_x=log_scale_x,
            log_scale_y=log_scale_y,
            x_min=property_to_start_end[property_x][0],
            x_max=property_to_start_end[property_x][1],
            axon_or_dendrite=axon_or_dendrite,
            super_classes=super_classes,
            forest=forest,
            interactive=False
        )
