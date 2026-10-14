"""Plot property distributions (violin/histogram/pairwise) grouped by super class."""
import os
import sys
import argparse

from compare_properties import CompareProperties, generate_savefig_path
from axon_dendrite_tree import AxonDendriteForest


if __name__ == "__main__":
    # NOTE: Change the directed/undirected cases, while experimenting
    tree_type = "undirected"  # or "undirected"
    parser = argparse.ArgumentParser(description="Add a new property to the saved properties.")
    parser.add_argument("split_type", type=str, help="Split type (navis or mb)")
    args = parser.parse_args()
    split_type = args.split_type

    if sys.platform == "darwin":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tree_dir = os.path.abspath(os.path.join(script_dir, f"../../data_{split_type}/{tree_type}/tree_properties/"))
        c_dir = os.path.abspath(os.path.join(script_dir, f"../../data_{split_type}/{tree_type}/c_values/"))
        class_fname_w_path = os.path.abspath(os.path.join(script_dir, f"../../data/classification.csv"))
    else:
        res_dir = "/data/RESULTS/USERS/bea/drosophila/"
        class_dir = "/data/RESULTS/PROJECTS/drosophila/"
        tree_dir = os.path.abspath(os.path.join(res_dir, f"data_{split_type}/{tree_type}/tree_properties/"))
        c_dir = os.path.abspath(os.path.join(res_dir, f"data_{split_type}/{tree_type}/c_values/"))
        class_fname_w_path = os.path.abspath(os.path.join(class_dir, f"classification.csv"))

    forest = AxonDendriteForest.build_forest_from_directory(tree_dir, c_dir, undirected=tree_type=='undirected', class_fname_w_path=class_fname_w_path)

    comparator = CompareProperties(forest)

    log_scale_x = False
    log_scale_y = False

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
        "fractal_dimension": (None, None),
        "num_nodes": (None, None),
        "num_edges": (None, None),
        "num_leaf_nodes": (0, 5000),
        "height": (0, None),
        "max_width": (0, 200),
        # "max_width": (10, 5000),
        "cable_length": (None, None),
        "num_filtered_pre_synapses": (None, None),
        "num_filtered_post_synapses": (None, None),
        "c_value": (None, None)
    }

    property_to_log = {
        "fractal_dimension": (False, False),
        "num_nodes": (False, True),
        "num_edges": (False, False),
        "num_leaf_nodes": (False, True),
        "height": (False, False),
        "max_width": (False, True),
        "cable_length": (False, False),
        "num_filtered_pre_synapses": (False, True),
        "num_filtered_post_synapses": (False, True),
        "c_value": (False, False)
    }

    properties = ["fractal_dimension", "num_filtered_pre_synapses", "num_filtered_post_synapses", "max_width", "height", "num_leaf_nodes", "num_nodes"]
    plot_type = "violin" # pairwise, histogram
    plot_multiple = True

    for property in properties:

        foldername = "figures_super_classes"

        if sys.platform == "darwin":
            script_dir = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.abspath(os.path.join(script_dir, f"../../{foldername}"))
        else:
            res_dir = "/data/RESULTS/USERS/bea/drosophila/"
            save_dir = os.path.abspath(os.path.join(res_dir, f"{foldername}"))

        super_classes = {}
        for tree in forest.axon_tree:
            if tree.super_class not in super_classes.keys():
                super_classes[tree.super_class] = {'axon_props': [], 'dendrite_props': [], 'full_props': []}
            super_classes[tree.super_class]['axon_props'].append(tree.tree_properties[property])
            if tree.tree_properties[property] is None or tree.tree_properties[property] <= 0.5:
                print("axon", property, tree.id, tree.tree_properties[property], tree.super_class)

        for tree in forest.dendrite_tree:
            if tree.super_class not in super_classes.keys():
                super_classes[tree.super_class] = {'axon_props': [], 'dendrite_props': [], 'full_props': []}
            super_classes[tree.super_class]['dendrite_props'].append(tree.tree_properties[property])
            if tree.tree_properties[property] is None or tree.tree_properties[property] <= 0.5:
                print("dendrite", tree.id, tree.tree_properties[property], tree.super_class)

        for tree in forest.full_tree:
            if tree.super_class not in super_classes.keys():
                super_classes[tree.super_class] = {'axon_props': [], 'dendrite_props': [], 'full_props': []}
            super_classes[tree.super_class]['full_props'].append(tree.tree_properties[property])
            if tree.tree_properties[property] is None or tree.tree_properties[property] <= 0.5:
                print("full", tree.id, tree.tree_properties[property], tree.super_class)

        if plot_type == "pairwise":
            comparator.plot_pairwise(property, property, property_to_name[property], property_to_name[property], save_path=generate_savefig_path(property, tree_type, 'pairwise_cut_new', save_dir, html=False), log_scale_x=log_scale_x, log_scale_y=log_scale_y, x_min=property_to_start_end[property][0], x_max=property_to_start_end[property][1])
        elif plot_type == "histogram":
            comparator.plot_histogram(property,
                                    property_to_name[property],
                                    tree_type='both',
                                    bins=700,
                                    save_path=generate_savefig_path(property, tree_type, f'histogram_cut_new_class_{split_type}', save_dir, html=False),
                                    log_scale_x=property_to_log[property][0],
                                    log_scale_y=property_to_log[property][1],
                                    x_min=property_to_start_end[property][0],
                                    x_max=property_to_start_end[property][1],
                                    super_classes=super_classes,
                                    plot_multiple=plot_multiple
                                    )
        elif plot_type == "violin":
            comparator.plot_multi_class_violin(
                                    property_to_name[property],
                                    bins=700,
                                    save_path=generate_savefig_path(property, tree_type, f'violin_new_class_{split_type}', save_dir, html=False),
                                    log_scale_y=property_to_log[property][1],
                                    y_min=property_to_start_end[property][0],
                                    y_max=property_to_start_end[property][1],
                                    super_classes=super_classes,
                                    plot_multiple=plot_multiple
                                    )
        else:
            print(f"{plot_type} not found...")
