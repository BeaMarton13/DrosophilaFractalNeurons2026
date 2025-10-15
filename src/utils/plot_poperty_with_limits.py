import os
import sys
import numpy as np
import matplotlib.pyplot as pl

from compare_properties import CompareProperties, generate_savefig_path
from axon_dendrite_tree import AxonDendriteForest


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

    comparator = CompareProperties(forest)


    log_scale_x = False
    log_scale_y = False
    if log_scale_x and log_scale_y:
        foldername = "figures_log_scale_xy"
    elif log_scale_x:
        foldername = "figures_log_scale_x"
    elif log_scale_y:
        foldername = "figures_log_scale_y"
    else:
        foldername = "figures"

    if sys.platform == "darwin":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.abspath(os.path.join(script_dir, f"../../{foldername}"))
    else:
        res_dir = "/data/RESULTS/USERS/bea/drosophila/"
        save_dir = os.path.abspath(os.path.join(res_dir, f"{foldername}"))

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
        "num_leaf_nodes": (0, 5000),
        "height": (0, None),
        "max_width": (10, 5000),
        "cable_length": (None, None),
        "num_filtered_pre_synapses": (None, None),
        "num_filtered_post_synapses": (None, None),
        "c_value": (None, None)
    }

    # properties = ['fractal_dimension', 'num_nodes', 'num_edges', 'num_leaf_nodes', 'height', 
    #               'max_width', 'cable_length', 'num_filtered_pre_synapses', 
    #               'num_filtered_post_synapses', 'c_value']
    property = "fractal_dimension"
    plot_type = "pairwise" # pairwise, histogram
    if plot_type == "pairwise":
        comparator.plot_pairwise(property, property, property_to_name[property], property_to_name[property], save_path=generate_savefig_path(property, tree_type, 'pairwise_cut_new', save_dir, html=False), log_scale_x=log_scale_x, log_scale_y=log_scale_y, x_min=0, x_max=5000)  
    elif plot_type == "histogram":
        comparator.plot_histogram(property,
                                property_to_name[property], 
                                tree_type='both', 
                                bins=700, 
                                save_path=generate_savefig_path(property, tree_type, 'histogram_cut_new', save_dir, html=False), 
                                log_scale_x=log_scale_x, 
                                log_scale_y=log_scale_y, 
                                x_min=property_to_start_end[property][0], 
                                x_max=property_to_start_end[property][1]
                                )
    else:
        print(f"{plot_type} not found...")
        
