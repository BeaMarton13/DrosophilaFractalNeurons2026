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
        class_fname_w_path = os.path.abspath(os.path.join(script_dir, f"../../data/classification.csv"))
    else:
        res_dir = "/data/RESULTS/USERS/bea/drosophila/"
        class_dir = "/data/RESULTS/PROJECTS/drosophila/"
        tree_dir = os.path.abspath(os.path.join(res_dir, f"data_navis/{tree_type}/tree_properties/"))
        c_dir = os.path.abspath(os.path.join(res_dir, f"data_navis/{tree_type}/c_values/"))
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
        # "fractal_dimension": (1.0, None),
        # "num_nodes": (None, None),
        # "num_edges": (None, None),
        # "num_leaf_nodes": (0, 5000),
        # "height": (0, None),
        # "max_width": (10, 5000),
        # "cable_length": (None, None),
        # "num_filtered_pre_synapses": (None, None),
        # "num_filtered_post_synapses": (None, None),
        # "c_value": (None, None)
        "fractal_dimension": (None, None),
        "num_nodes": (None, None),
        "num_edges": (None, None),
        "num_leaf_nodes": (None, None),
        "height": (None, None),
        "max_width": (None, None),
        "cable_length": (None, None),
        # "num_filtered_pre_synapses": (None, None),
        # "num_filtered_post_synapses": (None, None),
        # "c_value": (None, None)
    }
    

    property_to_log_scale = {
        "fractal_dimension": (False, False),
        "height": (True, False),
        "max_width": (True, False),
        "num_nodes": (True, False),
        "num_edges": (True, False),
        "cable_length": (False, False),
        # "cable_length": (True, False),
        "num_leaf_nodes": (True, False),
    }

    properties = ['fractal_dimension', 'num_nodes', 'num_edges', 'num_leaf_nodes', 'height', 
                  'max_width', 'cable_length', 'num_filtered_pre_synapses', 
                  'num_filtered_post_synapses', 'c_value']

    properties = ['fractal_dimension', 'num_nodes', 'num_edges', 'num_leaf_nodes', 'height', 
                  'max_width', 'cable_length']
    
    for property in properties:
        plot_type = "histogram" # pairwise, histogram



        log_scale_x = property_to_log_scale[property][0]
        log_scale_y = property_to_log_scale[property][1]
        # if log_scale_x and log_scale_y:
        #     foldername = "figures_log_scale_xy"
        # elif log_scale_x:
        #     foldername = "figures_log_scale_x"
        # elif log_scale_y:
        #     foldername = "figures_log_scale_y"
        # else:
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

        foldername = "figures"

        if sys.platform == "darwin":
            script_dir = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.abspath(os.path.join(script_dir, f"../../{foldername}"))
        else:
            res_dir = "/data/RESULTS/USERS/bea/drosophila/"
            save_dir = os.path.abspath(os.path.join(res_dir, f"{foldername}"))


        if plot_type == "pairwise":
            comparator.plot_pairwise(property, property, property_to_name[property], property_to_name[property], save_path=generate_savefig_path(property, tree_type, 'pairwise_cut_new', save_dir, html=False), log_scale_x=log_scale_x, log_scale_y=log_scale_y, x_min=property_to_start_end[property][0], x_max=property_to_start_end[property][1], super_classes=super_classes)  
        elif plot_type == "histogram":
            full_props = forest.get_property_list(property, tree_type='full')
            axon_props = forest.get_property_list(property, tree_type='axon')
            dendrite_props = forest.get_property_list(property, tree_type='dendrite')
            # print("None values in axon_props:", sum(1 for x in axon_props if x is None), len(axon_props)) 
            # print("NaN count in axon_props:", sum(1 for x in axon_props if np.isnan(x)))
            # print("Sample values:", axon_props[:10])
            # Calculate means

            full_mean = np.mean([x for x in full_props if x is not None])
            print(f'Full mean {property}: {full_mean}')
            axon_mean = np.mean([x for x in axon_props if x is not None])
            dendrite_mean = np.mean([x for x in dendrite_props if x is not None])
            
            print(f'Axon mean {property}: {axon_mean}')
            print(f'Dendrite mean {property}: {dendrite_mean}')
            comparator.plot_histogram(property,
                                    property_to_name[property], 
                                    tree_type='both', 
                                    bins=700, 
                                    save_path=generate_savefig_path(property, tree_type, 'histogram_cut_new', save_dir, html=False), 
                                    log_scale_x=log_scale_x, 
                                    log_scale_y=log_scale_y, 
                                    x_min=property_to_start_end[property][0], 
                                    x_max=property_to_start_end[property][1],
                                    super_classes=super_classes,
                                    filter_by_fd=True
                                    )
        else:
            print(f"{plot_type} not found...")
            
