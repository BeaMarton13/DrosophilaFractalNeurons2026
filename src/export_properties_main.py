import navis
import json
import os
import numpy as np

from fafbseg import flywire

from src.utils.skeleton_tree import SkeletonTree
from src.utils.synapses import Synapses


script_dir = os.path.dirname(os.path.abspath(__file__))


# Re-define the custom encoder
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return int(obj) if isinstance(obj, np.integer) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def generate_directory_structure():
    """Generate the directory structure for saving properties."""
    os.makedirs(os.path.abspath(os.path.join(script_dir, "../data/undirected/tree_properties")), exist_ok=True)
    os.makedirs(os.path.abspath(os.path.join(script_dir, "../data/undirected/c_values")), exist_ok=True)
    os.makedirs(os.path.abspath(os.path.join(script_dir, "../data/directed/tree_properties")), exist_ok=True)
    os.makedirs(os.path.abspath(os.path.join(script_dir, "../data/directed/c_values")), exist_ok=True)
    os.makedirs(os.path.abspath(os.path.join(script_dir, "../input_data")), exist_ok=True)
    os.makedirs(os.path.abspath(os.path.join(script_dir, "../input_data/axon")), exist_ok=True)
    os.makedirs(os.path.abspath(os.path.join(script_dir, "../input_data/dendrite")), exist_ok=True)

def properties(skeleton_tree: SkeletonTree, undirected, filename=None):
    if filename is not None:
        if undirected:
            file_with_path = os.path.abspath(os.path.join(script_dir, f"../data/undirected/tree_properties/{filename}"))
        else:
            file_with_path = os.path.abspath(os.path.join(script_dir, f"../data/directed/tree_properties/{filename}"))

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
            file_with_path = os.path.join(script_dir, f"../data/undirected/c_values/{filename}")
        else:
            file_with_path = os.path.join(script_dir, f"../data/directed/c_values/{filename}")
        # Save properties to a JSON file
        with open(file_with_path, 'w') as fp:
            json.dump(skeleton_tree.c_value_properties, fp)
            print(f"Properties saved to {file_with_path}")
    else:
        print("\n--- C Values ---")
        for key, value in skeleton_tree.c_value_properties.items():
            print(f"{key}: {value}")


def add_synapse_properties(skeleton_tree_p: SkeletonTree):
    synapses = Synapses(skeleton_tree_p)
    skeleton_tree_p.add_property("num_pre_synapses", len(synapses.pre_synapses))
    skeleton_tree_p.add_property("num_post_synapses", len(synapses.post_synapses))
    skeleton_tree_p.add_property("num_pre_synapses_unique", len(set(synapses.pre_synapses)))
    skeleton_tree_p.add_property("num_post_synapses_unique", len(set(synapses.post_synapses)))
    skeleton_tree_p.add_property("num_filtered_pre_synapses", len(synapses.filtered_pre_synapses))
    skeleton_tree_p.add_property("num_filtered_post_synapses", len(synapses.filtered_post_synapses))
    skeleton_tree_p.add_property("num_filtered_pre_synapses_unique", len(set(synapses.filtered_pre_synapses)))
    skeleton_tree_p.add_property("num_filtered_post_synapses_unique", len(set(synapses.filtered_post_synapses)))

def process_skeleton(skeleton_id, undirected, skeletons_dir):
    skeleton_tree = SkeletonTree(skeletons_dir, skeleton_id, skeleton_type="full", undirected=undirected)

    # NOTE we need this function to get the synapses and to navis.split_axon_dendrite work
    flywire.get_synapses(skeleton_tree.skeleton, attach=True, neuropils=True, materialization=783)
    split = navis.split_axon_dendrite(skeleton_tree.skeleton, metric='synapse_flow_centrality', reroot_soma=True, cellbodyfiber="soma")

    # Full skeleton
    skeleton_tree.add_property("num_axons", len(split[(split.compartment == 'axon')]))
    skeleton_tree.add_property("num_dendrites", len(split[(split.compartment == 'dendrite')]))
    add_synapse_properties(skeleton_tree)
    properties(skeleton_tree, undirected, filename=f"{skeleton_id}_full.json")

    # Split skeleton
    dendrite_skeleton = split[(split.compartment == 'dendrite')][0]
    axon_skeleton = split[(split.compartment == 'axon')][0]

    # Axon skeleton
    skeleton_tree_axon = SkeletonTree.from_skeleton(skeleton=axon_skeleton, skeleton_type="axon", undirected=undirected)
    # skeleton_tree_axon = SkeletonTree(skeletons_dir=skeletons_dir, skeleton_id=skeleton_id, undirected=undirected, skeleton=axon_skeleton)
    add_synapse_properties(skeleton_tree_axon)
    properties(skeleton_tree_axon, undirected, filename=f"{skeleton_id}_axon.json")

    # Dendrite skeleton
    skeleton_tree_dendrite = SkeletonTree.from_skeleton(skeleton=dendrite_skeleton, skeleton_type="dendrite", undirected=undirected)
    # skeleton_tree_dendrite = SkeletonTree(skeletons_dir=skeletons_dir, skeleton_id=skeleton_id, undirected=undirected, skeleton=dendrite_skeleton)
    add_synapse_properties(skeleton_tree_dendrite)
    properties(skeleton_tree_dendrite, undirected, filename=f"{skeleton_id}_dendrite.json")