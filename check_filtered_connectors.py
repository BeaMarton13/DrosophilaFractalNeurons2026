"""For each extrinsic filtered connector, split its skeleton into axon/dendrite and report the neuron's super class."""
import sys
import os
import navis
import pandas as pd

from src.utils.skeleton_tree import SkeletonTree

if sys.platform == "darwin":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    connector_filename = os.path.abspath(os.path.join(script_dir, f"./filtered_connectors/{skeleton_id}.csv"))
    SKELETONS_DIR = None
else:
    # We will call this script_dir for easier usage
    script_dir = "/data/RESULTS/USERS/bea/drosophila/"
    class_dir = "/data/RESULTS/PROJECTS/drosophila/"
    in_script_dir = "/data/RESULTS/PROJECTS/drosophila/filtered_connectors/"
    SKELETONS_DIR = "/data/RESULTS/PROJECTS/drosophila/input_data/full"
    filtered_connectors_ext_filelist = "/data/RESULTS/PROJECTS/drosophila/filtered_connectors/filtered_connectors-GOOD_EXTRINSIC.filelist"
    class_fname_w_path = os.path.abspath(os.path.join(class_dir, f"classification.csv"))


filtered_connectors_ext = [x for x in pd.read_csv(filtered_connectors_ext_filelist, header=None).iloc[:,0].tolist() if ".csv" in x]



df = pd.read_csv(class_fname_w_path)
super_classes_dict = {int(row['root_id']): row['super_class'] for _, row in filtered_df.iterrows()} 

for conncector in filtered_connectors_ext:
    print(conncector, super_classes_dict.get(int(conncector.split(".csv")[0]), "Unknown"))
    connector_filename = os.path.abspath(os.path.join(in_script_dir, f"{conncector}"))
    filtered_connectors = pd.read_csv(connector_filename)
    skeleton_id = int(conncector.split(".csv")[0])

    try:
        skeleton_tree = SkeletonTree.from_properties(SKELETONS_DIR, skeleton_id, skeleton_type="full", undirected=True)
        skeleton_tree.skeleton._set_connectors(filtered_connectors)
    
        split = navis.split_axon_dendrite(skeleton_tree.skeleton, metric='synapse_flow_centrality', reroot_soma=True, cellbodyfiber="soma")
        dendrite_skeleton = split[(split.compartment == 'dendrite')][0]
        axon_skeleton = split[(split.compartment == 'axon')][0]
    except Exception as e:
        print(f"Error processing skeleton ID {skeleton_id}: {e}")
        continue
    