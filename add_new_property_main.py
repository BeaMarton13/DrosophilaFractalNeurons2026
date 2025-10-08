import os
import sys
import time
import json
import gc
import multiprocessing
from tqdm import tqdm
import psutil
import pandas as pd
import numpy as np

# --- Import your modules ---
import navis
from fafbseg import flywire
from src.utils.synapses import Synapses
from src.utils.plot import plot_tree_digraph
from src.export_properties_main import generate_directory_structure
from src.utils.fractal_dimension import fractal_dimension_sparse, plot

# ===============================
# Configuration
# ===============================
NUM_CPUS = 4
TIMEOUT = 120  # seconds per skeleton

if sys.platform == "darwin":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    SKELETONS_DIR = os.path.join(SCRIPT_DIR, "./input_data/full")
else:
    SCRIPT_DIR = "/data/RESULTS/USERS/bea/drosophila/"
    SKELETONS_DIR = "/data/RESULTS/PROJECTS/drosophila/input_data/full"

# ===============================
# Custom JSON encoder
# ===============================
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return int(obj) if isinstance(obj, np.integer) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

# ===============================
# SkeletonTree class
# ===============================
class SkeletonTree:
    def __init__(self, skeletons_dir, skeleton_id, skeleton_type, undirected=False, skeleton=None, scaled_coords=None):
        self.skeleton_id = skeleton_id
        self.skeleton_type = skeleton_type
        self.undirected = undirected
        self.scaled_coords = scaled_coords

        if skeleton is None:
            self.skeleton = self._read_skeleton(skeletons_dir)
        else:
            self.skeleton = skeleton

        self.fractal_dimension = self._calculate_fractal_dimension(self.scaled_coords, with_plot=False)

    def _read_skeleton(self, skeletons_dir):
        return navis.read_swc(os.path.join(skeletons_dir, f"{self.skeleton_id}.swc"))

    def _calculate_fractal_dimension(self, scaled_coords, with_plot=False):
        self.coords = self.skeleton.nodes[['x', 'y', 'z']].values
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        if scaled_coords is None:
            self.scaled_coords = scaler.fit_transform(self.coords)
        distances = self._calculate_scaled_lengths(self.scaled_coords)
        coeffs, _, _ = fractal_dimension_sparse(scaler.fit_transform(self.coords), 2 * max(distances))
        return coeffs[0]

    def _calculate_scaled_lengths(self, scaled_points):
        distances = []
        for idx, parent in zip(self.skeleton.nodes['node_id'], self.skeleton.nodes['parent_id']):
            if parent != -1:
                distances.append(np.linalg.norm(scaled_points[parent - 1] - scaled_points[idx - 1]))
        return distances

    @classmethod
    def from_skeleton(cls, skeleton, skeleton_type, undirected, scaled_coords=None):
        return cls("dummy_value", skeleton.id, skeleton_type=skeleton_type, skeleton=skeleton, undirected=undirected, scaled_coords=scaled_coords)

# ===============================
# Functions to handle skeleton properties
# ===============================
def properties(skeleton_tree: SkeletonTree, undirected, filename):
    if undirected:
        dir_val = "undirected"
    else:
        dir_val = "directed"

    file_with_path = os.path.join(SCRIPT_DIR, f"data/{dir_val}/tree_properties/{filename}")

    try:
        with open(file_with_path, 'r') as fp:
            tree_properties = json.load(fp)
    except (FileNotFoundError, json.JSONDecodeError):
        tree_properties = {}

    tree_properties['fractal_dimension'] = skeleton_tree.fractal_dimension

    with open(file_with_path, 'w') as fp:
        json.dump(tree_properties, fp, cls=NpEncoder, indent=4)

def add_synapse_properties(skeleton_tree: SkeletonTree):
    return
    synapses = Synapses(skeleton_tree)
    skeleton_tree.add_property("num_pre_synapses", len(synapses.pre_synapses))
    skeleton_tree.add_property("num_post_synapses", len(synapses.post_synapses))
    skeleton_tree.add_property("num_pre_synapses_unique", len(set(synapses.pre_synapses)))
    skeleton_tree.add_property("num_post_synapses_unique", len(set(synapses.post_synapses)))
    skeleton_tree.add_property("num_filtered_pre_synapses", len(synapses.filtered_pre_synapses))
    skeleton_tree.add_property("num_filtered_post_synapses", len(synapses.filtered_post_synapses))
    skeleton_tree.add_property("num_filtered_pre_synapses_unique", len(set(synapses.filtered_pre_synapses)))
    skeleton_tree.add_property("num_filtered_post_synapses_unique", len(set(synapses.filtered_post_synapses)))

# ===============================
# Process a single skeleton
# ===============================
def process_skeleton_wrapper(skeleton_id, undirected, skeletons_dir):
    try:
        skeleton_tree = SkeletonTree(skeletons_dir, skeleton_id, skeleton_type="full", undirected=undirected)

        # NOTE we need this function to get the synapses and to navis.split_axon_dendrite work
        # flywire.get_synapses(skeleton_tree.skeleton, attach=True, neuropils=True, materialization=783)



        if sys.platform == "darwin":
            script_dir = os.path.dirname(os.path.abspath(__file__))
            connector_filename = os.path.abspath(os.path.join(script_dir, f"./filtered_connectors/{skeleton_id}.csv"))
        else:
            # We will call this script_dir for easier usage
            script_dir = "/data/RESULTS/USERS/bea/drosophila/"
            in_script_dir = "/data/RESULTS/PROJECTS/drosophila/filtered_connectors/"
            connector_filename = os.path.abspath(os.path.join(in_script_dir, f"{skeleton_id}.csv"))

        # print("KKKKKKKKKKKKK ", connector_filename)
        filtered_connectors = None
        # skeleton_tree = None
        skeleton_tree_axon = None
        skeleton_tree_dendrite = None
        axon_skeleton = None
        dendrite_skeleton = None
        if os.path.isfile(connector_filename):
            filtered_connectors = pd.read_csv(connector_filename)
            skeleton_tree.skeleton._set_connectors(filtered_connectors)
            split = navis.split_axon_dendrite(skeleton_tree.skeleton, metric='synapse_flow_centrality', reroot_soma=True, cellbodyfiber="soma")



            # Full skeleton
            properties(skeleton_tree, undirected, filename=f"{skeleton_id}_full.json")

            # Split skeleton
            dendrite_skeleton = split[(split.compartment == 'dendrite')][0]
            axon_skeleton = split[(split.compartment == 'axon')][0]

            # Axon skeleton
            skeleton_tree_axon = SkeletonTree.from_skeleton(skeleton=axon_skeleton, skeleton_type="axon", undirected=undirected, scaled_coords=skeleton_tree.scaled_coords)
            properties(skeleton_tree_axon, undirected, filename=f"{skeleton_id}_axon.json")

            # Dendrite skeleton
            skeleton_tree_dendrite = SkeletonTree.from_skeleton(skeleton=dendrite_skeleton, skeleton_type="dendrite", undirected=undirected, scaled_coords=skeleton_tree.scaled_coords)
            properties(skeleton_tree_dendrite, undirected, filename=f"{skeleton_id}_dendrite.json")

    except Exception as e:
        print(e)

    finally:
        # Clean up large objects
        del filtered_connectors
        del skeleton_tree
        del skeleton_tree_axon
        del skeleton_tree_dendrite
        del axon_skeleton
        del dendrite_skeleton
        gc.collect()

# ===============================
# Parallel execution manager
# ===============================
def run_parallel_skeletons(skeleton_ids, undirected, skeletons_dir, processed_file, num_cpus=NUM_CPUS, timeout=TIMEOUT):
    active_processes = []
    results = []

    if not os.path.exists(processed_file):
        open(processed_file, "w").close()

    with tqdm(total=len(skeleton_ids), desc="Processing skeletons") as pbar:
        for skeleton_id in skeleton_ids:
            p = multiprocessing.Process(target=process_skeleton_wrapper, args=(skeleton_id, undirected, skeletons_dir))
            p.start()
            start_time = time.time()
            active_processes.append((p, skeleton_id, start_time))

            # Maintain NUM_CPUS processes at most
            while len(active_processes) >= num_cpus:
                new_active = []
                for proc, sid, st in active_processes:
                    if proc.is_alive():
                        if time.time() - st > timeout:
                            proc.terminate()
                            proc.join()
                            with open("timeouts.txt", "a") as f:
                                f.write(f"{skeleton_id}\n")
                        else:
                            new_active.append((proc, sid, st))
                    else:
                        proc.join()
                        results.append((sid, True))
                        pbar.update(1)
                        with open(processed_file, "a") as f:
                            f.write(f"{skeleton_id}\n")
                active_processes = new_active
                time.sleep(0.1)

        # Wait for remaining processes
        for proc, sid, st in active_processes:
            proc.join(timeout)
            if proc.is_alive():
                proc.terminate()
                proc.join()
                with open("timeouts.txt", "a") as f:
                    f.write(f"{skeleton_id}\n")
            else:
                results.append((sid, True))
                pbar.update(1)
    return results

# ===============================
# Main execution
# ===============================
if __name__ == "__main__":
    undirected = True
    generate_directory_structure()

    # Load skeleton IDs
    ids = [int(x.split('.swc')[0]) for x in os.listdir(SKELETONS_DIR) if x.endswith('.swc')]

    # Filter already processed skeletons
    processed = set()
    if os.path.exists("processed_fractal_dimension.txt"):
        with open("processed_fractal_dimension.txt", 'r') as file:
            processed = {int(line.strip().lstrip('>')) for line in file}

    remaining_ids = [sid for sid in ids if sid not in processed]
    # remaining_ids = remaining_ids[:500]

    # Remove previous timeouts file
    try:
        os.remove("timeouts.txt")
    except OSError:
        pass

    # Run skeletons in parallel
    results = run_parallel_skeletons(remaining_ids, undirected, SKELETONS_DIR, "processed_fractal_dimension.txt")

    print(f"Processing complete. Successfully processed: {len([r for r in results if r[1]])}/{len(remaining_ids)}")
