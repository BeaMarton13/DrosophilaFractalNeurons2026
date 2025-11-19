import os
import sys
import time
import json
import gc
import multiprocessing
from tqdm import tqdm
import pandas as pd
import numpy as np
import math

# --- Import your modules ---
import navis
from src.utils.synapses import Synapses
from src.export_properties_main import generate_directory_structure
from src.utils.fractal_dimension import fractal_dimension_sparse, plot
from src.utils.FractalDimension import fractal_dimension

# ===============================
# Configuration
# ===============================
NUM_CPUS = 64
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

        self.fractal_dimension = self._calculate_fractal_dimension(self.scaled_coords)
        self.fractal_dimension_new = self._fractal_dimension_new(self.scaled_coords)
        print(">>>>>>>> ", self.fractal_dimension, self.fractal_dimension_new)

    def _read_skeleton(self, skeletons_dir):
        return navis.read_swc(os.path.join(skeletons_dir, f"{self.skeleton_id}.swc"))

   
    def calculate_scaled_coords(self, coords):
        mins = coords.min(axis=0, keepdims=True)
        maxs = coords.max(axis=0, keepdims=True)
        shifted = coords - mins
        scale = (maxs - mins).max()
        return shifted / scale
   
    # def _calculate_fractal_dimension(self, scaled_coords, with_plot=False):
    #     self.coords = self.skeleton.nodes[['x', 'y', 'z']].values
    #     # from sklearn.preprocessing import MinMaxScaler
    #     # scaler = MinMaxScaler()
    #     if scaled_coords is None:
    #         self.scaled_coords = self.calculate_scaled_coords(self.coords)
    #     distances, additional_points = self._calculate_scaled_lengths(self.scaled_coords)
    #     self.max_dist = max(distances)
    #     self.avg_dist = sum(distances) / len(distances)
    #     print("KHM ", len(self.calculate_scaled_coords(self.coords)), len(additional_points))
    #     final_points = list(self.calculate_scaled_coords(self.coords)) + additional_points
    #     print("ERR?")
    #     coeffs, _, _ = fractal_dimension_sparse(final_points, 0.001)
    #     # coeffs, _, _ = fractal_dimension_sparse(self.calculate_scaled_coords(self.coords) + additional_points, self.max_dist)
    #     print("ERR???", coeffs[0], self.skeleton_id, self.skeleton_type)
    #     return coeffs[0]

    def _calculate_fractal_dimension(self, scaled_coords, fname=None):
        self.coords = self.skeleton.nodes[['x', 'y', 'z']].values
        # from sklearn.preprocessing import MinMaxScaler
        # scaler = MinMaxScaler()
        if scaled_coords is None:
            self.scaled_coords = self.calculate_scaled_coords(self.coords)
        distances, additional_points = self._calculate_scaled_lengths(self.scaled_coords)
        self.max_dist = max(distances)
        self.avg_dist = sum(distances) / len(distances)
        final_points = list(self.calculate_scaled_coords(self.coords)) + additional_points
        
        
        if fname is not None:
            coeffs, sizes, counts = fractal_dimension_sparse(final_points, self.max_dist + 0.001)
            plot(counts, sizes, coeffs, fname=fname)
            # plot(counts, sizes, coeffs, fname=f"fractal_dimension{self.skeleton_type}.png")
        else:
            # coeffs, _, _ = fractal_dimension_sparse(self.calculate_scaled_coords(self.coords), self.max_dist)

            coeffs, _, _ = fractal_dimension_sparse(final_points, self.max_dist + 0.001)
        # coeffs, _, _ = fractal_dimension_sparse(self.calculate_scaled_coords(self.coords) + additional_points, self.max_dist)
        return coeffs[0]
    
    def _fractal_dimension_new(self, scaled_coords=None):
        self.coords = self.skeleton.nodes[['x', 'y', 'z']].values
        if scaled_coords is None:
            self.scaled_coords = self.calculate_scaled_coords(self.coords)
        distances, _ = self._calculate_scaled_lengths(self.scaled_coords)
        self.max_dist = max(distances)
        print(self.scaled_coords)
        fd = fractal_dimension(self.scaled_coords, min_box_size=-math.log2(self.max_dist), max_box_size=1, n_samples=20, n_offsets=0, plot=True)
        return fd

    def _calculate_scaled_lengths(self, scaled_points):
        distances = []
        additional_points = []
        for idx, parent in zip(self.skeleton.nodes['node_id'], self.skeleton.nodes['parent_id']):
            if parent != -1:
                additional_points.extend(self.interpolate_3d(scaled_points[parent - 1], scaled_points[idx - 1], distance=0.0002))
                distances.append(np.linalg.norm(scaled_points[parent - 1] - scaled_points[idx - 1]))
        return distances, additional_points
    
    def interpolate_3d(self, point1, point2, distance):
        """
        Interpolates points between two 3D points (x, y, z) so that
        the distance between consecutive points is <= `distance`.

        Args:
            point1: tuple (x1, y1, z1)
            point2: tuple (x2, y2, z2)
            distance: float, maximum allowed spacing between points

        Returns:
            list of tuples: [(x, y, z), ..., (x_end, y_end, z_end)]
        """
        x1, y1, z1 = point1
        x2, y2, z2 = point2

        # Compute total distance between points
        total_dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

        # If already close enough, just return endpoints
        if total_dist <= distance:
            return [point1, point2]

        # Number of segments
        num_segments = int(math.ceil(total_dist / distance))

        # Linear interpolation for each coordinate
        points = [
            (
                x1 + (x2 - x1) * i / num_segments,
                y1 + (y2 - y1) * i / num_segments,
                z1 + (z2 - z1) * i / num_segments
            )
            for i in range(num_segments + 1)
        ]

        return points


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
        print("Updated properties saved to", file_with_path)

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
    # try:
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


        with open("distances.txt", "a") as f_full:
            f_full.write(f"{skeleton_tree.max_dist} - {skeleton_tree.skeleton_id}\n")

        with open("avg_distances.txt", "a") as f_full:
            f_full.write(f"{skeleton_tree.max_dist - skeleton_tree.avg_dist} - {skeleton_tree.avg_dist} - {skeleton_tree.skeleton_id}\n")

        with open("filtered_skeleton_ids(03).txt", "a") as f_full:
            if skeleton_tree.max_dist <= 0.3:
                f_full.write(f"{skeleton_tree.skeleton_id}\n")

    # except Exception as e:
    #     print(e)

    # finally:
    #     # Clean up large objects
    #     del filtered_connectors
    #     del skeleton_tree
    #     del skeleton_tree_axon
    #     del skeleton_tree_dendrite
    #     del axon_skeleton
    #     del dendrite_skeleton
    #     gc.collect()

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
    # ids = [int(x.split('.swc')[0]) for x in os.listdir(SKELETONS_DIR) if x.endswith('.swc')]

    ids = [720575940627756304, 720575940625149966, 720575940611083955, 720575940613378986, 720575940652963830, 720575940629382730, 720575940634024599, 720575940632747404]
    ids = [720575940627624847, 720575940617911104, 720575940630197701, 720575940642026203]
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
