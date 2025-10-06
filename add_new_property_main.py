import navis
import os
import multiprocessing
import time
import json
import sys
import numpy as np
import psutil
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


from fafbseg import flywire
from tqdm import tqdm

from src.utils.synapses import Synapses
from src.utils.plot import plot_tree_digraph

from src.export_properties_main import generate_directory_structure
from src.utils.fractal_dimension import fractal_dimension_sparse, plot


class SkeletonTree:
    def __init__(
            self, 
            skeletons_dir: str, 
            skeleton_id: int, 
            skeleton_type: str,
            undirected: bool = False, 
            skeleton: navis.Neuron = None, 
            autosave: bool = False,
            scaled_coords: list = None
        ):
        """
        Initializes the SkeletonTree with the given directory and skeleton ID.
        
        Args:
            skeletons_dir (str): Directory where skeleton files are stored.
            skeleton_id (int): ID of the skeleton to prepare.
        """
        self.skeleton_id = skeleton_id
        if skeleton is None:
            # Read skeleton from the specified directory
            self.skeleton = self._read_skeleton(skeletons_dir)
        else:
            self.skeleton = skeleton
        self.skeleton_type = skeleton_type
        self.undirected = undirected

        self.scaled_coords = scaled_coords

        self.fractal_dimension = self._calculate_fractal_dimension(self.scaled_coords, with_plot=True)

    # --- Skeleton Reading and Preparation ---
    def _read_skeleton(self, skeletons_dir: str) -> navis.Neuron:
        """
        Read a skeleton from the specified directory and skeleton ID.
        """
        return navis.read_swc(os.path.join(skeletons_dir, f'{self.skeleton_id}.swc'))

    def _calculate_fractal_dimension(self, scaled_coords, with_plot=False):
        self.coords = self.skeleton.nodes[['x', 'y', 'z']].values
        scaler = MinMaxScaler()
        if scaled_coords is None:
            self.scaled_coords = scaler.fit_transform(self.coords)

        if with_plot:
            coeffs, sizes, counts = fractal_dimension_sparse(scaler.fit_transform(self.coords), 2 * max(self._calculate_scaled_lengths(self.scaled_coords)))
            plot(counts, sizes, coeffs, fname=f"fractal_dimension{self.skeleton_type}.png")

        coeffs, _, _ = fractal_dimension_sparse(scaler.fit_transform(self.coords), 2 * max(self._calculate_scaled_lengths(self.scaled_coords)))
        return coeffs[0]
    
    def _calculate_scaled_lengths(self, scaled_points):
        distances = []
        adjacency = self.skeleton.get_igraph().get_adjacency()
        neighbor_pairs = []
        for idx, lst in enumerate(adjacency):
            try:
                p_idx = next((i for i, x in enumerate(lst) if x), None)
            except:
                p_idx = -1


            neighbor_pairs.append([idx, p_idx])
        for idx, parent in zip(self.skeleton.nodes['node_id'], self.skeleton.nodes['parent_id']):
            if parent != -1:
                distances.append(np.linalg.norm(scaled_points[parent - 1] - scaled_points[idx - 1]))
        return distances
    
    @classmethod
    def from_skeleton(cls, skeleton: navis.Neuron, skeleton_type:str, undirected: bool, autosave: bool = False, scaled_coords: list = None):
        return cls("dummy_value", skeleton.id, skeleton=skeleton, skeleton_type=skeleton_type, undirected=undirected, autosave=autosave, scaled_coords=scaled_coords)



def properties(skeleton_tree: SkeletonTree, undirected, filename):
    if undirected:
        dir_val = "undirected"
    else:
        dir_val = "directed"

    if sys.platform == "darwin":
        file_with_path = os.path.abspath(os.path.join(script_dir, f"./data/{dir_val}/tree_properties/{filename}"))
    else:
        file_with_path = os.path.abspath(os.path.join(script_dir, f"data/{dir_val}/tree_properties/{filename}"))
    try:
        with open(file_with_path, 'r') as fp:
            tree_properties = json.load(fp)

        # NOTE: update here the property name you want to add
        tree_properties['fractal_dimension'] = skeleton_tree.fractal_dimension
        with open(file_with_path, 'w') as fp:
            json.dump(tree_properties, fp, cls=NpEncoder, indent=4)
        # print("Data successfully written to output.json.")
    except TypeError as e:
        print(f"Error: {e}")


def process_skeleton(skeleton_id, undirected, skeletons_dir):
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

    print("KKKKKKKKKKKKK ", connector_filename)
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



# NOTE: File structure: 
    # ./data/{directed or undirected}/{tree_properties or c_values}/{skeleton_id}_full.json
    # ./data/{directed or undirected}/{tree_properties or c_values}/{skeleton_id}_axon.json
    # ./data/{directed or undirected}/{tree_properties or c_values}/{skeleton_id}_dendrite.json
    # ./input_data/skeletons/full/{skeleton_id}.swc
    # ./input_data/skeletons/axon/{skeleton_id}.swc
    # ./input_data/skeletons/dendrite/{skeleton_id}.swc



if sys.platform == "darwin":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path_to_skeletons = "./input_data/full"
    SKELETONS_DIR = os.path.join(script_dir, path_to_skeletons)
else:
    # We will call this script_dir for easier usage
    script_dir = "/data/RESULTS/USERS/bea/drosophila/"
    in_script_dir = "/data/RESULTS/PROJECTS/drosophila/input_data/"
    SKELETONS_DIR = in_script_dir

# path_to_skeletons = "./sk_lod1_783_healed"

# arrays used as indices must be of integer (or boolean) type
# SKELETON_ID = 720575940626286432

# list index out of range:
# SKELETON_ID = 720575940628446888

# SKELETON_ID = 720575940608945163
SKELETON_ID = 720575940661305217

processed = []
if os.path.exists("processed.txt"):
    with open('processed.txt', 'r') as file:
        for line in file:
            if line.startswith(">"):
                processed.append(int(line.strip()[1:]))
            else:
                processed.append(int(line.strip()))


# Re-define the custom encoder
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return int(obj) if isinstance(obj, np.integer) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    

def run_with_timeout(func, timeout, *args, **kwargs):
    """
    Runs a function with a specified timeout.

    Args:
        func (callable): The function to run.
        timeout (int): The maximum time in seconds.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.
    """
    process = multiprocessing.Process(target=func, args=args, kwargs=kwargs)
    process.start()

    # Wait for the process to complete or for the timeout to expire
    process.join(timeout=timeout)

    # Check if the process is still alive after the join call
    if process.is_alive():
        # print(f"Timeout of {timeout} seconds exceeded. Terminating process...")
        with open("timeouts.txt", "a") as f:
            id = kwargs["skeleton_id"]
            f.write(f"{id}\n")
        process.terminate()
        # Clean up the terminated process
        process.join()
        return False  # Indicate that the process was killed
    else:
        # print("Process completed within the timeout.")
        return True # Indicate successful completion



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


def example_usage():
    undirected = True
    skeleton_tree = SkeletonTree(SKELETONS_DIR, SKELETON_ID, skeleton_type="full", undirected=undirected)
    # skeleton_nx = skeleton_tree.skeleton_nx
    print("\n\n\n=================================")
    print("--- Full Skeleton Properties ---")
    print("=================================\n")

    # NOTE we need this function to get the synapses and to navis.split_axon_dendrite works
    flywire.get_synapses(skeleton_tree.skeleton, attach=True, neuropils=True, materialization=783)

    split = navis.split_axon_dendrite(skeleton_tree.skeleton, metric='synapse_flow_centrality', reroot_soma=True, cellbodyfiber="soma")
    
    print(split.compartment, split.compartment == 'axon')
    dendrite_skeleton = split[(split.compartment == 'dendrite')][0]
    axon_skeleton = split[(split.compartment == 'axon')][0]

    skeleton_tree.add_property("num_axons", len(split[(split.compartment == 'axon')]))
    skeleton_tree.add_property("num_dendrites", len(split[(split.compartment == 'dendrite')]))
    plot_tree_digraph(skeleton_tree.skeleton_nx, filename="full_skeleton_tree.pdf")

    # NOTE same type as skeleton, so we can build a SkeletonTree from it
    #print(type(dendrite_skeleton), type(axon_skeleton))
    add_synapse_properties(skeleton_tree)
    properties(skeleton_tree, undirected=undirected)

    print("\n\n\n========================")
    print("--- Axon Properties ---")
    print("========================\n")
    skeleton_tree_axon = SkeletonTree.from_skeleton(skeleton=axon_skeleton, skeleton_type="axon", undirected=True, scaled_coords=skeleton_tree.scaled_coords)
    # skeleton_tree_axon = SkeletonTree(skeletons_dir=SKELETONS_DIR, skeleton_id=SKELETON_ID, undirected=True, skeleton=axon_skeleton)
    add_synapse_properties(skeleton_tree_axon)
    properties(skeleton_tree_axon, undirected=undirected)


    print("\n\n\n============================")
    print("--- Dendrite Properties ---")
    print("============================\n")
    skeleton_tree_dendrite = SkeletonTree.from_skeleton(skeleton=dendrite_skeleton, skeleton_type="dendrite", undirected=True, scaled_coords=skeleton_tree.scaled_coords)
    # skeleton_tree_dendrite = SkeletonTree(skeletons_dir=SKELETONS_DIR, skeleton_id=SKELETON_ID, undirected=True, skeleton=dendrite_skeleton)
    add_synapse_properties(skeleton_tree_dendrite)
    properties(skeleton_tree_dendrite, undirected=undirected)

    exit()

def timeout(seconds):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            if end_time - start_time > seconds:
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            return result
        return wrapper
    return decorator

def kill_process_tree(pid):
    """Kill a process and all its children."""
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            child.kill()
        parent.kill()
    except psutil.NoSuchProcess:
        pass

# Helper function to process a single skeleton.
def process_single_skeleton(skeleton_id, undirected, skeletons_dir):
    """
    Wrapper function to process a skeleton.
    This is what the worker process will execute.
    """
    try:
        # Pass all relevant arguments to the main function
        process_skeleton(skeleton_id=skeleton_id, undirected=undirected, skeletons_dir=skeletons_dir)
        return True # Return success status
    except Exception as e:
        # An exception here means the process failed for some other reason
        print(f"Error processing {skeleton_id}: {e}")
        return False # Return failure status

def limit_cpu_usage(process, num_cpus=4):
    """Limit process and all its children to use only specified CPUs"""
    try:
        # Get process group
        pgid = os.getpgid(process.pid)
        
        # Get all processes in the same group
        group_processes = [p for p in psutil.process_iter() if os.getpgid(p.pid) == pgid]
        
        # Set CPU affinity for each process in the group
        cpu_list = list(range(num_cpus))  # Use only first num_cpus CPUs
        for p in group_processes:
            try:
                p.cpu_affinity(cpu_list)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception as e:
        print(f"Warning: Could not set CPU affinity: {e}")
        # Fallback to nice
        os.nice(num_cpus)

def process_all_w_timeout():
    # Set strict CPU limit
    NUM_CPUS = 4  # Strict limit to 4 CPUs
    undirected = True
    
    timeout = 180
    generate_directory_structure()
    
    ids = [int(x.split('.swc')[0]) for x in os.listdir(SKELETONS_DIR) if x.endswith('.swc')]
    ids = ids[:10]
    
    # Use set for faster lookups
    processed = set()
    if os.path.exists("processed_fractal_dimension.txt"):
        with open('processed_fractal_dimension.txt', 'r') as file:
            processed = {int(line.strip().lstrip('>')) for line in file}

    try:
        os.remove("timeouts.txt")
    except OSError:
        pass

    print("Starting multiprocessing with a timeout...")
    start_time = time.time()
    
    # Limit number of workers to exact CPU count
    num_workers = NUM_CPUS
    print(f"Using exactly {num_workers} worker processes")
    
    # Initialize process start method
    multiprocessing.set_start_method('fork')
    
    results = []
    remaining_ids = [id for id in ids if id not in processed]
    
    with tqdm(total=len(remaining_ids), desc="Processing skeletons") as pbar:
        # Process in chunks to maintain CPU control
        chunk_size = num_workers
        for i in range(0, len(remaining_ids), chunk_size):
            chunk_ids = remaining_ids[i:i + chunk_size]
            
            with multiprocessing.Pool(processes=num_workers) as pool:
                chunk_results = []
                for id in chunk_ids:
                    try:
                        result = pool.apply_async(process_single_skeleton, 
                                               args=(id, undirected, SKELETONS_DIR))
                        success = result.get(timeout=timeout)
                        
                        if success:
                            with open("processed_fractal_dimension.txt", "a") as f:
                                f.write(f"{id}\n")
                            chunk_results.append((id, True))
                        else:
                            chunk_results.append((id, False))
                            
                    except multiprocessing.TimeoutError:
                        print(f"\nTimeout ({timeout}s) reached for skeleton {id}")
                        with open("timeouts.txt", "a") as f:
                            f.write(f"{id}\n")
                        chunk_results.append((id, False))
                        
                    except Exception as e:
                        print(f"\nError processing {id}: {e}")
                        chunk_results.append((id, False))
                    
                    finally:
                        pbar.update(1)
                
                # Clean up pool after each chunk
                pool.close()
                pool.join()
                results.extend(chunk_results)
            
            # Small delay between chunks
            time.sleep(0.1)
    
    end_time = time.time()
    print("\n--- Processing Complete ---")
    successful = sum(1 for _, success in results if success)
    print(f"Successfully processed: {successful}/{len(remaining_ids)}")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    process_all_w_timeout()
    # example_usage()
