import navis
import os
import multiprocessing
import time
import json
import sys
import numpy as np
import psutil


from fafbseg import flywire
from tqdm import tqdm

from src.utils.skeleton_tree import SkeletonTree
from src.utils.synapses import Synapses
from src.utils.plot import plot_tree_digraph

from src.export_properties_main import process_skeleton, generate_directory_structure

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
    in_script_dir = "/data/RESULTS/PROJECTS/drosophila/input_data/full"
    SKELETONS_DIR = in_script_dir

# path_to_skeletons = "./sk_lod1_783_healed"

# arrays used as indices must be of integer (or boolean) type
# SKELETON_ID = 720575940626286432

# list index out of range:
# SKELETON_ID = 720575940628446888

# SKELETON_ID = 720575940608945163
SKELETON_ID = 720575940638111872
# SKELETON_ID = 720575940661305217

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


def properties(skeleton_tree: SkeletonTree, undirected, filename=None):
    if filename is not None:
        if undirected:
            dir_val = "undirected"
        else:
            dir_val = "directed"

        if sys.platform == "darwin":
            file_with_path = os.path.abspath(os.path.join(script_dir, f"../data/{dir_val}/tree_properties/{filename}"))
        else:
            file_with_path = os.path.abspath(os.path.join(script_dir, f"data/{dir_val}/tree_properties/{filename}"))
    
        try:
            with open(file_with_path, 'w') as fp:
                json.dump(skeleton_tree.tree_properties, fp, cls=NpEncoder, indent=4)
        except TypeError as e:
            print(f"Error: {e}")
    else:
        print("--- Tree Properties ---")
        for key, value in skeleton_tree.tree_properties.items():
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
    
    timeout = 180
    generate_directory_structure()
    undirected = True
    
    ids = [int(x.split('.swc')[0]) for x in os.listdir(SKELETONS_DIR) if x.endswith('.swc')]
    ids = ids[:10]
    
    # Use set for faster lookups
    processed = set()
    if os.path.exists("processed.txt"):
        with open('processed.txt', 'r') as file:
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
                            with open("processed.txt", "a") as f:
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
