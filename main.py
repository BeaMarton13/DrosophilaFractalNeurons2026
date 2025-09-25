import navis
import os
import multiprocessing
import time
import concurrent.futures

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


script_dir = os.path.dirname(os.path.abspath(__file__))
path_to_skeletons = "./input_data/full"
# path_to_skeletons = "./sk_lod1_783_healed"
SKELETONS_DIR = os.path.join(script_dir, path_to_skeletons)

# arrays used as indices must be of integer (or boolean) type
SKELETON_ID = 720575940626286432

# list index out of range:
# SKELETON_ID = 720575940628446888

# SKELETON_ID = 720575940608945163
# SKELETON_ID = 720575940661305217

processed = []
if os.path.exists("processed.txt"):
    with open('processed.txt', 'r') as file:
        for line in file:
            if line.startswith(">"):
                processed.append(int(line.strip()[1:]))
            else:
                processed.append(int(line.strip()))

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


def properties(skeleton_tree: SkeletonTree):
    print("--- Tree Properties ---")
    for key, value in skeleton_tree.tree_properties.items():
        print(f"{key}: {value}")

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


def example_usage():
    skeleton_tree = SkeletonTree(SKELETONS_DIR, SKELETON_ID, skeleton_type="full", undirected=True)
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
    properties(skeleton_tree)

    print("\n\n\n========================")
    print("--- Axon Properties ---")
    print("========================\n")
    skeleton_tree_axon = SkeletonTree.from_skeleton(skeleton=axon_skeleton, skeleton_type="axon", undirected=True)
    # skeleton_tree_axon = SkeletonTree(skeletons_dir=SKELETONS_DIR, skeleton_id=SKELETON_ID, undirected=True, skeleton=axon_skeleton)
    add_synapse_properties(skeleton_tree_axon)
    properties(skeleton_tree_axon)


    print("\n\n\n============================")
    print("--- Dendrite Properties ---")
    print("============================\n")
    skeleton_tree_dendrite = SkeletonTree.from_skeleton(skeleton=dendrite_skeleton, skeleton_type="dendrite", undirected=True)
    # skeleton_tree_dendrite = SkeletonTree(skeletons_dir=SKELETONS_DIR, skeleton_id=SKELETON_ID, undirected=True, skeleton=dendrite_skeleton)
    add_synapse_properties(skeleton_tree_dendrite)
    properties(skeleton_tree_dendrite)

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

def process_all_w_timeout():
    timeout = 60
    generate_directory_structure()
    # Define a list of skeletons to process
    full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./input_data/full")
    ids = [int(x.split('.swc')[0]) for x in os.listdir(full_path) if x.endswith('.swc')]

    # Load processed IDs
    processed = []
    if os.path.exists("processed.txt"):
        with open('processed.txt', 'r') as file:
            for line in file:
                if line.startswith(">"):
                    processed.append(int(line.strip()[1:]))
                else:
                    processed.append(int(line.strip()))

    try:
        os.remove("timeouts.txt")
    except OSError:
        pass

    print("Starting multiprocessing with a timeout...")
    start_time = time.time()
    
    # Use a multiprocessing Pool
    num_workers = max(1, int(os.cpu_count() * 0.85))
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Create a list of async results
        async_results = []
        for id in ids:
            if id not in processed:
                # Submit the task for the current ID and store the result object
                res = pool.apply_async(
                    process_single_skeleton,
                    args=(id, True, full_path)
                )
                async_results.append((id, res))

        # Wait for and collect the results with a timeout
        with tqdm(total=len(async_results), desc="Processing skeletons") as pbar:
            for id, res in async_results:
                try:
                    # Get the result with a timeout. This will block until the result is available or a timeout occurs.
                    success = res.get(timeout=timeout)
                    if success:
                        with open("processed.txt", "a") as f:
                            f.write(f"{id}\n")
                except multiprocessing.TimeoutError:
                    # `apply_async` handles termination of the worker, so we just log the timeout
                    with open("timeouts.txt", "a") as f:
                        f.write(f"{id}\n")
                    with open("processed.txt", "a") as f:
                        f.write(f">{id}\n")
                except Exception as e:
                    print(f"\nAn unexpected error occurred for skeleton ID {id}: {e}")
                finally:
                    pbar.update(1)

    end_time = time.time()
    print("\n--- Processing Complete ---")
    print(f"\nTotal time taken: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    process_all_w_timeout()
    # example_usage()