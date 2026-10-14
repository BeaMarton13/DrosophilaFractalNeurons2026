import os

# Disable internal parallelism to prevent subprocess spawning
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ.setdefault("NAVIS_MAX_CORES", "1")

import navis
navis.set_pbars(hide=True)  # Disable progress bars
navis.config.parallel = False  # Disable navis parallelism if available

import multiprocessing as mp
import sys
import argparse
import psutil
import pandas as pd

from tqdm import tqdm

from src.export_properties_main import process_skeleton, generate_directory_structure

# Output file structure:
#   ./data_{split_type}/{directed or undirected}/{tree_properties or c_values}/{skeleton_id}_full.json
#   ./data_{split_type}/{directed or undirected}/{tree_properties or c_values}/{skeleton_id}_axon.json
#   ./data_{split_type}/{directed or undirected}/{tree_properties or c_values}/{skeleton_id}_dendrite.json
#   ./input_data/full/{skeleton_id}.swc

if sys.platform == "darwin":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    SKELETONS_DIR = os.path.join(script_dir, "./input_data/full")
else:
    script_dir = "/data/RESULTS/USERS/bea/drosophila/"
    SKELETONS_DIR = "/data/RESULTS/PROJECTS/drosophila/input_data/full"

NUM_CPUS = 10
TIMEOUT = 120
CPU_LIST = list(range(NUM_CPUS))


def process_single_skeleton(skeleton_id, undirected, skeletons_dir, axon_dendrite_node_ids, split_type):
    """Process a single skeleton; run by the worker pool, returns success status."""
    try:
        if axon_dendrite_node_ids is None or skeleton_id in axon_dendrite_node_ids['root_id'].values:
            process_skeleton(skeleton_id=skeleton_id, undirected=undirected, skeletons_dir=skeletons_dir, axon_dendrite_node_ids=axon_dendrite_node_ids, split_type=split_type)
        else:
            print(f"Skipping skeleton {skeleton_id} as it is not in axon_dendrite_node_ids.")
        return True
    except Exception as e:
        print(f"Error processing {skeleton_id}: {e}")
        return False


def process_all_w_timeout(axon_dendrite_node_ids, split_type):
    generate_directory_structure(split_type)
    undirected = True

    ids = [int(x.split('.swc')[0]) for x in os.listdir(SKELETONS_DIR) if x.endswith('.swc')]
    ids = ids[:100]

    processed = set()
    if os.path.exists("processed.txt"):
        with open('processed.txt', 'r') as file:
            processed = {int(line.strip().lstrip('>')) for line in file}

    try:
        os.remove("timeouts.txt")
    except OSError:
        pass

    remaining_ids = [id for id in ids if id not in processed]
    run_parallel_skeletons(skeleton_ids=remaining_ids, undirected=undirected, skeletons_dir=SKELETONS_DIR, processed_file='processed.txt', num_cpus=NUM_CPUS, timeout=TIMEOUT, axon_dendrite_node_ids=axon_dendrite_node_ids, split_type=split_type)


def _worker_tuple(args):
    """Top-level wrapper so Pool can return (skeleton_id, success) with timeout handling."""
    sid, undirected, skeletons_dir, axon_dendrite_node_ids, split_type = args
    try:
        ok = process_single_skeleton(sid, undirected, skeletons_dir, axon_dendrite_node_ids, split_type)
        return sid, ok
    except Exception as e:
        print(f"Error processing {sid}: {e}", flush=True)
        return sid, False


def _worker_init():
    """Pins each worker process to one CPU (round-robin by worker identity)."""
    try:
        import navis
    except ImportError:
        pass

    num_cpus = len(CPU_LIST)
    worker_identity = mp.current_process()._identity[0] - 1  # identity is 1-based
    cpu_id = CPU_LIST[worker_identity % num_cpus]

    p = psutil.Process()
    try:
        p.cpu_affinity([cpu_id])
    except Exception as e:
        print(f"[worker init] affinity failed: {e}", flush=True)

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    try:
        navis.config.set_temp({'max_cores': 1})
    except Exception:
        pass


def run_parallel_skeletons(skeleton_ids, undirected, skeletons_dir, processed_file, num_cpus=4, timeout=180, axon_dendrite_node_ids=None, split_type='navis'):
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=num_cpus, initializer=_worker_init, maxtasksperchild=1)

    futures = []
    for sid in skeleton_ids:
        args = (sid, undirected, skeletons_dir, axon_dendrite_node_ids, split_type)
        fut = pool.apply_async(_worker_tuple, args=(args,))
        futures.append((sid, fut))

    pool.close()

    results = []
    with tqdm(total=len(futures), desc="Processing skeletons") as pbar:
        for sid, fut in futures:
            try:
                sid_out, ok = fut.get(timeout=TIMEOUT)
                if ok:
                    with open(processed_file, "a") as f:
                        f.write(f"{sid_out}\n")
                else:
                    with open("timeouts.txt", "a") as f:
                        f.write(f"{sid_out}\n")

                results.append((sid_out, ok))

            except mp.TimeoutError:
                with open("timeouts.txt", "a") as f:
                    f.write(f"{sid}\n")
                results.append((sid, False))

            pbar.update(1)

    pool.join()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process skeletons and export their tree/synapse properties.")
    parser.add_argument("split_type", type=str, help="Split type (navis or mb)")
    args = parser.parse_args()

    split_type = args.split_type
    axon_dendrite_node_ids = None
    if split_type != "navis":
        if sys.platform == "darwin":
            split_filename = os.path.abspath(os.path.join(script_dir, "./data/calculate_dendrite_axonend_distances-MIN.csv"))
        else:
            split_filename = "/data/RESULTS/PROJECTS/drosophila/calculate_dendrite_axonend_distances-MIN/calculate_dendrite_axonend_distances-MIN.csv"
        axon_dendrite_node_ids = pd.read_csv(split_filename)

    process_all_w_timeout(axon_dendrite_node_ids, split_type)
