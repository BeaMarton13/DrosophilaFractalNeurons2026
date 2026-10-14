"""Report how many neurons have a fractal dimension < 1 for each compartment (full/axon/dendrite)."""
import sys
import os
# Disable internal parallelism to prevent subprocess spawning
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ.setdefault("NAVIS_MAX_CORES", "1")

from src.utils.axon_dendrite_tree import AxonDendriteForest

if __name__ == "__main__":
    # NOTE: Change the directed/undirected cases, while experimenting
    tree_type = "undirected"  # or "undirected"

    if sys.platform == "darwin":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tree_dir = os.path.abspath(os.path.join(script_dir, f"../../data/{tree_type}/tree_properties/"))
        c_dir = os.path.abspath(os.path.join(script_dir, f"../../data/{tree_type}/c_values/"))
    else:
        res_dir = "/data/RESULTS/USERS/bea/drosophila/"
        tree_dir = os.path.abspath(os.path.join(res_dir, f"data_navis/{tree_type}/tree_properties/"))
        c_dir = os.path.abspath(os.path.join(res_dir, f"data_navis/{tree_type}/c_values/"))

    forest = AxonDendriteForest.build_forest_from_directory(tree_dir, c_dir, undirected=tree_type=='undirected')
    print(len(forest.axon_tree), len(forest.dendrite_tree), len(forest.full_tree))

    count = 0
    counts = {"full":0, "axon_only":0, "dendrite_only":0}
    fd_axon = forest.get_property_list("fractal_dimension", tree_type="axon")
    fd_dendrite = forest.get_property_list("fractal_dimension", tree_type="dendrite")
    fd_full = forest.get_property_list("fractal_dimension", tree_type="full")
    for fda in fd_axon:
        if fda is not None and fda < 1:
            counts["axon_only"] += 1
    for fdd in fd_dendrite:
        if fdd is not None and fdd < 1:
            counts["dendrite_only"] += 1
    for fdf in fd_full:
        if fdf is not None and fdf < 1:
            counts["full"] += 1
    print("Number of neurons with all three FD values:", count)
    print("Counts of neurons with FD < 1:")
    print(counts)
    print("Total neurons:", len(forest.axon_tree))