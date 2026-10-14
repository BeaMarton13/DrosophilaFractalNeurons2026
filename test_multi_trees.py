import sys
import os

# Disable internal parallelism to prevent subprocess spawning
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ.setdefault("NAVIS_MAX_CORES", "1")

import navis
from fafbseg import flywire

if sys.platform == "darwin":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path_to_skeletons = "./input_data/full"
    SKELETONS_DIR = os.path.join(script_dir, path_to_skeletons)
else:
    # We will call this script_dir for easier usage
    script_dir = "/data/RESULTS/USERS/bea/drosophila/"
    in_script_dir = "/data/RESULTS/PROJECTS/drosophila/input_data/full"
    SKELETONS_DIR = in_script_dir

flywire.set_default_dataset('public')
skeleton_id = 720575940632021867
# skeleton = navis.read_swc(os.path.join(SKELETONS_DIR, f'{skeleton_id}.swc'))
skeleton = flywire.get_skeletons(skeleton_id, dataset=783)
print(skeleton)
frags = navis.break_fragments(skeleton)
print(len(frags))
exit()