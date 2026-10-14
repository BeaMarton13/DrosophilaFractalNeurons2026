"""Load a single skeleton by ID and render it in an interactive 3D plot."""

import os
import sys
import navis

if sys.platform == "darwin":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path_to_skeletons = "./input_data/full"
    SKELETONS_DIR = os.path.join(script_dir, path_to_skeletons)
else:
    script_dir = "/data/RESULTS/USERS/bea/drosophila/"
    in_script_dir = "/data/RESULTS/PROJECTS/drosophila/input_data/full"
    SKELETONS_DIR = in_script_dir

SKELETON_ID = 720575940613583001
skeleton_path = os.path.join(SKELETONS_DIR, f"{SKELETON_ID}.swc")

# Load skeleton
skeleton = navis.read_swc(skeleton_path)

# Plot interactively
navis.plot3d(skeleton, backend='plotly').show()
