import os
import sys
import navis
from fafbseg import flywire
import matplotlib.pyplot as plt

flywire.set_default_dataset("public")

if sys.platform == "darwin":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path_to_skeletons = "./input_data/full"
    SKELETONS_DIR = os.path.join(script_dir, path_to_skeletons)
else:
    script_dir = "/data/RESULTS/USERS/bea/drosophila/"
    in_script_dir = "/data/RESULTS/PROJECTS/drosophila/input_data/full"
    SKELETONS_DIR = in_script_dir

SKELETON_ID = 720575940620971605
# SKELETON_ID = 720575940623165257
# SKELETON_ID = 720575940624436458


SKELETON_ID = 720575940629121223
SKELETON_ID = 720575940618533265
SKELETON_ID = 720575940619601462
SKELETON_ID = 720575940637000590
SKELETON_ID = 720575940642943885
SKELETON_ID = 720575940637041271
SKELETON_ID = 720575940623504086
SKELETON_ID = 720575940628505132

# Dendrite (> 2)
SKELETON_ID = 720575940632167085
SKELETON_ID = 720575940628069501

# Axon (> 2)
SKELETON_ID = 720575940611671506
skeleton_path = os.path.join(SKELETONS_DIR, f"{SKELETON_ID}.swc")

# Load skeleton
skeleton = navis.read_swc(skeleton_path)

# Plot with matplotlib
fig, ax = plt.subplots(figsize=(8, 8))
# navis.plot2d(skeleton, ax=ax)
navis.plot3d(skeleton, backend='plotly').show()
ax.set_title(f"Skeleton ID: {SKELETON_ID}")
plt.show()