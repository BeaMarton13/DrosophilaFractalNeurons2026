"""Plot a histogram of the saved max_dist property across all processed
skeletons, and list the ones above 0.4 to longer_skeletons.txt."""

import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np

# NOTE: res_dir (used below) is only defined on the non-darwin branch; on
# macOS this script raises a NameError once it reaches the directory listing.
if sys.platform == "darwin":
    script_dir = os.path.dirname(os.path.abspath(__file__))
else:
    script_dir = "/data/RESULTS/USERS/bea/drosophila/"
    res_dir = "/data/RESULTS/USERS/bea/drosophila/data_navis/undirected/tree_properties/"

max_dists = []
longer_ids = []
for f in os.listdir(res_dir):
    if f.endswith("_full.json"):
        properties = json.load(open(os.path.join(res_dir, f), "r"))
        if "max_dist" in properties:
            max_dist = properties["max_dist"]
            if max_dist is not None:
                max_dists.append(max_dist)
                if max_dist > 0.4:
                    skeleton_id = f.split("_")[0]
                    longer_ids.append((skeleton_id, max_dist))

with open(os.path.join(script_dir, "longer_skeletons.txt"), "w") as f:
    for skeleton_id, max_dist in longer_ids:
        f.write(f"{skeleton_id}: {max_dist}\n")


# Calculate statistics
mean_dist = np.mean(max_dists)
median_dist = np.median(max_dists)

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(max_dists, bins=50, alpha=0.7, edgecolor='black')

# Add vertical line at 0.25
plt.axvline(x=0.25, color='red', linestyle='--', linewidth=2, label='x = 0.25')

# Add text with mean and median
plt.text(0.95, 0.95, f'Mean: {mean_dist:.4f}\nMedian: {median_dist:.4f}', 
         transform=plt.gca().transAxes, 
         fontsize=12, 
         verticalalignment='top',
         horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.xlabel('Max Distance', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Histogram of Max Distances', fontsize=16)
plt.legend(fontsize=12)
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "max_dist_histogram.png"), dpi=300)
plt.show()