import matplotlib.pyplot as plt

# File path
distances_file = "distances.txt"

# Read max distances
max_distances = []
skeleton_ids = []

with open(distances_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # Split by " - "
        dist_str, sid_str = line.split(" - ")
        max_distances.append(float(dist_str))
        skeleton_ids.append(sid_str)

# Plot histogram
plt.figure(figsize=(10,6))
plt.hist(max_distances, bins=100, color='skyblue', edgecolor='black')
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.xlabel('Max Distance')
plt.ylabel('Number of Skeletons (log scale)')
plt.title('Histogram of Max Distances per Skeleton')
plt.grid(True)
plt.show()