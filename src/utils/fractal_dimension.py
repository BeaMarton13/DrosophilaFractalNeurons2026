import numpy as np
import matplotlib.pyplot as plt
import math


def boxcount_sparse(points, box_size):
    """Sparse box counting using hash-based set of occupied cubes."""
    indices = np.floor(points / box_size).astype(int)
    occupied = {tuple(idx) for idx in indices}  # set for uniqueness
    return len(occupied)

def fractal_dimension_sparse(scaled_points, max_dist, limit=1, recalculate=True):
    """Estimate fractal dimension with sparse box-counting."""
    sizes = np.logspace(np.log10(max_dist), np.log10(limit), 20)

    counts = [boxcount_sparse(scaled_points, s) for s in sizes]
    coeffs = np.polyfit(np.log10(1/sizes), np.log10(counts), 1)
    if coeffs[0] < 0.95 and recalculate:
        print("Recalculating fractal dimension after removing plateau points...")
        return fractal_dimension_sparse_remove_points(scaled_points, max_dist)
    return coeffs, sizes, counts


def fractal_dimension_sparse_remove_points(scaled_points, max_dist):
    """Estimate fractal dimension with sparse box-counting, discarding count-plateau points at both ends."""
    sizes = np.logspace(np.log10(max_dist), np.log10(1), 20)

    counts = [boxcount_sparse(scaled_points, s) for s in sizes]
    
    # Convert to arrays for filtering
    sizes = np.array(sizes)
    counts = np.array(counts)
    
    # Remove plateau points: keep only the LAST occurrence of each count value
    # (which corresponds to the smallest box size for that count)
    mask = np.ones(len(counts), dtype=bool)
    for i in range(len(counts) - 1):
        if abs(counts[i] - counts[i + 1]) < 0.05:
            mask[i] = False
        else:
            break 

    for i in range(len(counts) - 1, 1, -1):
        if abs(counts[i] - counts[i - 1]) < 0.05:
            mask[i] = False
        else:
            break 
    
    sizes_filtered = sizes[mask]
    counts_filtered = counts[mask]
    
    # Need at least 2 points for a fit
    if len(counts_filtered) < 2:
        print("Warning: Not enough varying points for fractal dimension fit")
        return (np.nan, np.nan), sizes, counts, sizes_filtered, counts_filtered
    
    # Fit using FILTERED data only
    coeffs = np.polyfit(np.log10(1/sizes_filtered), np.log10(counts_filtered), 1)
    return coeffs, sizes_filtered, counts_filtered


def plot(counts, sizes, coeffs, fname):
    # Log10 values
    x = np.log10(1/sizes)
    y = np.log10(counts)
    slope, intercept = coeffs
    y_fit = slope * x + intercept

    # Plot data + regression line
    plt.figure(figsize=(6,4))
    plt.plot(x, y, 'o-', label="Data (log-log)")
    plt.plot(x, y_fit, 'r--', label=f"Fit (slope={slope:.3f})")
    plt.xlabel("log(1/box_size)")
    plt.ylabel("log(N boxes)")
    plt.title("Fractal Dimension via Sparse Box-Counting")
    plt.legend()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close('all')
