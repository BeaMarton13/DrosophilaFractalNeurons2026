import numpy as np
import matplotlib.pyplot as plt


def boxcount_sparse(points, box_size):
    """Sparse box counting using hash-based set of occupied cubes."""
    indices = np.floor(points / box_size).astype(int)
    occupied = {tuple(idx) for idx in indices}  # set for uniqueness
    return len(occupied)

def fractal_dimension_sparse(scaled_points, max_dist):
    """Estimate fractal dimension with sparse box-counting."""
    sizes = np.logspace(np.log10(max_dist), np.log10(1), 10)

    counts = [boxcount_sparse(scaled_points, s) for s in sizes]
    coeffs = np.polyfit(np.log(1/sizes), np.log(counts), 1)
    return coeffs, sizes, counts
    # return coeffs[0]


def plot(counts, sizes, coeffs, fname):
    # Log values
    x = np.log(1/sizes)
    y = np.log(counts)
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
    # plt.show()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close('all')
