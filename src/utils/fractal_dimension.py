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



# def boxcount_sparse(points, box_size):
#     """Sparse box counting using hash-based set of occupied cubes.
    
#     Args:
#         points (np.ndarray): Array of points (N x 3) for 3D data
#         box_size (float): Size of each box
#     """
#     # Add small epsilon to avoid edge cases
#     eps = 1e-10
#     indices = np.floor((points + eps) / box_size).astype(int)
#     occupied = {tuple(idx) for idx in indices}
#     return len(occupied)


# def fractal_dimension_sparse(scaled_points, max_dist):
#     """Estimate fractal dimension with sparse box-counting.
#     Args:
#         scaled_points (np.ndarray): Points in normalized [0,1] space
#         max_dist (float): Maximum box size based on max segment length
        
#     Returns:
#         tuple: (slope, sizes, counts)
#     """
#     # Use more box sizes and ensure minimum size isn't too small
#     min_size = max_dist / 100  # Avoid boxes smaller than 1% of max length
#     num_sizes = 20
#     sizes = np.logspace(np.log10(max_dist), np.log10(min_size), num_sizes)
    
#     counts = [boxcount_sparse(scaled_points, s) for s in sizes]
    
#     # Use only valid counts (avoid log(0))
#     valid_idx = [i for i, c in enumerate(counts) if c > 0]
#     valid_counts = [counts[i] for i in valid_idx]
#     valid_sizes = [sizes[i] for i in valid_idx]
    
#     # Fit line to log-log plot
#     coeffs = np.polyfit(np.log(1/np.array(valid_sizes)), 
#                        np.log(np.array(valid_counts)), 1)
    
#     return coeffs, valid_sizes, valid_counts


# def plot(counts, sizes, coeffs, fname, show_validation=True):
#     """Plot box counting results with validation.
    
#     Args:
#         counts (np.ndarray): Box counts
#         sizes (np.ndarray): Box sizes
#         coeffs (np.ndarray): Polyfit coefficients
#         fname (str): Output filename
#         show_validation (bool): Show validation metrics
#     """
#     x = np.log(1/sizes)
#     y = np.log(counts)
#     slope, intercept = coeffs
#     y_fit = slope * x + intercept
    
#     # Calculate R-squared
#     r_squared = np.corrcoef(x, y)[0,1] ** 2
    
#     plt.figure(figsize=(8,6))
#     plt.plot(x, y, 'o', label="Data points", markersize=4)
#     plt.plot(x, y_fit, 'r--', label=f"Fit (D={slope:.3f})")
    
#     plt.xlabel("log(1/box size)")
#     plt.ylabel("log(N boxes)")
#     plt.title("Fractal Dimension Analysis")
    
#     if show_validation:
#         plt.text(0.05, 0.95, 
#                 f'D = {slope:.3f}\nR² = {r_squared:.3f}', 
#                 transform=plt.gca().transAxes,
#                 bbox=dict(facecolor='white', alpha=0.8))
    
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.savefig(fname, dpi=300, bbox_inches='tight')
#     plt.close('all')
