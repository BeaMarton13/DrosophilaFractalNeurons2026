import numpy as np
from scipy.spatial import cKDTree
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from src.utils.fractal_dimension_manual import fractal_dimension as fd

points = np.array([
    [604870.7, 290591.25, 77228.625],
    [603943.56, 289850.94, 70767.53],
    [603949.44, 290129.66, 70504.59],
    [603988.9, 290143.72, 70222.59],
    [604005.5, 290206.1, 70091.375],
    [603968.8, 290122.28, 70639.766],
    [604429.0, 290805.62, 70662.375],
    [603967.0, 290144.47, 70383.26],
    [603997.8, 289827.38, 68304.53],
    [604000.0, 290126.4, 69963.02],
    [604037.5, 290064.03, 69428.31],
    [604146.1, 289810.84, 68196.06],
    [604010.1, 290072.44, 69558.12],
    [604346.3, 289858.5, 68048.54],
    [603989.0, 290086.56, 69695.21],
    [603993.25, 290117.56, 69824.125],
    [604031.1, 290052.1, 69288.055],
    [604079.5, 290056.75, 69025.95],
    [604059.75, 290091.12, 69129.75],
    [603992.8, 289777.78, 68438.9],
    [603985.0, 289748.4, 68553.06],
    [604103.75, 289965.44, 68907.46],
    [604100.4, 289933.47, 68791.125],
    [604050.7, 289794.3, 68633.05],
    [604052.2, 289867.44, 68719.734],
    [604392.94, 289889.56, 67935.29],
    [604306.8, 290361.0, 72452.34],
    [604327.0, 290324.0, 72731.61],
    [604332.6, 290243.25, 72237.664],
    [604334.0, 290365.12, 72854.21],
    [604338.3, 290269.28, 72292.914],
    [604341.3, 290328.62, 72055.38],
    [604401.94, 289974.9, 67814.42],
    [604399.0, 290856.4, 70279.48],
    [604351.8, 290366.94, 72529.58],
    [604365.44, 290377.34, 71962.02],
    [604379.4, 290392.62, 72983.836],
    [604467.25, 289816.38, 67437.64],
    [604428.94, 290000.47, 67695.12],
    [604387.2, 290018.56, 67574.19],
    [604402.0, 290563.8, 70932.055],
    [604403.2, 290449.1, 71834.82],
    [604348.7, 290814.22, 70523.64],
    [604409.2, 290610.6, 70794.22],
    [604428.3, 290460.78, 71706.945],
    [604433.56, 289893.12, 67473.45],
    [604333.25, 290809.53, 70326.68],
    [604217.25, 290185.56, 78335.945],
    [604463.5, 290126.06, 78625.64],
    [604445.0, 290498.75, 71355.95],
    [604056.1, 290346.78, 78224.84],
    [604474.2, 290587.88, 71073.39],
    [604476.5, 290379.34, 73065.94],
    [604478.75, 290794.2, 75339.37],
    [604482.9, 290580.66, 71243.06],
    [604487.1, 290518.6, 71484.59],
    [604504.1, 290816.44, 75216.72],
    [604510.56, 290514.97, 71628.73],
    [604546.3, 290380.38, 73153.29],
    [604558.94, 290636.0, 75491.37],
    [604565.4, 290400.7, 73285.88],
    [604577.5, 290479.56, 73580.71],
    [604581.3, 290805.62, 75050.62],
    [604586.1, 290587.5, 75619.984],
    [604594.8, 290480.56, 73704.53],
    [604595.8, 290434.97, 73426.65],
    [604620.2, 290555.03, 73976.805],
    [604632.25, 290513.75, 73835.35],
    [604929.94, 290556.25, 77869.05],
    [604884.75, 290588.12, 77129.75],
    [604638.75, 290551.7, 75726.086],
    [604658.9, 290587.44, 74219.46],
    [604671.2, 290565.56, 74099.47],
    [604680.1, 290788.1, 74830.64],
    [604720.9, 289991.94, 77163.39],
    [604709.94, 289957.97, 76879.96],
    [604702.44, 289946.22, 76891.56],
    [604716.9, 289971.6, 77038.9],
    [604725.1, 290652.94, 74320.42],
    [604742.1, 290746.6, 74700.016],
    [604786.9, 290720.12, 74557.44],
    [604827.5, 289119.16, 79490.19],
    [605224.3, 290070.56, 79583.74],
    [604812.6, 289251.7, 79562.33],
    [604871.25, 288965.5, 79342.016],
    [604958.9, 288798.84, 79256.04],
    [604855.56, 290686.97, 74461.25],
    [604810.9, 289491.8, 79602.664],
    [605225.94, 290132.06, 79464.42],
    [604904.94, 290640.53, 77140.164],
    [604877.1, 290525.06, 77983.8],
    [605017.1, 288702.94, 79293.945],
    [604799.1, 290076.25, 78766.09],
    [605069.4, 290412.22, 78271.22],
    [604441.44, 290066.06, 79046.55],
    [604893.2, 289625.84, 79698.08],
    [604955.0, 290705.97, 77100.32],
    [604792.44, 289393.47, 79616.16],
    [605026.56, 290700.6, 77075.9],
    [605137.1, 290332.34, 78152.19],
    [604867.56, 290129.3, 78707.12],
    [605084.25, 290715.78, 77080.83],
    [605131.94, 288597.8, 79309.98],
    [605127.4, 288509.6, 79324.766],
    [604871.94, 290117.47, 78557.93],
    [605086.94, 290806.44, 77082.22],
    [604862.56, 289753.47, 79623.93],
    [604443.7, 289996.66, 79221.34],
    [604769.5, 290933.78, 78632.63],
    [604707.75, 290129.97, 77428.0],
    [605026.56, 290176.66, 78281.266],
    [605128.06, 290687.7, 77010.98],
    [605100.7, 290824.28, 77119.83],
    [605097.6, 290798.03, 77078.23],
    [605105.5, 290698.56, 77030.26],
    [605084.1, 290767.44, 76971.41],
    [604948.25, 290089.28, 78406.16],
    [604725.0, 290131.5, 77305.234],
    [605097.1, 290648.1, 77056.39],
    [605136.56, 288385.9, 79380.09],
    [605146.3, 288437.25, 79320.82],
    [605105.2, 290204.84, 78268.336],
    [604696.1, 290942.94, 78517.484],
    [604882.0, 289909.5, 79626.89],
    [605058.5, 289934.47, 79663.02],
    [605138.56, 290005.66, 79631.99]
])


def boxcount_sparse(points, box_size):
    """Sparse box counting using hash-based set of occupied cubes."""
    # Normalize points into [0,1]^3
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(points)

    # Compute cube indices
    indices = np.floor(scaled / box_size).astype(int)
    occupied = {tuple(idx) for idx in indices}  # set for uniqueness
    return len(occupied)

def fractal_dimension_sparse(points, exps=range(1,12)):
    """Estimate fractal dimension with sparse box-counting."""
    sizes = 2.0 ** -np.array(exps)
    counts = [boxcount_sparse(points, s) for s in sizes]
    coeffs = np.polyfit(np.log(1/sizes), np.log(counts), 1)
    return coeffs[0], sizes, counts

# ==============================================================================================
def boxcount_kdtree(points, exponents=None, min_exp=0, max_exp=8, return_counts=False):
    """
    points : (N,3) array
    exponents define box_size = 2**(-exp) after normalizing points to [0,1]^3
    """
    scaler = MinMaxScaler()
    pts = scaler.fit_transform(points)
    if exponents is None:
        exponents = np.arange(min_exp, max_exp+1)
    counts = []
    for e in exponents:
        box_size = 2.0 ** (-e)
        # integer indexing avoids floating rounding issues
        idx = np.floor(pts / box_size).astype(int)
        # ensure uniqueness caps: sometimes idx may equal max index; that's fine
        unique = np.unique(idx, axis=0)
        counts.append(unique.shape[0])
    if return_counts:
        return np.array(counts), exponents
    return np.array(counts)


# ==============================================================================================
def auto_select_scale_range(logsizes, logcounts, min_window=3, r2_threshold=0.95):
    """
    Find best contiguous window across scales that maximizes R^2 of linear fit.
    Returns (best_start_index, best_end_index, best_r2)
    """
    best = None
    L = len(logsizes)
    for i in range(0, L-min_window+1):
        for j in range(i+min_window-1, L):
            xs = np.array(logsizes[i:j+1])
            ys = np.array(logcounts[i:j+1])
            if len(xs) < 2:
                continue
            coeffs = np.polyfit(xs, ys, 1)
            fit = np.polyval(coeffs, xs)
            ss_res = np.sum((ys - fit)**2)
            ss_tot = np.sum((ys - np.mean(ys))**2)
            r2 = 1 - ss_res / (ss_tot + 1e-12)
            # prefer larger windows by slightly boosting r2 with window size factor
            score = r2 + 0.001*(j-i)
            if best is None or score > best[0]:
                best = (score, i, j, r2, coeffs[0])
    if best is None:
        return None
    return (best[1], best[2], best[3], best[4])  # start, end, r2, slope

def fractal_dimension_box(points, min_exp=0, max_exp=8, auto=True, plot=True):
    counts, exponents = boxcount_kdtree(points, min_exp=min_exp, max_exp=max_exp, return_counts=True)
    logsizes = np.log(2.0**exponents)  # log(1/box_size) = log(2^exp) = exp * log(2)
    logcounts = np.log(counts + 1e-12)
    if auto:
        sel = auto_select_scale_range(logsizes, logcounts)
        if sel:
            i,j,r2,slope = sel
            xs = logsizes[i:j+1]; ys = logcounts[i:j+1]
            D = slope
            chosen = (exponents[i], exponents[j], r2)
        else:
            coeffs = np.polyfit(logsizes, logcounts, 1)
            D = coeffs[0]
            chosen = (exponents[0], exponents[-1], None)
    else:
        coeffs = np.polyfit(logsizes, logcounts, 1)
        D = coeffs[0]
        chosen = (exponents[0], exponents[-1], None)

    return D, chosen


# ==============================================================================================
def correlation_dimension(points, n_r=20, r_min_ratio=1e-3, r_max_ratio=0.5, plot=True):
    """
    Estimate correlation dimension D2 via C(r) ~ r^D2.
    Uses KDTree for pair counting.
    """
    pts = (points - points.min(axis=0)) / points.ptp(axis=0)
    N = pts.shape[0]
    tree = cKDTree(pts)
    # choose r values between r_min and r_max based on bounding box diag
    # r in normalized units
    r_min = r_min_ratio
    r_max = r_max_ratio
    rs = np.logspace(np.log10(r_min), np.log10(r_max), n_r)
    C = []
    for r in rs:
        # query pairs up to radius r, count pairs (i<j)
        counts = tree.query_ball_tree(tree, r)
        # counts is list of neighbors per point (including itself)
        s = 0
        for i, neigh in enumerate(counts):
            # count neighbors with index > i to avoid double-count
            s += sum(1 for j in neigh if j > i)
        # normalization: number of pairs = N*(N-1)/2
        C.append(s / (N*(N-1)/2.0))
    logr = np.log(rs)
    logC = np.log(np.array(C) + 1e-18)
    coeffs = np.polyfit(logr, logC, 1)
    D2 = coeffs[0]
    return D2


def boxcount_3d(points, box_size):
    """
    Count how many cubes of size box_size are occupied by at least one point.
    """
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(points)
    indices = np.floor(scaled / box_size).astype(int)
    unique_cubes = np.unique(indices, axis=0)
    return len(unique_cubes)


def suggest_box_exp(points, max_exp=12, stability_threshold=0.1, window=3):
    """
    Suggest stable ranges of box exponents for fractal dimension estimation.
    """
    sizes = 2.0 ** -np.arange(1, max_exp+1)
    counts = [boxcount_3d(points, s) for s in sizes]

    logsizes = np.log(1/sizes)
    logcounts = np.log(counts)

    slopes = np.diff(logcounts) / np.diff(logsizes)
    stable_ranges = []

    for i in range(len(slopes) - window):
        window_slopes = slopes[i:i+window]
    if np.std(window_slopes) < stability_threshold:
        stable_ranges.append((i+1, i+window+1))

    return stable_ranges, (logsizes, logcounts, slopes)


def fractal_dimension_3d(points, max_exp=12, auto=True, plot=True):
    """
    Estimate fractal dimension of 3D point cloud using box counting.

    Parameters:
    - points: numpy array (N,3)
    - max_exp: maximum exponent to test
    - auto: if True, automatically pick best exponent range
    - plot: if True, show log-log plot

    Returns:
    - Estimated fractal dimension
    """
    # Suggest stable range if auto is enabled
    stable_ranges, (logsizes, logcounts, slopes) = suggest_box_exp(points, max_exp=max_exp)

    if auto and stable_ranges:
        min_idx, max_idx = stable_ranges[0]
    else:
        min_idx, max_idx = 1, max_exp # fallback

    sizes = 2.0 ** -np.arange(min_idx, max_idx+1)
    counts = [boxcount_3d(points, s) for s in sizes]

    coeffs = np.polyfit(np.log(1/sizes), np.log(counts), 1)
    D = coeffs[0]

    if plot:
        plt.scatter(logsizes, logcounts, label="data")
        plt.plot(np.log(1/sizes), np.polyval(coeffs, np.log(1/sizes)), 'r--',
        label=f"fit: D={D:.3f}")
        plt.xlabel("log(1/box size)")
        plt.ylabel("log(count)")
        plt.legend()
        plt.show()

    return D


print(fractal_dimension_sparse(points))
print()
print(boxcount_kdtree(points))
print()
print(fractal_dimension_box(points))
print()
print(correlation_dimension(points))
print()
print(fractal_dimension_3d(points))
print()
print(fd(points))