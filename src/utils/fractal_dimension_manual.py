import numpy as np


def find_bounding_box(points):
    """
    Finds the smallest axis-aligned bounding box for a set of 3D points.

    Args:
        points (list): A list of points, where each point is a list/tuple of (x, y, z) coordinates.

    Returns:
        tuple: A tuple containing the min and max corners of the bounding box.
               Example: ((min_x, min_y, min_z), (max_x, max_y, max_z))
    """

    # Unpack the points into separate lists for each coordinate
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]

    # Find the minimum and maximum values for each axis
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)
    
    # Construct the min and max corners of the bounding box
    min_corner = (min_x, min_y, min_z)
    max_corner = (max_x, max_y, max_z)

    return (min_corner, max_corner)


def get_box_dimensions(bounding_box):
    """
    Calculates the width, height, and depth of a 3D bounding box.

    Args:
        bounding_box (tuple): A tuple containing the min and max corners.
                              Example: ((min_x, min_y, min_z), (max_x, max_y, max_z))

    Returns:
        tuple: A tuple with the (width, height, depth) of the box.
    """
    min_corner, max_corner = bounding_box
    
    min_x, min_y, min_z = min_corner
    max_x, max_y, max_z = max_corner
    
    width = max_x - min_x
    height = max_y - min_y
    depth = max_z - min_z
    
    return (width, height, depth)


def num_occupied_boxes(points, n_pieces, min_corner, dimensions):
    """Count how many grid cells (n_pieces per axis) contain at least one point."""
    start_x, start_y, start_z = min_corner
    width, height, depth = dimensions
    dx = width / n_pieces
    dy = height / n_pieces
    dz = depth / n_pieces

    occupied = set()
    for p in points:
        ix = int((p[0] - start_x) // dx)
        iy = int((p[1] - start_y) // dy)
        iz = int((p[2] - start_z) // dz)
        # Clip indices to be inside the grid
        ix = min(ix, n_pieces - 1)
        iy = min(iy, n_pieces - 1)
        iz = min(iz, n_pieces - 1)
        occupied.add((ix, iy, iz))
    return len(occupied)

def fractal_dimension(points, num_pieces_list=None):
    """Estimate the box-counting fractal dimension of a 3D point set."""
    # 1. Find the bounding box and its dimensions for your point set
    bounding_box = find_bounding_box(points)
    min_corner, _ = bounding_box
    dimensions = get_box_dimensions(bounding_box)

    if num_pieces_list is None:
        num_pieces_list = [2, 4, 8, 16, 32, 64, 128]

    log_N = []
    log_L = []

    for n in num_pieces_list:
        occupied_boxes = num_occupied_boxes(points, n, min_corner, dimensions)
        log_N.append(np.log(occupied_boxes))
        log_L.append(np.log(n))  # or log(1/box size)


    # 5. Fit a line to the logarithmic data to find the slope
    # The slope is the fractal dimension
    fractal_dimension, _ = np.polyfit(log_L, log_N, 1)
    return fractal_dimension


if __name__ == "__main__":
    # Sample point set from a previous discussion
    points = [
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
    ]

    fractal_dimension_val = fractal_dimension(points)
    print(fractal_dimension_val)