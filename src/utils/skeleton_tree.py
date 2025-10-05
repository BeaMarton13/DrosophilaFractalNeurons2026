import networkx as nx
import navis
from fafbseg import flywire
import numpy as np
import os
import sys
from sklearn.preprocessing import MinMaxScaler

from src.utils.fractal_dimension import fractal_dimension_sparse, plot



if sys.platform == "darwin":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    autosave_path = "../../input_data/"
    AUTOSAVE_PATH = os.path.abspath(os.path.join(script_dir, autosave_path))
else:
    AUTOSAVE_PATH = os.path.abspath("/data/RESULTS/PROJECTS/drosophila/input_data/")


class SkeletonTree:
    def __init__(
            self, 
            skeletons_dir: str, 
            skeleton_id: int, 
            skeleton_type: str,
            undirected: bool = False, 
            skeleton: navis.Neuron = None, 
            autosave: bool = False,
            scaled_coords: list = None
        ):
        """
        Initializes the SkeletonTree with the given directory and skeleton ID.
        
        Args:
            skeletons_dir (str): Directory where skeleton files are stored.
            skeleton_id (int): ID of the skeleton to prepare.
        """
        self.skeleton_id = skeleton_id
        if skeleton is None:
            # Read skeleton from the specified directory
            self.skeleton = self._read_skeleton(skeletons_dir)
        else:
            self.skeleton = skeleton
        self.skeleton_nx = self._prepare_skeleton()
        self.skeleton_nx_tree = self.skeleton_nx.copy()
        self.skeleton_type = skeleton_type
        self.undirected = undirected

        self.scaled_coords = scaled_coords

        self.fractal_dimension = self._calculate_fractal_dimension(self.scaled_coords, with_plot=True)

        if self.undirected:
            self.skeleton_nx_tree = self.skeleton_nx.to_undirected()
        self.level_w_max_width, self.max_width = self.calculate_max_width_at_level()
        self.tree_properties = self.build_tree_properties()
        self.c_value_properties = self.build_c_value_properties()
        if autosave:
            full_path = os.path.join(AUTOSAVE_PATH, f"{skeleton_type}/{self.skeleton_id}.swc")
            if not os.path.exists(full_path):
                navis.write_swc(self.skeleton, full_path)

    @property
    def id(self) -> int:
        """
        Returns the ID of the skeleton.
        """
        return self.skeleton_id

    @property
    def graph(self) -> nx.DiGraph:
        """
        Returns the directed graph representation of the skeleton.
        """
        return self.skeleton_nx

    @property
    def num_nodes(self) -> int:
        """
        Returns the number of nodes in the skeleton graph.
        """
        return self.skeleton_nx.number_of_nodes()

    @property
    def num_edges(self) -> int:
        """
        Returns the number of edges in the skeleton graph.
        """
        return self.skeleton_nx.number_of_edges()

    @property
    def root_node(self) -> int:
        """
        Returns the root node of the skeleton graph.
        """
        roots = [int(n) for n in self.skeleton_nx.nodes if self.skeleton_nx.in_degree(n) == 0]
        return roots[0] if roots else None

    @property
    def leaf_nodes(self) -> list:
        """
        Returns a list of leaf nodes in the skeleton graph.
        """
        return [n for n in self.skeleton_nx.nodes if self.skeleton_nx.out_degree(n) == 0]

    @property
    def height(self) -> int:
        """
        Returns the height of the skeleton graph.
        """
        if not self.skeleton_nx:
            return 0
        root = self.root_node
        if root is None:
            return 0
        max_depth = 0
        for node in self.skeleton_nx.nodes():
            depth = self._get_depth(node)
            if depth > max_depth:
                max_depth = depth
        return max_depth

    @property
    def number_of_three_children(self) -> int:
        """
        Returns the number of nodes with exactly three children in the skeleton graph.
        """
        return sum(1 for n in self.skeleton_nx.nodes if self.skeleton_nx.out_degree(n) == 3)

    @property
    def number_of_multi_children(self) -> int:
        """
        Returns the number of nodes with more than three children in the skeleton graph.
        """
        return sum(1 for n in self.skeleton_nx.nodes if self.skeleton_nx.out_degree(n) > 3)

    @property
    def number_of_leaf_nodes(self) -> int:
        """
        Returns the number of leaf nodes in the skeleton graph.
        """
        return len(self.leaf_nodes)

    @property
    def driver_nodes(self) -> tuple[int, list[int]]:
        """
        Computes the number of driver nodes and the list of driver nodes in a given directed graph.

        Driver nodes are the nodes that are not matched in the maximum matching of the bipartite representation of the graph.

        Returns:
            list[int]: A tuple containing the number of driver nodes and the list of driver nodes.
        """
        _, driver_nodes = self._count_driver_nodes()
        return driver_nodes

    @property
    def num_driver_nodes(self) -> tuple[int, list[int]]:
        """
        Computes the number of driver nodes and the list of driver nodes in a given directed graph.

        Driver nodes are the nodes that are not matched in the maximum matching of the bipartite representation of the graph.

        Returns:
            list[int]: A tuple containing the number of driver nodes and the list of driver nodes.
        """
        num, _ = self._count_driver_nodes()
        return num

    @property
    def num_driver_nodes_matching(self) -> int:
        """
        Computes the number of driver nodes and the list of driver nodes in a given directed graph.

        Driver nodes are the nodes that are not matched in the maximum matching of the bipartite representation of the graph.

        Returns:
            int: The number of driver nodes.
        """
        return self._count_driver_nodes_matching()

    @property
    def zero_eigenvalues(self) -> tuple[int, list[float]]:
        """
        Computes the number of zero eigenvalues and the list of eigenvalues for a given directed graph.

        The number of zero eigenvalues corresponds to the number of driver nodes in the graph, which are the nodes that are not matched in the maximum matching of the bipartite representation of the graph.

        Returns:
            tuple[int, list[float]]: A tuple containing the number of zero eigenvalues and the list of eigenvalues.
        """
        _, eigenvalues = self._count_zero_eigenvalues()
        return eigenvalues

    @property
    def num_zero_eigenvalues(self) -> tuple[int, list[float]]:
        """
        Computes the number of zero eigenvalues and the list of eigenvalues for a given directed graph.

        The number of zero eigenvalues corresponds to the number of driver nodes in the graph, which are the nodes that are not matched in the maximum matching of the bipartite representation of the graph.

        Returns:
            tuple[int, list[float]]: A tuple containing the number of zero eigenvalues and the list of eigenvalues.
        """
        num, _ = self._count_zero_eigenvalues()
        return num

    @property
    def eigenvalue_geometric_multiplicity(self) -> int:
        """
        Gets the geometric multiplicity of zero eigenvalues for a graph.
        
        Returns:
            int: The geometric multiplicity of zero eigenvalues.
        """
        return self._get_eigenvalue_geometric_multiplicity(0)

    @property
    def cable_length(self) -> float:
        """
        Returns the total cable length of the skeleton.
        
        Returns:
            float: The total cable length of the skeleton.
        """
        return float(self.skeleton.cable_length) if hasattr(self.skeleton, 'cable_length') else 0.0

    # --- Properties Management ---
    def add_property(self, key: str, value: any):
        """
        Adds a property to the skeleton tree.
        
        Args:
            key (str): The key for the property.
            value (any): The value of the property.
        """
        self.tree_properties[key] = value

    def build_tree_properties(self) -> dict:
        """
        Returns a dictionary of properties for the skeleton tree.
        """
        return {
            "id": self.id,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "root_node": self.root_node,
            "leaf_nodes": self.leaf_nodes,
            "num_leaf_nodes": len(self.leaf_nodes),
            "height": self.height,
            "level_w_max_width": self.level_w_max_width,
            "max_width": self.max_width,
            "number_of_three_children": self.number_of_three_children,
            "number_of_multi_children": self.number_of_multi_children,
            "number_of_leaf_nodes": self.number_of_leaf_nodes,
            "cable_length": self.cable_length,
            "fractal_dimension": self.fractal_dimension
        }

    def build_c_value_properties(self) -> dict:
        """
        Gets the C value properties for the skeleton tree.
        
        Returns:
            dict: A dictionary containing the C value properties.
        """
        if self.num_driver_nodes:
            driver_nodes_count = self.num_driver_nodes / self.num_nodes if self.num_nodes > 0 else 0,
        else:
            driver_nodes_count = None,
        return {
            "driver_nodes_matching_count": driver_nodes_count,
            "zero_eigenvalues_count": self.num_zero_eigenvalues / self.num_nodes if self.num_nodes > 0 else 0,
            "eigenvalue_geometric_multiplicity": self.eigenvalue_geometric_multiplicity / self.num_nodes if self.num_nodes > 0 else 0,
        }
    
    def _calculate_fractal_dimension(self, scaled_coords, with_plot=False):
        self.coords = self.skeleton.nodes[['x', 'y', 'z']].values
        scaler = MinMaxScaler()
        if scaled_coords is None:
            self.scaled_coords = scaler.fit_transform(self.coords)

        if with_plot:
            coeffs, sizes, counts = fractal_dimension_sparse(scaler.fit_transform(self.coords), max(self._calculate_scaled_lengths(self.scaled_coords)))
            plot(counts, sizes, coeffs, fname=f"fractal_dimension{self.skeleton_type}.png")

        coeffs, _, _ = fractal_dimension_sparse(scaler.fit_transform(self.coords), max(self._calculate_scaled_lengths(self.scaled_coords)))
        return coeffs[0]
    
    def _calculate_scaled_lengths(self, scaled_points):
        distances = []
        adjacency = self.skeleton.get_igraph().get_adjacency()
        neighbor_pairs = []
        for idx, lst in enumerate(adjacency):
            try:
                p_idx = next((i for i, x in enumerate(lst) if x), None)
            except:
                p_idx = -1


            neighbor_pairs.append([idx, p_idx])
        for idx, parent in zip(self.skeleton.nodes['node_id'], self.skeleton.nodes['parent_id']):
            if parent != -1:
                distances.append(np.linalg.norm(scaled_points[parent - 1] - scaled_points[idx - 1]))
        return distances
    
    # --- Skeleton Reading and Preparation ---
    def _read_skeleton(self, skeletons_dir: str) -> navis.Neuron:
        """
        Read a skeleton from the specified directory and skeleton ID.
        """
        return navis.read_swc(os.path.join(skeletons_dir, f'{self.skeleton_id}.swc'))

    def _prepare_skeleton(self, step_size: int = 200) -> nx.DiGraph:
        """
        Prepare the skeleton by resampling and converting it to a directed graph.
        
        Args:
            step_size (int): The step size for resampling.
        
        Returns:
            nx.DiGraph: The prepared directed graph representation of the skeleton.
        """
        resampled_skeleton = self._resample_skeleton(step_size)
        reversed_skeleton = self._reverse_graph(resampled_skeleton.get_graph_nx())
        removed_middle_nodes_skeleton = self._remove_all_middle_nodes(reversed_skeleton)
        return removed_middle_nodes_skeleton

    def _resample_skeleton(self, step_size: int = 200) -> navis.Neuron:
        """
        Resamples the skeleton to a specified step size.
        
        Args:
            step_size (int): The step size for resampling.
        """
        # NOTE: The properties of the skeleton are not changed by resampling, so we can use the same functions
        return navis.downsample_neuron(self.skeleton, step_size)

    def _remove_all_middle_nodes(self, G: nx.DiGraph) -> nx.DiGraph:
        """
        Removes all middle nodes from the given directed graph (DiGraph).

        A middle node is a node that has an in-degree of 1 and an out-degree of 1. This function iteratively removes all such middle nodes by connecting their predecessor to their successor, effectively collapsing the middle node.

        Parameters:
            G (nx.DiGraph): The directed graph to remove middle nodes from.

        Returns:
            nx.DiGraph: The modified graph with all middle nodes removed.
        """
        while True:
            # Find all middle nodes
            middle_nodes = [node for node in G.nodes if G.out_degree(node) == 1 and G.in_degree(node) == 1]

            # Stop if no more middle nodes
            if not middle_nodes:
                break

            # Process each middle node
            for node in middle_nodes:
                # Get predecessor and successor
                pred = list(G.predecessors(node))[0]
                succ = list(G.successors(node))[0]

                # Connect predecessor to successor
                G.add_edge(pred, succ)

                # Remove the middle node
                G.remove_node(node)

        return G

    def _reverse_graph(self, G: nx.DiGraph) -> nx.DiGraph:
        """
        Reverse the direction of edges in a NetworkX graph representation of a skeleton.
        
        Parameters:
        G (nx.DiGraph): The directed graph to reverse.

        Returns:
        nx.DiGraph: A directed graph with reversed edges.
        """
        return G.reverse()

    # --- Tree Properties Calculation ---
    def _get_depth(self, node: int) -> int:
        if not self.skeleton_nx.has_node(node):
            return -1
        depth = 0
        while node in self.skeleton_nx and self.skeleton_nx.in_degree(node) > 0:
            node = next(iter(self.skeleton_nx.predecessors(node)))
            depth += 1
        return depth

    # --- C Values Calculation ---
    def _count_driver_nodes(self) -> tuple[int, list[int]]:
        """
        Computes the number of driver nodes and the list of driver nodes in a given directed graph.

        Driver nodes are the nodes that are not matched in the maximum matching of the bipartite representation of the graph.

        Returns:
            tuple[int, list[int]]: A tuple containing the number of driver nodes and the list of driver nodes.
        """
        try:
        # Compute maximum matching in the bipartite representation
            matching = nx.bipartite.maximum_matching(self.skeleton_nx)

            # Find unmatched nodes (driver nodes)
            matched_nodes = set(matching.keys())
            driver_nodes = [node for node in self.skeleton_nx.nodes if node not in matched_nodes]
            return len(driver_nodes), driver_nodes
        except:
            # print("Error in computing driver nodes. Returning empty list.")
            return None, []

    def _count_driver_nodes_matching(self) -> int:
        """
        Computes the number of driver nodes in a given directed graph.

        Driver nodes are the nodes that are not matched in the maximum matching of the bipartite representation of the graph.

        Returns:
            int: Number of driver nodes.
        """
        # Create bipartite graph with proper node naming
        B = nx.Graph() # Use undirected graph for bipartite matching

        # Add nodes with explicit bipartite sets
        for node in self.skeleton_nx_tree.nodes():
            B.add_node(f"L_{node}", bipartite=0)
            B.add_node(f"R_{node}", bipartite=1)

        # Add edges ensuring all nodes exist
        for u, v in self.skeleton_nx_tree.edges():
            B.add_edge(f"L_{u}", f"R_{v}")

        # Get left nodes set
        left = {n for n, d in B.nodes(data=True) if d["bipartite"] == 0}

        # Calculate maximum matching - maximum number of edges where no two edges share a vertex
        matching = nx.bipartite.maximum_matching(B, top_nodes=left)

        # Calculate number of driver nodes - nodes not matched in the maximum matching (since all nodes appear twice in the bipartite graph we have to divide by 2)
        driver_nodes = len(self.skeleton_nx_tree.nodes()) - len(matching) // 2
        return driver_nodes

    def _count_zero_eigenvalues(self, matrix_type: str = "adjacency") -> tuple[int, list[float]]:
        """
        Computes the number of zero eigenvalues and the list of eigenvalues for a given directed graph.

        The number of zero eigenvalues corresponds to the number of driver nodes in the graph, which are the nodes that are not matched in the maximum matching of the bipartite representation of the graph.

        Parameters:
            matrix_type (str): The type of matrix to use for the eigenvalue computation. Can be 'adjacency' or 'laplacian'.

        Returns:
            tuple[int, list[float]]: A tuple containing the number of zero eigenvalues and the list of eigenvalues.
        """
        # TODO: check first if it's an undirected network -> all links to all directions
        if matrix_type == "adjacency":
            A = nx.to_numpy_array(self.skeleton_nx_tree)  # Adjacency matrix
        elif matrix_type == "laplacian":
            A = nx.laplacian_matrix(self.skeleton_nx_tree).toarray()  # Laplacian matrix
        else:
            raise ValueError("Invalid matrix_type. Choose 'adjacency' or 'laplacian'.")

        # Compute eigenvalues
        eigenvalues = np.linalg.eigvals(A)

        # Count eigenvalues close to zero (considering numerical precision)
        zero_count = np.sum(np.isclose(eigenvalues, 0, atol=1e-9))

        return zero_count, eigenvalues

    def _calculate_geometric_multiplicity(self, matrix: np.ndarray, eigenvalue: float, tolerance: float = 1e-10) -> int:
        """
        Calculates the geometric multiplicity of a given eigenvalue.

        Parameters
        ----------
        matrix : np.ndarray
            The input matrix
        eigenvalue : float
            The eigenvalue to calculate multiplicity for
        tolerance : float
            Numerical tolerance for comparing floating point values

        Returns
        -------
        int
            The geometric multiplicity
        """
        # Subtract eigenvalue from diagonal to get (A - λI)
        shifted_matrix = matrix - eigenvalue * np.eye(matrix.shape[0])

        # Calculate rank
        rank = np.linalg.matrix_rank(shifted_matrix, tol=tolerance)

        # Geometric multiplicity is dimension of nullspace
        # = n - rank(A - λI) where n is matrix dimension
        geometric_multiplicity = matrix.shape[0] - rank

        return geometric_multiplicity

    def _get_eigenvalue_geometric_multiplicity(self, eigenvalue: float = 0) -> int:
        """
        Gets the geometric multiplicity of zero eigenvalues for a graph.

        Parameters:
            eigenvalue (float): The eigenvalue to calculate multiplicity for.
        """
        # Get adjacency matrix
        A = nx.adjacency_matrix(self.skeleton_nx_tree).toarray()
        # Calculate geometric multiplicity
        return self._calculate_geometric_multiplicity(A, eigenvalue=eigenvalue)

    # --- Max width at level ---
    def _count_nodes_at_levels(self):
        """
        Counts the number of nodes at each level in the skeleton tree.

        Returns:
            dict: A dictionary where keys are levels and values are counts of nodes at those levels.
        """
        levels = {}
        for node in self.skeleton_nx_tree.nodes():
            depth = self._get_depth(node)
            if depth not in levels:
                levels[depth] = 0
            levels[depth] += 1
        return levels

    def calculate_max_width_at_level(self) -> int:
        """
        Calculates the maximum width at a given level in the skeleton tree.

        Args:
            level (int): The level for which to calculate the maximum width.

        Returns:
            int: The maximum width at the specified level.
        """
        levels = self._count_nodes_at_levels()
        max_key = max(levels, key=levels.get)
        max_value = levels[max_key]

        return max_key, max_value

    @classmethod
    def from_skeleton(cls, skeleton: navis.Neuron, skeleton_type:str, undirected: bool, autosave: bool = False, scaled_coords: list = None):
        return cls("dummy_value", skeleton.id, skeleton=skeleton, skeleton_type=skeleton_type, undirected=undirected, autosave=autosave, scaled_coords=scaled_coords)
