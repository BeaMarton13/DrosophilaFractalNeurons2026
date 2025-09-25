import networkx as nx
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


def plot_tree_digraph(
    G: nx.DiGraph, filename: str = "binary_tree.png"
) -> None:  # Use graphviz_layout for tree-like layout
    """
    Removes all middle nodes from the given directed graph (DiGraph).

    A middle node is a node that has an in-degree of 1 and an out-degree of 1. This function iteratively removes all such middle nodes by connecting their predecessor to their successor, effectively collapsing the middle node.

    Parameters:
        G (nx.DiGraph): The directed graph to remove middle nodes from.

    Returns:
        nx.DiGraph: The modified graph with all middle nodes removed.
    """
    plt.switch_backend("Agg")
    plt.ioff()

    # Layout with Graphviz
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

    # Determine node colors based on out-degree
    node_colors = []
    for node in G.nodes():
        if G.out_degree(node) > 2:
            node_colors.append("red")  # Red for nodes with out-degree > 2
        else:
            node_colors.append("skyblue")  # Skyblue for all others

    # Draw the graph with custom arrow size and node colors
    nx.draw(
        G,
        pos,
        with_labels=True,
        arrows=True,
        node_size=10,
        node_color=node_colors,
        font_size=5,
        arrowsize=5,
    )

    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()