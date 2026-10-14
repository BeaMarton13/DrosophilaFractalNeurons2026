"""Draw an example tree annotated with its maximum width and height, used
to illustrate the max-width/height tree properties in the paper figures."""

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch

def visualize_3ary_tree_max_width_simple():
    """
    Visualizes a 3-ary tree where the maximum width is in the middle level.
    Shows only the tree with a simple arrow indicating max width and tree height.
    """
    # Create a 3-ary tree (each node has at most 3 children)
    G = nx.DiGraph()
    
    # Build tree structure with max width in the middle
    # Level 0: 1 node (root)
    G.add_node(0, level=0)
    
    # Level 1: 2 nodes (fewer than max)
    for i in range(1, 3):
        G.add_node(i, level=1)
        G.add_edge(0, i)
    
    # Level 2: 6 nodes (MAXIMUM WIDTH - middle level)
    node_id = 3
    children_per_parent = [3, 3]  # Each node at level 1 gets 3 children
    for parent_idx, parent in enumerate(range(1, 3)):
        for _ in range(children_per_parent[parent_idx]):
            G.add_node(node_id, level=2)
            G.add_edge(parent, node_id)
            node_id += 1
    
    # Level 3: 4 nodes (decreasing after max width)
    # Only some nodes at level 2 have children
    for parent in [3, 4, 7, 8]:  # Only 4 out of 6 nodes have children
        G.add_node(node_id, level=3)
        G.add_edge(parent, node_id)
        node_id += 1
    
    # Level 4: 1 node (continuing to decrease)
    for parent in [12]:
        G.add_node(node_id, level=4)
        G.add_edge(parent, node_id)
        node_id += 1
    
    # Count nodes at each level
    levels = {}
    for node in G.nodes():
        level = G.nodes[node]['level']
        if level not in levels:
            levels[level] = 0
        levels[level] += 1
    
    # Find max width and tree height
    max_level = max(levels, key=levels.get)
    max_width = levels[max_level]
    tree_height = max(levels.keys())
    
    # Create hierarchical layout
    pos = {}
    for node in G.nodes():
        level = G.nodes[node]['level']
        nodes_at_level = [n for n in G.nodes() if G.nodes[n]['level'] == level]
        nodes_at_level.sort()
        index = nodes_at_level.index(node)
        
        # Position: x based on index, y based on level (inverted for top-down)
        x = (index - len(nodes_at_level) / 2 + 0.5) * (12 / max(len(nodes_at_level), 1))
        y = -level * 2
        pos[node] = (x, y)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=10, 
                          edge_color='gray', width=1.5, connectionstyle='arc3,rad=0.1')
    
    # Draw nodes with different colors per level
    colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6']
    for level in range(5):
        nodes_at_level = [n for n in G.nodes() if G.nodes[n]['level'] == level]
        if not nodes_at_level:
            continue
        color = colors[level % len(colors)]
        alpha = 1.0 if level == max_level else 0.7
        size = 900 if level == max_level else 600
        linewidth = 3 if level == max_level else 1.5
        nx.draw_networkx_nodes(G, pos, nodelist=nodes_at_level, 
                              node_color=color, node_size=size, 
                              alpha=alpha, ax=ax, linewidths=linewidth,
                              edgecolors='black')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=9, font_color='white', 
                           font_weight='bold', ax=ax)
    
    # Add arrow pointing to maximum width level
    y_max = -max_level * 2
    arrow_width = FancyArrowPatch(
        (-8, y_max), (-6.5, y_max),
        arrowstyle='->', mutation_scale=30, linewidth=3,
        color='red', zorder=10
    )
    ax.add_patch(arrow_width)
    
    # Add text label for max width
    ax.text(-8.5, y_max, f'Maximum Width\n({max_width} nodes)', 
           fontsize=14, fontweight='bold', color='red',
           ha='right', va='center',
           bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
                    edgecolor='white', linewidth=2, alpha=0.9))
    
    # Add vertical arrow showing tree height
    y_top = 0
    y_bottom = -tree_height * 2
    arrow_height = FancyArrowPatch(
        (7, y_top), (7, y_bottom),
        arrowstyle='<->', mutation_scale=25, linewidth=3,
        color='blue', zorder=10
    )
    ax.add_patch(arrow_height)
    
    # Add text label for tree height
    ax.text(7.5, (y_top + y_bottom) / 2, f'Tree Height\n({tree_height})', 
           fontsize=14, fontweight='bold', color='blue',
           ha='left', va='center',
           bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
                    edgecolor='white', linewidth=2, alpha=0.9))
    
    ax.set_title('Tree Structure with Maximum Width and Height', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.axis('off')
    ax.set_xlim(-10, 11)
    ax.set_ylim(-9, 1)
    
    plt.tight_layout()
    plt.savefig('tree_max_width_simple.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("TREE MAXIMUM WIDTH AND HEIGHT")
    print("="*60)
    print(f"\nTree Height: {tree_height} (levels 0 to {tree_height})")
    print(f"Maximum Width: {max_width} nodes at level {max_level}")
    print("\nNodes per level: " + " → ".join([str(levels[i]) for i in sorted(levels.keys())]))
    print("="*60)

if __name__ == "__main__":
    visualize_3ary_tree_max_width_simple()