from axon_dendrite_tree import AxonDendriteForest

import os
import matplotlib.pyplot as plt
import sys

import plotly.graph_objects as go
import numpy as np


def generate_savefig_path(property_name, tree_type, plot_type, save_dir, html=False):
    os.makedirs(save_dir, exist_ok=True)
    if html:
        filename = f"{property_name}_{tree_type}_{plot_type}.html"
    else:
        filename = f"{property_name}_{tree_type}_{plot_type}.png"
    return os.path.join(save_dir, filename)


class CompareProperties:
    def __init__(self, axon_dendrite_forest):
        self.axon_dendrite_forest = axon_dendrite_forest

    def _axis_range(self, values, margin_ratio=0.05):
        """Compute min–max range with small margin."""
        if not values:
            return [0, 1]
        vmin, vmax = min(values), max(values)
        if vmin == vmax:
            return [vmin - 1, vmax + 1]
        margin = (vmax - vmin) * margin_ratio
        return [vmin - margin, vmax + margin]

    def plot_histogram(self, property_name, tree_type, bins=30, save_path=None):
        if tree_type == "both":
            axon_props = self.axon_dendrite_forest.get_property_list(property_name, tree_type='axon')
            dendrite_props = self.axon_dendrite_forest.get_property_list(property_name, tree_type='dendrite')
            all_values = axon_props + dendrite_props

            plt.hist(axon_props, bins=bins, alpha=0.5, label='Axon')
            plt.hist(dendrite_props, bins=bins, alpha=0.5, label='Dendrite')
            plt.xlim(self._axis_range(all_values))
            plt.title(f'Histogram of {property_name} for Axon and Dendrite trees')
            plt.xlabel(property_name)
            plt.ylabel('Frequency')
            plt.legend()
        else:
            properties = self.axon_dendrite_forest.get_property_list(property_name, tree_type=tree_type)
            plt.hist(properties, bins=bins)
            plt.xlim(self._axis_range(properties))
            plt.title(f'Histogram of {property_name} for {tree_type} trees')
            plt.xlabel(property_name)
            plt.ylabel('Frequency')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def plot_pairwise(self, property_name_x, property_name_y, save_path=None):
        axon_props_x = self.axon_dendrite_forest.get_property_list(property_name_x, tree_type='axon')
        dendrite_props_y = self.axon_dendrite_forest.get_property_list(property_name_y, tree_type='dendrite')
        min_len = min(len(axon_props_x), len(dendrite_props_y))
        x_vals = axon_props_x[:min_len]
        y_vals = dendrite_props_y[:min_len]

        plt.figure(figsize=(6, 5))
        plt.scatter(x_vals, y_vals, alpha=0.7)
        plt.xlim(self._axis_range(x_vals))
        plt.ylim(self._axis_range(y_vals))
        plt.xlabel(f'Axon {property_name_x}')
        plt.ylabel(f'Dendrite {property_name_y}')
        plt.title(f'Pairwise scatter: Axon {property_name_x} vs Dendrite {property_name_y}')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def plot_indexed_pair(self, property_name, save_path=None):
        axon_values = self.axon_dendrite_forest.get_property_list(property_name, tree_type='axon')
        dendrite_values = self.axon_dendrite_forest.get_property_list(property_name, tree_type='dendrite')
        all_values = axon_values + dendrite_values

        plt.figure(figsize=(8, 5))
        plt.scatter(range(len(axon_values)), axon_values, color='red', label='Axon')
        plt.scatter(range(len(dendrite_values)), dendrite_values, color='green', label='Dendrite')
        plt.ylim(self._axis_range(all_values))
        plt.xlabel('Index (n)')
        plt.ylabel(property_name)
        plt.title(f'Indexed values of {property_name}: Axon (red) & Dendrite (green)')
        plt.legend()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def plot_diff(self, property_name, save_path=None):
        axon_values = self.axon_dendrite_forest.get_property_list(property_name, tree_type='axon')
        dendrite_values = self.axon_dendrite_forest.get_property_list(property_name, tree_type='dendrite')
        diff = [a - d for a, d in zip(axon_values, dendrite_values)]

        plt.figure(figsize=(8, 5))
        plt.scatter(range(len(diff)), diff, color='red', label='Difference (Axon - Dendrite)')
        plt.ylim(self._axis_range(diff))
        plt.xlabel('Index (n)')
        plt.ylabel(f'Difference in {property_name}')
        plt.title(f'Indexed difference in {property_name} (Axon - Dendrite)')
        plt.legend()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def plot_diff_histogram(self, property_name, bins=30, save_path=None):
        axon_values = self.axon_dendrite_forest.get_property_list(property_name, tree_type='axon')
        dendrite_values = self.axon_dendrite_forest.get_property_list(property_name, tree_type='dendrite')
        diff = [a - d for a, d in zip(axon_values, dendrite_values)]

        plt.hist(diff, bins=bins)
        plt.xlim(self._axis_range(diff))
        plt.title(f'Histogram of difference in {property_name} (Axon - Dendrite)')
        plt.xlabel('Difference')
        plt.ylabel('Frequency')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()
        plt.close()


class ComparePropertiesPlotly:
    def __init__(self, axon_dendrite_forest):
        self.axon_dendrite_forest = axon_dendrite_forest

    def _axis_range(self, values, margin_ratio=0.05):
        """Compute range for axis with a small margin."""
        if not values:
            return [0, 1]
        vmin, vmax = min(values), max(values)
        if vmin == vmax:
            return [vmin - 1, vmax + 1]
        margin = (vmax - vmin) * margin_ratio
        return [vmin - margin, vmax + margin]

    def plot_histogram(self, property_name, tree_type, bins=30, save_path=None):
        if tree_type == "both":
            axon_props = self.axon_dendrite_forest.get_property_list(property_name, tree_type='axon')
            dendrite_props = self.axon_dendrite_forest.get_property_list(property_name, tree_type='dendrite')
            all_values = axon_props + dendrite_props

            fig = go.Figure()
            fig.add_trace(go.Histogram(x=axon_props, name='Axon', nbinsx=bins, opacity=0.5))
            fig.add_trace(go.Histogram(x=dendrite_props, name='Dendrite', nbinsx=bins, opacity=0.5))
            
            fig.update_layout(
                barmode='overlay',
                title_text=f'Histogram of {property_name} for Axon and Dendrite trees',
                xaxis_title=property_name,
                yaxis_title='Frequency',
                legend_title='Tree Type',
                xaxis=dict(range=self._axis_range(all_values))
            )
        else:
            properties = self.axon_dendrite_forest.get_property_list(property_name, tree_type=tree_type)
            fig = go.Figure(data=[go.Histogram(x=properties, nbinsx=bins)])
            fig.update_layout(
                title_text=f'Histogram of {property_name} for {tree_type} trees',
                xaxis_title=property_name,
                yaxis_title='Frequency',
                xaxis=dict(range=self._axis_range(properties))
            )

        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()

    def plot_pairwise(self, property_name_x, property_name_y, save_path=None):
        axon_props_x = self.axon_dendrite_forest.get_property_list(property_name_x, tree_type='axon')
        dendrite_props_y = self.axon_dendrite_forest.get_property_list(property_name_y, tree_type='dendrite')
        min_len = min(len(axon_props_x), len(dendrite_props_y))
        x_vals = axon_props_x[:min_len]
        y_vals = dendrite_props_y[:min_len]
        
        fig = go.Figure(data=go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='markers'
        ))
        
        fig.update_layout(
            title_text=f'Pairwise scatter: Axon {property_name_x} vs Dendrite {property_name_y}',
            xaxis_title=f'Axon {property_name_x}',
            yaxis_title=f'Dendrite {property_name_y}',
            xaxis=dict(range=self._axis_range(x_vals)),
            yaxis=dict(range=self._axis_range(y_vals))
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()

    def plot_indexed_pair(self, property_name, save_path=None):
        axon_values = self.axon_dendrite_forest.get_property_list(property_name, tree_type='axon')
        dendrite_values = self.axon_dendrite_forest.get_property_list(property_name, tree_type='dendrite')
        all_values = axon_values + dendrite_values
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(axon_values))),
            y=axon_values,
            mode='markers',
            name='Axon',
            marker=dict(color='red')
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(dendrite_values))),
            y=dendrite_values,
            mode='markers',
            name='Dendrite',
            marker=dict(color='green')
        ))
        
        fig.update_layout(
            title_text=f'Indexed values of {property_name}: Axon (red) & Dendrite (green)',
            xaxis_title='Index (n)',
            yaxis_title=property_name,
            legend_title='Tree Type',
            yaxis=dict(range=self._axis_range(all_values))
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()

    def plot_diff(self, property_name, save_path=None):
        axon_values = self.axon_dendrite_forest.get_property_list(property_name, tree_type='axon')
        dendrite_values = self.axon_dendrite_forest.get_property_list(property_name, tree_type='dendrite')
        diff = [a - d for a, d in zip(axon_values, dendrite_values)]
        
        fig = go.Figure(data=go.Scatter(
            x=list(range(len(diff))),
            y=diff,
            mode='markers',
            name='Difference (Axon - Dendrite)',
            marker=dict(color='red')
        ))
        
        fig.update_layout(
            title_text=f'Indexed difference in {property_name} (Axon - Dendrite)',
            xaxis_title='Index (n)',
            yaxis_title=f'Difference in {property_name}',
            yaxis=dict(range=self._axis_range(diff))
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()

    def plot_diff_histogram(self, property_name, bins=30, save_path=None):
        axon_values = self.axon_dendrite_forest.get_property_list(property_name, tree_type='axon')
        dendrite_values = self.axon_dendrite_forest.get_property_list(property_name, tree_type='dendrite')
        diff = [a - d for a, d in zip(axon_values, dendrite_values)]
        
        fig = go.Figure(data=[go.Histogram(x=diff, nbinsx=bins)])
        fig.update_layout(
            title_text=f'Histogram of difference in {property_name} (Axon - Dendrite)',
            xaxis_title='Difference',
            yaxis_title='Frequency',
            xaxis=dict(range=self._axis_range(diff))
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()


if __name__ == "__main__":
    # NOTE: Change the directed/undirected cases, while experimenting
    tree_type = "undirected"  # or "undirected"


    if sys.platform == "darwin":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tree_dir = os.path.abspath(os.path.join(script_dir, f"../../data/{tree_type}/tree_properties/"))
        c_dir = os.path.abspath(os.path.join(script_dir, f"../../data/{tree_type}/c_values/"))
    else:
        res_dir = "/data/RESULTS/USERS/bea/drosophila/"
        tree_dir = os.path.abspath(os.path.join(res_dir, f"data/{tree_type}/tree_properties/"))
        c_dir = os.path.abspath(os.path.join(res_dir, f"data/{tree_type}/c_values/"))

    forest = AxonDendriteForest.build_forest_from_directory(tree_dir, c_dir, undirected=tree_type=='undirected')

    comparator = CompareProperties(forest)
    comparator_plotly = ComparePropertiesPlotly(forest)

    properties = forest.axon_tree[0].get_all_properties()

    if sys.platform == "darwin":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.abspath(os.path.join(script_dir, "../../figures"))
    else:
        res_dir = "/data/RESULTS/USERS/bea/drosophila/"
        save_dir = os.path.abspath(os.path.join(res_dir, "figures"))


    for property in properties:
        comparator.plot_histogram(property, tree_type='both', bins=100, save_path=generate_savefig_path(property, tree_type, 'histogram', save_dir, html=False))
        comparator.plot_pairwise(property, property, save_path=generate_savefig_path(property, tree_type, 'pairwise', save_dir, html=False))
        comparator.plot_indexed_pair(property, save_path=generate_savefig_path(property, tree_type, 'indexed_pair', save_dir, html=False))
        comparator.plot_diff(property, save_path=generate_savefig_path(property, tree_type, 'diff', save_dir, html=False))
        comparator.plot_diff_histogram(property, bins=100, save_path=generate_savefig_path(property, tree_type, 'diff_histogram', save_dir, html=False))

        comparator_plotly.plot_histogram(property, tree_type='both', bins=100, save_path=generate_savefig_path(property, tree_type, 'histogram', save_dir, html=True))
        comparator_plotly.plot_pairwise(property, property, save_path=generate_savefig_path(property, tree_type, 'pairwise', save_dir, html=True))
        comparator_plotly.plot_indexed_pair(property, save_path=generate_savefig_path(property, tree_type, 'indexed_pair', save_dir, html=True))
        comparator_plotly.plot_diff(property, save_path=generate_savefig_path(property, tree_type, 'diff', save_dir, html=True))
        comparator_plotly.plot_diff_histogram(property, bins=100, save_path=generate_savefig_path(property, tree_type, 'diff_histogram', save_dir, html=True))

        