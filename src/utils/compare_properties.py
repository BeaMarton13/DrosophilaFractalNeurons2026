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


import numpy as np
import matplotlib.pyplot as plt


class CompareProperties:
    def __init__(self, axon_dendrite_forest):
        self.axon_dendrite_forest = axon_dendrite_forest

    def _axis_range(self, values, margin_ratio=0.05):
        """Compute min–max range with small margin for linear axes."""
        if not values:
            return [0, 1]
        vmin, vmax = min(values), max(values)
        if vmin == vmax:
            return [vmin - 1, vmax + 1]
        margin = (vmax - vmin) * margin_ratio
        return [vmin - margin, vmax + margin]

    def plot_histogram(self, property_name, tree_type, bins=30, log_scale=False, save_path=None):
        plt.figure(figsize=(10, 6))
        
        # --- Data preparation ---
        if tree_type == "both":
            axon_props = self.axon_dendrite_forest.get_property_list(property_name, tree_type='axon')
            dendrite_props = self.axon_dendrite_forest.get_property_list(property_name, tree_type='dendrite')
            
            if property_name == 'fractal_dimension':
                axon_props_temp = []
                dendrite_props_temp = []
                for i in range(len(axon_props)):
                    if axon_props[i] >= 1 and dendrite_props[i] >= 1:
                        axon_props_temp.append(axon_props[i])
                        dendrite_props_temp.append(dendrite_props[i])
                axon_props = axon_props_temp
                dendrite_props = dendrite_props_temp
            # Filter non-positive values ONLY if log_scale is requested
            elif log_scale:
                axon_props = [v for v in axon_props if v > 0]
                dendrite_props = [v for v in dendrite_props if v > 0]
            
            all_values = axon_props + dendrite_props
            
            # --- Binning and Plotting ---
            current_bins = bins
            if log_scale and all_values:
                # Use log-spaced bins if X-axis is log
                log_min = np.log10(min(all_values)) if min(all_values) > 0 else 0
                log_max = np.log10(max(all_values))
                current_bins = np.logspace(log_min, log_max, bins)
            
            plt.hist(axon_props, bins=current_bins, alpha=0.5, label='Axon')
            plt.hist(dendrite_props, bins=current_bins, alpha=0.5, label='Dendrite')
            
            # --- Scaling and Labels ---
            if log_scale:
                plt.xscale('log')
                plt.xlabel(f'{property_name} (Log Scale)')
                plt.title(f'Log-Scale Histogram of {property_name}')
            else:
                plt.xlim(self._axis_range(all_values))
                plt.xlabel(property_name)
                plt.title(f'Histogram of {property_name}')

            plt.ylabel('Frequency')
            plt.legend()
            
        else: # tree_type is 'axon' or 'dendrite'
            properties = self.axon_dendrite_forest.get_property_list(property_name, tree_type=tree_type)
            
            if log_scale:
                properties = [v for v in properties if v > 0]

            current_bins = bins
            if log_scale and properties:
                log_min = np.log10(min(properties)) if min(properties) > 0 else 0
                log_max = np.log10(max(properties))
                current_bins = np.logspace(log_min, log_max, bins)

            plt.hist(properties, bins=current_bins)
            
            if log_scale:
                plt.xscale('log')
                plt.xlabel(f'{property_name} (Log Scale)')
                plt.title(f'Log-Scale Histogram of {property_name} for {tree_type} trees')
            else:
                plt.xlim(self._axis_range(properties))
                plt.xlabel(property_name)
                plt.title(f'Histogram of {property_name} for {tree_type} trees')
                
            plt.ylabel('Frequency')


        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()
        plt.close()


    def plot_pairwise(self, property_name_x, property_name_y, log_scale=False, save_path=None):
        axon_props_x = self.axon_dendrite_forest.get_property_list(property_name_x, tree_type='axon')
        dendrite_props_y = self.axon_dendrite_forest.get_property_list(property_name_y, tree_type='dendrite')

        min_len = min(len(axon_props_x), len(dendrite_props_y))

        x_raw = axon_props_x[:min_len]
        y_raw = dendrite_props_y[:min_len]

        if property_name_x == 'fractal_dimension':
            axon_props_temp = []
            dendrite_props_temp = []
            for i in range(len(x_raw)):
                if x_raw[i] >= 1 and y_raw[i] >= 1:
                    axon_props_temp.append(x_raw[i])
                    dendrite_props_temp.append(y_raw[i])
            x_vals = axon_props_temp
            y_vals = dendrite_props_temp
        # Prepare data based on log_scale flag
        elif log_scale:
            # Keep only pairs where both x and y are positive
            x_vals = [x_raw[i] for i in range(min_len) if x_raw[i] > 0 and y_raw[i] > 0]
            y_vals = [y_raw[i] for i in range(min_len) if x_raw[i] > 0 and y_raw[i] > 0]
        else:
            x_vals = x_raw
            y_vals = y_raw

        plt.figure(figsize=(6, 5))
        plt.scatter(x_vals, y_vals, alpha=0.7)
        
        # --- Scaling and Labels ---
        if log_scale:
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(f'Axon {property_name_x} (Log Scale)')
            plt.ylabel(f'Dendrite {property_name_y} (Log Scale)')
            plt.title(f'Pairwise scatter (Log-Log): Axon {property_name_x} vs Dendrite {property_name_y}')
        else:
            # plt.xlim(0, 1000)
            # plt.ylim(0, 1000)
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

    def plot_indexed_pair(self, property_name, log_scale=False, save_path=None):
        axon_values_raw = self.axon_dendrite_forest.get_property_list(property_name, tree_type='axon')
        dendrite_values_raw = self.axon_dendrite_forest.get_property_list(property_name, tree_type='dendrite')
        
        # Prepare data based on log_scale flag

        if property_name == 'fractal_dimension':
            axon_props_temp = []
            dendrite_props_temp = []
            for i in range(len(axon_values_raw)):
                if axon_values_raw[i] >= 1 and dendrite_values_raw[i] >= 1:
                    axon_props_temp.append(axon_values_raw[i])
                    dendrite_props_temp.append(dendrite_values_raw[i])
            axon_values = axon_props_temp
            dendrite_values = dendrite_props_temp
        elif log_scale:
            # Filter non-positive values for log Y-scale
            axon_values = [v for v in axon_values_raw if v > 0]
            dendrite_values = [v for v in dendrite_values_raw if v > 0]
        else:
            axon_values = axon_values_raw
            dendrite_values = dendrite_values_raw
        
        all_values = axon_values + dendrite_values

        plt.figure(figsize=(8, 5))
        # X-axis (Index) remains linear
        plt.scatter(range(len(axon_values)), axon_values, color='red', label='Axon')
        plt.scatter(range(len(dendrite_values)), dendrite_values, color='green', label='Dendrite')
        
        # --- Scaling and Labels ---
        if log_scale:
            plt.yscale('log')
            plt.ylabel(f'{property_name} (Log Scale)')
            plt.title(f'Indexed values (Log Y): {property_name}')
            # Need to manually set Y limits since _axis_range is linear
            plt.ylim(min(v for v in all_values if v > 0) * 0.9, max(all_values) * 1.1)
        else:
            plt.ylim(self._axis_range(all_values))
            plt.ylabel(property_name)
            plt.title(f'Indexed values of {property_name}: Axon (red) & Dendrite (green)')
            
        plt.xlabel('Index (n)')
        plt.legend()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    # NOTE: plot_diff and plot_diff_histogram should remain linear
    # because the difference can be zero or negative, making log scales invalid/misleading.

    def plot_diff(self, property_name, save_path=None, log_scale=False):
        axon_values = self.axon_dendrite_forest.get_property_list(property_name, tree_type='axon')
        dendrite_values = self.axon_dendrite_forest.get_property_list(property_name, tree_type='dendrite')
        
        if property_name == 'fractal_dimension':
            axon_props_temp = []
            dendrite_props_temp = []
            for i in range(len(axon_values)):
                if axon_values[i] >= 1 and dendrite_values[i] >= 1:
                    axon_props_temp.append(axon_values[i])
                    dendrite_props_temp.append(dendrite_values[i])
            axon_values = axon_props_temp
            dendrite_values = dendrite_props_temp
        
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

    def plot_diff_histogram(self, property_name, bins=30, save_path=None, log_scale=False):
        axon_values = self.axon_dendrite_forest.get_property_list(property_name, tree_type='axon')
        dendrite_values = self.axon_dendrite_forest.get_property_list(property_name, tree_type='dendrite')
        
        if property_name == 'fractal_dimension':
            axon_props_temp = []
            dendrite_props_temp = []
            for i in range(len(axon_values)):
                if axon_values[i] >= 1 and dendrite_values[i] >= 1:
                    axon_props_temp.append(axon_values[i])
                    dendrite_props_temp.append(dendrite_values[i])
            axon_values = axon_props_temp
            dendrite_values = dendrite_props_temp

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
        """Compute range for axis with a small margin (for linear axes)."""
        if not values:
            return [0, 1]
        vmin, vmax = min(values), max(values)
        if vmin == vmax:
            return [vmin - 1, vmax + 1]
        margin = (vmax - vmin) * margin_ratio
        return [vmin - margin, vmax + margin]

    def plot_histogram(self, property_name, tree_type, bins=30, log_scale=False, save_path=None):
        
        # --- Data preparation ---
        if tree_type == "both":
            axon_props = self.axon_dendrite_forest.get_property_list(property_name, tree_type='axon')
            dendrite_props = self.axon_dendrite_forest.get_property_list(property_name, tree_type='dendrite')
            
            if log_scale:
                axon_props = [v for v in axon_props if v > 0]
                dendrite_props = [v for v in dendrite_props if v > 0]
            
            all_values = axon_props + dendrite_props

            fig = go.Figure()
            fig.add_trace(go.Histogram(x=axon_props, name='Axon', nbinsx=bins, opacity=0.5))
            fig.add_trace(go.Histogram(x=dendrite_props, name='Dendrite', nbinsx=bins, opacity=0.5))
            
            # --- Scaling and Labels ---
            if log_scale:
                fig.update_xaxes(type='log') 
                fig.update_layout(
                    title_text=f'Log-Scale Histogram of {property_name}',
                    xaxis_title=f'{property_name} (Log Scale)',
                    yaxis_title='Frequency (Linear Scale)',
                    legend_title='Tree Type',
                )
            else:
                 fig.update_layout(
                    title_text=f'Histogram of {property_name}',
                    xaxis_title=property_name,
                    yaxis_title='Frequency',
                    legend_title='Tree Type',
                    xaxis=dict(range=self._axis_range(all_values))
                )
            fig.update_layout(barmode='overlay')
            
        else: # tree_type is 'axon' or 'dendrite'
            properties = self.axon_dendrite_forest.get_property_list(property_name, tree_type=tree_type)
            
            if log_scale:
                properties = [v for v in properties if v > 0]

            fig = go.Figure(data=[go.Histogram(x=properties, nbinsx=bins)])
            
            # --- Scaling and Labels ---
            if log_scale:
                fig.update_xaxes(type='log')
                fig.update_layout(
                    title_text=f'Log-Scale Histogram of {property_name} for {tree_type} trees',
                    xaxis_title=f'{property_name} (Log Scale)',
                    yaxis_title='Frequency (Linear Scale)',
                )
            else:
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

    def plot_pairwise(self, property_name_x, property_name_y, log_scale=False, save_path=None):
        axon_props_x = self.axon_dendrite_forest.get_property_list(property_name_x, tree_type='axon')
        dendrite_props_y = self.axon_dendrite_forest.get_property_list(property_name_y, tree_type='dendrite')
        min_len = min(len(axon_props_x), len(dendrite_props_y))
        
        x_raw = axon_props_x[:min_len]
        y_raw = dendrite_props_y[:min_len]

        # Prepare data based on log_scale flag
        if log_scale:
            x_vals = [x_raw[i] for i in range(min_len) if x_raw[i] > 0 and y_raw[i] > 0]
            y_vals = [y_raw[i] for i in range(min_len) if x_raw[i] > 0 and y_raw[i] > 0]
        else:
            x_vals = x_raw
            y_vals = y_raw
        
        fig = go.Figure(data=go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='markers'
        ))
        
        # --- Scaling and Labels ---
        if log_scale:
            fig.update_xaxes(type='log')
            fig.update_yaxes(type='log')
            fig.update_layout(
                title_text=f'Pairwise scatter (Log-Log): Axon {property_name_x} vs Dendrite {property_name_y}',
                xaxis_title=f'Axon {property_name_x} (Log Scale)',
                yaxis_title=f'Dendrite {property_name_y} (Log Scale)',
            )
        else:
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

    def plot_indexed_pair(self, property_name, log_scale=False, save_path=None):
        axon_values_raw = self.axon_dendrite_forest.get_property_list(property_name, tree_type='axon')
        dendrite_values_raw = self.axon_dendrite_forest.get_property_list(property_name, tree_type='dendrite')
        
        # Prepare data based on log_scale flag
        if log_scale:
            axon_values = [v for v in axon_values_raw if v > 0]
            dendrite_values = [v for v in dendrite_values_raw if v > 0]
        else:
            axon_values = axon_values_raw
            dendrite_values = dendrite_values_raw
        
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
        
        # --- Scaling and Labels ---
        if log_scale:
            fig.update_yaxes(type='log')
            fig.update_layout(
                title_text=f'Indexed values (Log Y): {property_name}',
                xaxis_title='Index (n) (Linear Scale)',
                yaxis_title=f'{property_name} (Log Scale)',
                legend_title='Tree Type',
            )
        else:
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

    def plot_diff(self, property_name, save_path=None, log_scale=False):
        # Linear scale is mandatory for difference plots
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

    def plot_diff_histogram(self, property_name, bins=30, save_path=None, log_scale=False):
        # Linear scale is mandatory for difference plots
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

    log_scale = False
    if log_scale:
        foldername = "figures_log_scale"
    else:
        foldername = "figures"

    if sys.platform == "darwin":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.abspath(os.path.join(script_dir, f"../../{foldername}"))
    else:
        res_dir = "/data/RESULTS/USERS/bea/drosophila/"
        save_dir = os.path.abspath(os.path.join(res_dir, f"{foldername}"))

    print(properties)
    for property in properties:
        comparator.plot_histogram(property, tree_type='both', bins=100, save_path=generate_savefig_path(property, tree_type, 'histogram', save_dir, html=False), log_scale=log_scale)
        comparator.plot_pairwise(property, property, save_path=generate_savefig_path(property, tree_type, 'pairwise', save_dir, html=False), log_scale=log_scale)
        comparator.plot_indexed_pair(property, save_path=generate_savefig_path(property, tree_type, 'indexed_pair', save_dir, html=False), log_scale=log_scale)
        comparator.plot_diff(property, save_path=generate_savefig_path(property, tree_type, 'diff', save_dir, html=False), log_scale=log_scale)
        comparator.plot_diff_histogram(property, bins=100, save_path=generate_savefig_path(property, tree_type, 'diff_histogram', save_dir, html=False), log_scale=log_scale)

        # comparator_plotly.plot_histogram(property, tree_type='both', bins=100, save_path=generate_savefig_path(property, tree_type, 'histogram', save_dir, html=True), log_scale=log_scale)
        # comparator_plotly.plot_pairwise(property, property, save_path=generate_savefig_path(property, tree_type, 'pairwise', save_dir, html=True), log_scale=log_scale)
        # comparator_plotly.plot_indexed_pair(property, save_path=generate_savefig_path(property, tree_type, 'indexed_pair', save_dir, html=True), log_scale=log_scale)
        # comparator_plotly.plot_diff(property, save_path=generate_savefig_path(property, tree_type, 'diff', save_dir, html=True), log_scale=log_scale)
        # comparator_plotly.plot_diff_histogram(property, bins=100, save_path=generate_savefig_path(property, tree_type, 'diff_histogram', save_dir, html=True), log_scale=log_scale)

        