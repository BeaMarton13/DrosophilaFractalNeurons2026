from axon_dendrite_tree import AxonDendriteForest

import os
import matplotlib.pyplot as plt
import sys

import plotly.graph_objects as go
import numpy as np
from scipy.stats import ttest_ind

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
        """Compute min–max range with small margin for linear axes."""
        if not values:
            return [0, 1]
        vmin, vmax = min(values), max(values)
        if vmin == vmax:
            return [vmin - 1, vmax + 1]
        margin = (vmax - vmin) * margin_ratio
        return [vmin - margin, vmax + margin]
    
    def plot_histogram(self, property, property_name, tree_type, bins=100, log_scale_x=False, log_scale_y=False, save_path=None, x_min=None, x_max=None):
        plt.figure(figsize=(10, 6))

        # --- Data preparation ---
        if tree_type == "both":
            axon_props_raw = self.axon_dendrite_forest.get_property_list(property, tree_type='axon')
            dendrite_props_raw = self.axon_dendrite_forest.get_property_list(property, tree_type='dendrite')

            axon_props = []
            dendrite_props = []

            # Fractal dimension filtering (retains pairs where either is >= 1)
            if property == 'fractal_dimension':
                for i in range(len(axon_props_raw)):
                    # Added check for list bounds for safety, though typically they should be the same length
                    if i < len(dendrite_props_raw) and \
                       (axon_props_raw[i] >= 1 and dendrite_props_raw[i] >= 1) and \
                       (axon_props_raw[i] <= 3 and dendrite_props_raw[i] <= 3):
                        axon_props.append(axon_props_raw[i])
                        dendrite_props.append(dendrite_props_raw[i])
            else:
                axon_props = axon_props_raw
                dendrite_props = dendrite_props_raw

            # Compute t-test
            t_stat, p_value = ttest_ind(axon_props, dendrite_props, equal_var=False, nan_policy='omit')
            
            if p_value == 0 or p_value < 1e-10:
                p_text = "p < 10^-10"
            elif p_value < 1e-3:
                exp = int(np.floor(np.log10(p_value)))
                p_text = f"p < 10^{exp}"
            else:
                p_text = f"p = {p_value:.3f}"

            print(f"p = {p_value:.17f}")


            all_values = axon_props + dendrite_props

            # Filter non-positive values ONLY if X-axis is log
            if log_scale_x:
                all_values = [v for v in all_values if v > 0]
                axon_props = [v for v in axon_props if v > 0]
                dendrite_props = [v for v in dendrite_props if v > 0]

            # --- Binning ---
            current_bins = bins
            if log_scale_x and all_values:
                # Use log spaced bins if X-axis is log
                log_min = np.log10(x_min) if x_min is not None and x_min > 0 else np.log10(min(v for v in all_values if v > 0)) if any(v > 0 for v in all_values) else 0
                log_max = np.log10(x_max) if x_max is not None else np.log10(max(all_values))
                current_bins = np.logspace(log_min, log_max, bins)
                # --- ADD A CHECK HERE ---
                if log_max <= log_min:
                    # If log_max is not greater than log_min, the bins will not be monotonic.
                    # This happens if max(all_values) is very close to min(all_values) 
                    # or if x_max is set too low. Revert to simple bins or handle the error.
                    current_bins = bins # Revert to integer bin count
                else:
                    current_bins = np.logspace(log_min, log_max, bins)
            
            # --- Plotting ---
            plt.hist(axon_props, bins=current_bins, alpha=0.5, label='Axon')
            plt.hist(dendrite_props, bins=current_bins, alpha=0.5, label='Dendrite')

            # --- Scaling and Labels ---
            title_parts = ['Histogram']
            x_label = property_name
            y_label = 'Frequency'

            if log_scale_x:
                plt.xscale('log')
            # --- X-axis Limit Setting (Both) ---
            if x_min is not None or x_max is not None:
                plt.xlim(x_min, x_max)
            elif not log_scale_x:
                # Only use the default _axis_range if no custom limits and not log scale
                plt.xlim(self._axis_range(all_values))
            
            if log_scale_y:
                plt.yscale('log')

            # --- FONT SIZE MODIFICATIONS ---
            plt.xlabel(x_label, fontsize=20)   # Set X-axis label font size
            plt.ylabel(y_label, fontsize=20)   # Set Y-axis label font size
            # plt.title(f"{' '.join(title_parts)} of {property_name}", fontsize=24) # Set Title font size
            
            
            plt.title(f"{' '.join(title_parts)} of {property_name}\nT-test \n{p_text}", fontsize=24) # Set Title font size with p-value
            plt.legend(fontsize=16)            # Set Legend font size
            
        else: # tree_type is 'axon' or 'dendrite'
            properties_raw = self.axon_dendrite_forest.get_property_list(property, tree_type=tree_type)
            properties = properties_raw

            # Filter non-positive values ONLY if X-axis is log
            if log_scale_x:
                properties = [v for v in properties_raw if v > 0]

            # --- Binning ---
            current_bins = bins
            if log_scale_x and properties:
                # Use x_min/x_max for bin range if provided
                log_min = np.log10(x_min) if x_min is not None and x_min > 0 else np.log10(min(v for v in properties if v > 0)) if any(v > 0 for v in properties) else 0
                log_max = np.log10(x_max) if x_max is not None else np.log10(max(properties))
                current_bins = np.logspace(log_min, log_max, bins)

            # --- Plotting ---
            plt.hist(properties, bins=current_bins)

            # --- Scaling and Labels ---
            title_parts = ['Histogram']
            x_label = property_name
            y_label = 'Frequency'

            if log_scale_x:
                plt.xscale('log')
            
            # --- X-axis Limit Setting (Single) ---
            if x_min is not None or x_max is not None:
                plt.xlim(x_min, x_max)
            elif not log_scale_x:
                # Only use the default _axis_range if no custom limits and not log scale
                plt.xlim(self._axis_range(properties))


            if log_scale_y:
                plt.yscale('log')

            # --- FONT SIZE MODIFICATIONS ---
            plt.xlabel(x_label, fontsize=20)   # Set X-axis label font size
            plt.ylabel(y_label, fontsize=20)   # Set Y-axis label font size
            plt.title(f"{' '.join(title_parts)} of {property_name} for {tree_type} trees", fontsize=24) # Set Title font size

        plt.tick_params(axis='both', which='major', labelsize=16)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

# -----------------------------------------------------------------------------
    def plot_pairwise(
        self,
        property_x,
        property_y,
        property_name_x,
        property_name_y,
        log_scale_x=False,
        log_scale_y=False,
        save_path=None,
        x_min=None,
        x_max=None,
    ):
        # --- Data Retrieval ---
        axon_props_x = self.axon_dendrite_forest.get_property_list(property_x, tree_type='axon')
        dendrite_props_y = self.axon_dendrite_forest.get_property_list(property_y, tree_type='dendrite')

        min_len = min(len(axon_props_x), len(dendrite_props_y))
        x_raw = axon_props_x[:min_len]
        y_raw = dendrite_props_y[:min_len]

        # --- Fractal Dimension Filtering ---
        if property_x == 'fractal_dimension':
            x_temp, y_temp = [], []
            for i in range(len(x_raw)):
                if (1 <= x_raw[i] <= 3) and (1 <= y_raw[i] <= 3):
                    x_temp.append(x_raw[i])
                    y_temp.append(y_raw[i])
            x_vals, y_vals = x_temp, y_temp
        else:
            x_vals, y_vals = x_raw, y_raw

        # --- Log Filtering (only keep positive pairs if log is applied) ---
        if log_scale_x or log_scale_y:
            filtered_pairs = [(x, y) for x, y in zip(x_vals, y_vals) if x > 0 and y > 0]
            if filtered_pairs:
                x_vals, y_vals = zip(*filtered_pairs)
                x_vals, y_vals = list(x_vals), list(y_vals)
            else:
                x_vals, y_vals = [], []

        # --- Plotting ---
        plt.figure(figsize=(10, 8))
        plt.scatter(x_vals, y_vals, alpha=0.7)

        # --- Titles and Labels ---
        title_parts = ['Pairwise scatter']
        x_label = f'Axon {property_name_x}'
        y_label = f'Dendrite {property_name_y}'

        # --- X-axis Scaling and Range ---
        if log_scale_x:
            plt.xscale('log')
        else:
            # Apply manual limits if provided
            if x_min is not None and x_max is not None:
                plt.ylim(x_min, x_max)
            else:
                plt.ylim(self._axis_range(y_vals))

        # --- Y-axis Scaling and Range ---
        if log_scale_y:
            plt.yscale('log')
        else:
            if x_min is not None and x_max is not None:
                plt.xlim(x_min, x_max)
                plt.ylim(x_min, x_max)
            else:
                plt.xlim(self._axis_range(x_vals))
                plt.ylim(self._axis_range(y_vals))

        # --- Y-axis Scaling and Range ---
        if log_scale_y:
            plt.yscale('log')
        else:
            if x_min is not None and x_max is not None:
                plt.ylim(x_min, x_max)
            else:
                plt.ylim(self._axis_range(y_vals))

        # --- Font sizes ---
        plt.xlabel(x_label, fontsize=20)
        plt.ylabel(y_label, fontsize=20)
        plt.title(f"{' '.join(title_parts)}: Axon {property_name_x} vs Dendrite {property_name_y}", fontsize=24)
        plt.gca().tick_params(axis='both', which='major', labelsize=16)

        plt.tick_params(axis='both', which='major', labelsize=16)
        # --- Save or Show ---
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

# -----------------------------------------------------------------------------
    def plot_indexed_pair(self, property, property_name, log_scale_x=False, log_scale_y=False, save_path=None):
        # NOTE: log_scale_x is not relevant here as the X-axis is the integer index (n).
        axon_values_raw = self.axon_dendrite_forest.get_property_list(property, tree_type='axon')
        dendrite_values_raw = self.axon_dendrite_forest.get_property_list(property, tree_type='dendrite')
        
        axon_values = []
        dendrite_values = []

        # Apply fractal dimension filtering
        if property == 'fractal_dimension':
            for i in range(len(axon_values_raw)):
                if (axon_values_raw[i] >= 1 and dendrite_values_raw[i] >= 1) and (axon_values_raw[i] <= 3 and dendrite_values_raw[i] <= 3):
                    axon_values.append(axon_values_raw[i])
                    dendrite_values.append(dendrite_values_raw[i])
        else:
            axon_values = axon_values_raw
            dendrite_values = dendrite_values_raw
        
        # Conditional filtering for log Y-scale
        if log_scale_y:
            # Keep only positive values for the Y-axis
            axon_values = [v for v in axon_values if v > 0]
            dendrite_values = [v for v in dendrite_values if v > 0]
        
        all_values = axon_values + dendrite_values

        plt.figure(figsize=(8, 5))
        # X-axis (Index) remains linear
        plt.scatter(range(len(axon_values)), axon_values, color='red', label='Axon')
        plt.scatter(range(len(dendrite_values)), dendrite_values, color='green', label='Dendrite')
        
        # --- Scaling and Labels ---
        title_parts = ['Indexed values']
        y_label = property_name

        if log_scale_y:
            plt.yscale('log')
            # Set Y limits for log scale
            if all_values:
                plt.ylim(min(v for v in all_values if v > 0) * 0.9, max(all_values) * 1.1)
        else:
            plt.ylim(self._axis_range(all_values))
        
        plt.xlabel('Index (n)')
        plt.ylabel(y_label)
        plt.title(f"{' '.join(title_parts)} of {property_name}: Axon (red) & Dendrite (green)")
        plt.legend()

        plt.tick_params(axis='both', which='major', labelsize=16)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

# -----------------------------------------------------------------------------
    # NOTE: plot_diff and plot_diff_histogram should remain linear
    # as the difference can be zero or negative, making log scales invalid/misleading.

    def plot_diff(self, property, property_name, log_scale_x=False, log_scale_y=False, save_path=None):
        axon_values = self.axon_dendrite_forest.get_property_list(property, tree_type='axon')
        dendrite_values = self.axon_dendrite_forest.get_property_list(property, tree_type='dendrite')
        
        if property == 'fractal_dimension':
            axon_props_temp = []
            dendrite_props_temp = []
            for i in range(len(axon_values)):
                if (axon_values[i] >= 1 and dendrite_values[i] >= 1) and (axon_values[i] <= 3 and dendrite_values[i] <= 3):
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

        plt.tick_params(axis='both', which='major', labelsize=16)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def plot_diff_histogram(self, property, property_name, log_scale_x=False, log_scale_y=False, bins=30, save_path=None):
        axon_values = self.axon_dendrite_forest.get_property_list(property, tree_type='axon')
        dendrite_values = self.axon_dendrite_forest.get_property_list(property, tree_type='dendrite')
        
        if property == 'fractal_dimension':
            axon_props_temp = []
            dendrite_props_temp = []
            for i in range(len(axon_values)):
                if (axon_values[i] >= 1 and dendrite_values[i] >= 1) and (axon_values[i] <= 3 and dendrite_values[i] <= 3):
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

        plt.tick_params(axis='both', which='major', labelsize=16)
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
                    title_text=f'Histogram of {property_name}',
                    xaxis_title=f'{property_name}',
                    yaxis_title='Frequency',
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
                    title_text=f'Histogram of {property_name} for {tree_type} trees',
                    xaxis_title=f'{property_name}',
                    yaxis_title='Frequency',
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
            pairs = [(a, d) for a, d in zip(x_raw, y_raw) if a > 0 and d > 0]
            x_vals, y_vals = zip(*pairs)
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
                title_text=f'Pairwise scatter: Axon {property_name_x} vs Dendrite {property_name_y}',
                xaxis_title=f'Axon {property_name_x}',
                yaxis_title=f'Dendrite {property_name_y}',
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
            pairs = [(a, d) for a, d in zip(axon_values_raw, dendrite_values_raw)
                    if a > 0 and d > 0]
            axon_values, dendrite_values = zip(*pairs)
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
                title_text=f'Indexed values: {property_name}',
                xaxis_title='Index (n)',
                yaxis_title=f'{property_name}',
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

    log_scale_x = False
    log_scale_y = False
    if log_scale_x and log_scale_y:
        foldername = "figures_log_scale_xy"
    elif log_scale_x:
        foldername = "figures_log_scale_x"
    elif log_scale_y:
        foldername = "figures_log_scale_y"
    else:
        foldername = "figures"

    if sys.platform == "darwin":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.abspath(os.path.join(script_dir, f"../../{foldername}"))
    else:
        res_dir = "/data/RESULTS/USERS/bea/drosophila/"
        save_dir = os.path.abspath(os.path.join(res_dir, f"{foldername}"))

    property_to_name = {
        "fractal_dimension": "Fractal Dimension",
        "num_nodes": "Number of Nodes",
        "num_edges": "Number of Edges",
        "num_leaf_nodes": "Number of Leaf Nodes",
        "height": "Height",
        "max_width": "Maximum Width",
        "cable_length": "Cable Length (nm)",
        "num_filtered_pre_synapses": "Number of Pre-synapses",
        "num_filtered_post_synapses": "Number of Post-synapses",
        "c_value": "C-value"
    }

    print(properties)
    for property in properties:
        comparator.plot_histogram(property, property_to_name[property], tree_type='both', bins=100, save_path=generate_savefig_path(property, tree_type, 'histogram', save_dir, html=False), log_scale_x=log_scale_x, log_scale_y=log_scale_y, x_min=10)
        comparator.plot_pairwise(property, property, property_to_name[property], property_to_name[property], property, save_path=generate_savefig_path(property, tree_type, 'pairwise', save_dir, html=False), log_scale_x=log_scale_x, log_scale_y=log_scale_y)
        comparator.plot_indexed_pair(property, property_to_name[property], save_path=generate_savefig_path(property, tree_type, 'indexed_pair', save_dir, html=False), log_scale_x=log_scale_x, log_scale_y=log_scale_y)
        comparator.plot_diff(property, property_to_name[property], save_path=generate_savefig_path(property, tree_type, 'diff', save_dir, html=False), log_scale_x=log_scale_x, log_scale_y=log_scale_y)
        comparator.plot_diff_histogram(property, property_to_name[property], bins=100, save_path=generate_savefig_path(property, tree_type, 'diff_histogram', save_dir, html=False), log_scale_x=log_scale_x, log_scale_y=log_scale_y)

        # comparator_plotly.plot_histogram(property, tree_type='both', bins=100, save_path=generate_savefig_path(property, tree_type, 'histogram', save_dir, html=True), log_scale=log_scale)
        # comparator_plotly.plot_pairwise(property, property, save_path=generate_savefig_path(property, tree_type, 'pairwise', save_dir, html=True), log_scale=log_scale)
        # comparator_plotly.plot_indexed_pair(property, save_path=generate_savefig_path(property, tree_type, 'indexed_pair', save_dir, html=True), log_scale=log_scale)
        # comparator_plotly.plot_diff(property, save_path=generate_savefig_path(property, tree_type, 'diff', save_dir, html=True), log_scale=log_scale)
        # comparator_plotly.plot_diff_histogram(property, bins=100, save_path=generate_savefig_path(property, tree_type, 'diff_histogram', save_dir, html=True), log_scale=log_scale)

        

# import os
# import sys
# import numpy as np
# import matplotlib.pyplot as plt
# import plotly.graph_objects as go
# from scipy.stats import ttest_ind


# # ==========================================================
# # Helper functions
# # ==========================================================

# def get_save_dir(foldername):
#     """
#     Determine save directory based on platform.
#     """
#     base = (
#         "/data/RESULTS/USERS/bea/drosophila/"
#         if sys.platform != "darwin"
#         else os.path.dirname(os.path.abspath(__file__))
#     )
#     return os.path.abspath(os.path.join(base, f"{foldername}"))


# def filter_fractal_dimension(axon, dendrite):
#     """
#     Keep only pairs where both fractal dimensions are within [1, 3].
#     """
#     return [(a, d) for a, d in zip(axon, dendrite) if 1 <= a <= 3 and 1 <= d <= 3]


# def generate_savefig_path(property_name, tree_type, plot_type, save_dir, html=False):
#     os.makedirs(save_dir, exist_ok=True)
#     if html:
#         filename = f"{property_name}_{tree_type}_{plot_type}.html"
#     else:
#         filename = f"{property_name}_{tree_type}_{plot_type}.png"
#     return os.path.join(save_dir, filename)


# # ==========================================================
# # Matplotlib-based plotting
# # ==========================================================

# class CompareProperties:
#     """
#     Class for comparing neuronal axon vs dendrite properties using Matplotlib.
#     """

#     def __init__(self):
#         pass
#         # self.save_dir = save_dir
#         # os.makedirs(save_dir, exist_ok=True) if save_dir else None

#     # ------------------------------------------------------
    

#     @staticmethod
#     def _axis_range(values, margin=0.05):
#         """
#         Compute axis limits with small margins.
#         """
#         if values is None or len(values) == 0:
#             return [0, 1]
#         vmin, vmax = np.min(values), np.max(values)
#         delta = (vmax - vmin) * margin
#         return [vmin - delta, vmax + delta]

#     # ------------------------------------------------------

#     def plot_histogram(
#         self,
#         axon_props,
#         dend_props,
#         title,
#         xlabel,
#         bins=100,
#         log_scale_x=False,
#         log_scale_y=False,
#         density_for_log=True,
#         save_path=None,
#     ):
#         """
#         Plot histogram of properties (Axon vs Dendrite).

#         log_scale_x/y toggle axis scaling.
#         density_for_log normalizes counts for log X.
#         """
#         # Convert to NumPy first (so len() works)
#         axon_props = np.array(axon_props)
#         dend_props = np.array(dend_props)

#         # Compute t-test
#         t_stat, p_value = ttest_ind(axon_props, dend_props, equal_var=False, nan_policy='omit')


#         # ✅ SAFER check for empty data
#         if axon_props.size == 0 or dend_props.size == 0:
#             print("Warning: empty property arrays provided.")
#             return

#         # Handle fractal dimension restriction
#         if "fractal" in xlabel.lower():
#             pairs = filter_fractal_dimension(axon_props, dend_props)
#             if not pairs:
#                 print("Warning: No valid fractal dimension pairs found.")
#                 return
#             axon_props, dend_props = zip(*pairs)

#         # Convert to numpy for safe processing
#         axon_props, dend_props = np.array(axon_props), np.array(dend_props)

#         # Filter out invalid values for log scaling
#         valid_axon = axon_props > 0 if log_scale_x else np.ones_like(axon_props, dtype=bool)
#         valid_dend = dend_props > 0 if log_scale_x else np.ones_like(dend_props, dtype=bool)
#         axon_props, dend_props = axon_props[valid_axon], dend_props[valid_dend]

#         # Axis range and bins
#         all_values = np.concatenate([axon_props, dend_props])
#         x_min, x_max = self._axis_range(all_values)
#         eps = 1e-10
#         if log_scale_x:
#             log_min, log_max = np.log10(max(x_min, eps)), np.log10(max(x_max, eps))
#             current_bins = np.logspace(log_min, log_max, bins) if log_max > log_min else bins
#         else:
#             current_bins = bins

#         # Plot
#         plt.figure(figsize=(8, 5))
#         plt.hist(
#             axon_props,
#             bins=current_bins,
#             alpha=0.5,
#             label="Axon",
#             edgecolor="black",
#             density=density_for_log if log_scale_x else False,
#         )
#         plt.hist(
#             dend_props,
#             bins=current_bins,
#             alpha=0.5,
#             label="Dendrite",
#             edgecolor="black",
#             density=density_for_log if log_scale_x else False,
#         )

#         plt.xlabel(f"{xlabel}{' (Log Scale)' if log_scale_x else ''}", fontsize=14)
#         plt.ylabel(f"Frequency{' (Log Scale)' if log_scale_y else ''}", fontsize=14)
#         plt.title(f"{title}\nT-test p = {p_value:.3e}", fontsize=16)
#         plt.legend()

#         if log_scale_x:
#             plt.xscale("log")
#         if log_scale_y:
#             plt.yscale("log")

#         plt.tight_layout()
#         if save_path:
#             plt.savefig(save_path, dpi=300)
#             plt.close()
#         else:
#             plt.show()

#     # ------------------------------------------------------

#     def plot_scatter(
#         self,
#         axon_props,
#         dend_props,
#         title,
#         xlabel,
#         ylabel,
#         log_scale_x=False,
#         log_scale_y=False,
#         save_path=None,
#     ):
#         """
#         Scatter plot comparing axon and dendrite properties.
#         """
#         if "fractal" in xlabel.lower():
#             pairs = filter_fractal_dimension(axon_props, dend_props)
#             if not pairs:
#                 print("Warning: No valid fractal dimension pairs found.")
#                 return
#             axon_props, dend_props = zip(*pairs)

#         x_vals, y_vals = np.array(axon_props), np.array(dend_props)
#         # Filter zeros for relevant axes
#         mask = (x_vals > 0 if log_scale_x else True) & (y_vals > 0 if log_scale_y else True)
#         x_vals, y_vals = x_vals[mask], y_vals[mask]

#         plt.figure(figsize=(6, 6))
#         plt.scatter(x_vals, y_vals, alpha=0.5, s=10)
#         plt.plot([min(x_vals), max(x_vals)], [min(x_vals), max(x_vals)], 'r--', lw=1)
#         plt.xlabel(xlabel, fontsize=14)
#         plt.ylabel(ylabel, fontsize=14)
#         plt.title(title, fontsize=16)

#         if log_scale_x:
#             plt.xscale("log")
#         if log_scale_y:
#             plt.yscale("log")

#         plt.axis("equal")
#         plt.tight_layout()
#         if save_path:
#             plt.savefig(save_path, dpi=300)
#             plt.close()
#         else:
#             plt.show()

#     # ------------------------------------------------------

#     def plot_diff(self, axon_props, dend_props, title, xlabel, save_path=None):
#         """
#         Plot histogram of differences (Dendrite - Axon).
#         """
#         diffs = np.array(dend_props) - np.array(axon_props)
#         plt.figure(figsize=(8, 5))
#         plt.hist(diffs, bins=100, alpha=0.7, color="gray", edgecolor="black")
#         plt.title(title, fontsize=16)
#         plt.xlabel(xlabel, fontsize=14)
#         plt.ylabel("Frequency", fontsize=14)
#         plt.tight_layout()

#         if save_path:
#             plt.savefig(save_path, dpi=300)
#             plt.close()
#         else:
#             plt.show()

#     # ------------------------------------------------------

#     def plot_diff_histogram(self, axon_props, dend_props, property_name, save_path=None):
#         """
#         Plot histogram of absolute differences.
#         """
#         diffs = np.abs(np.array(dend_props) - np.array(axon_props))
#         plt.figure(figsize=(8, 5))
#         plt.hist(diffs, bins=100, alpha=0.7, color="teal", edgecolor="black")
#         plt.title(f"Histogram of |Dendrite - Axon| ({property_name})", fontsize=16)
#         plt.xlabel(property_name, fontsize=14)
#         plt.ylabel("Frequency", fontsize=14)
#         plt.tight_layout()

#         if save_path:
#             plt.savefig(save_path, dpi=300)
#             plt.close()
#         else:
#             plt.show()


# # ==========================================================
# # Plotly version (interactive)
# # ==========================================================

# class ComparePropertiesPlotly:
#     """
#     Interactive version using Plotly for exploration.
#     """

#     @staticmethod
#     def scatter(axon, dend, title, xlabel, ylabel, log_scale_x=False, log_scale_y=False):
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(
#             x=axon, y=dend, mode='markers',
#             marker=dict(size=5, opacity=0.5),
#             name='Points'
#         ))
#         fig.add_trace(go.Scatter(
#             x=axon, y=axon,
#             mode='lines', line=dict(color='red', dash='dash'),
#             name='y=x'
#         ))
#         fig.update_layout(
#             title=title,
#             xaxis_title=xlabel,
#             yaxis_title=ylabel,
#             xaxis_type='log' if log_scale_x else 'linear',
#             yaxis_type='log' if log_scale_y else 'linear',
#             width=700, height=700
#         )
#         fig.show()


# # ==========================================================
# # Example usage block
# # ==========================================================

# if __name__ == "__main__":
#     # Example data for demonstration
#     tree_type = "undirected"  # or "undirected"


#     if sys.platform == "darwin":
#         script_dir = os.path.dirname(os.path.abspath(__file__))
#         tree_dir = os.path.abspath(os.path.join(script_dir, f"../../data/{tree_type}/tree_properties/"))
#         c_dir = os.path.abspath(os.path.join(script_dir, f"../../data/{tree_type}/c_values/"))
#     else:
#         res_dir = "/data/RESULTS/USERS/bea/drosophila/"
#         tree_dir = os.path.abspath(os.path.join(res_dir, f"data/{tree_type}/tree_properties/"))
#         c_dir = os.path.abspath(os.path.join(res_dir, f"data/{tree_type}/c_values/"))

#     forest = AxonDendriteForest.build_forest_from_directory(tree_dir, c_dir, undirected=tree_type=='undirected')

#     plotter = CompareProperties()
#     comparator_plotly = ComparePropertiesPlotly()

#     properties = forest.axon_tree[0].get_all_properties()

#     log_scale_x = True
#     log_scale_y = True
#     if log_scale_x and log_scale_y:
#         foldername = "figures_log_scale_xy"
#     elif log_scale_x:
#         foldername = "figures_log_scale_x"
#     elif log_scale_y:
#         foldername = "figures_log_scale_y"
#     else:
#         foldername = "figures"

#     if sys.platform == "darwin":
#         script_dir = os.path.dirname(os.path.abspath(__file__))
#         save_dir = os.path.abspath(os.path.join(script_dir, f"../../{foldername}"))
#     else:
#         res_dir = "/data/RESULTS/USERS/bea/drosophila/"
#         save_dir = os.path.abspath(os.path.join(res_dir, f"{foldername}"))
#     for property in properties:
#         axon = forest.get_property_list(property, tree_type="axon")
#         dend = forest.get_property_list(property, tree_type="dendrite")
#         plotter.plot_histogram(
#             axon, dend,
#             title=f"Histogram of {property}",
#             xlabel=f"{property}",
#             bins=100,
#             log_scale_x=log_scale_x,
#             log_scale_y=log_scale_y,
#             save_path=generate_savefig_path(property, tree_type, 'histogramsss', save_dir, html=False),
#         )

#         plotter.plot_scatter(
#             axon, dend,
#             title=f"Axon vs Dendrite ({property})",
#             xlabel=f"Axon {property}",
#             ylabel=f"Dendrite {property}",
#             log_scale_x=True,
#             log_scale_y=True,
#             save_path=generate_savefig_path(property, tree_type, 'indexed_pairsss', save_dir, html=False),
#         )

#         plotter.plot_diff(
#             axon, dend,
#             title="Difference (Dendrite - Axon)",
#             xlabel=f"{property}",
#             save_path=generate_savefig_path(property, tree_type, 'diffsss', save_dir, html=False),
#         )

#         plotter.plot_diff_histogram(
#             axon, dend,
#             property_name=f"{property}",
#             save_path=generate_savefig_path(property, tree_type, 'diff_histogramsss', save_dir, html=False),
#         )

#         print(f"Plots saved to: {save_dir}")
