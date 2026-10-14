"""Plotting and statistical comparison utilities for axon vs dendrite tree properties."""
from axon_dendrite_tree import AxonDendriteForest
import os


# Disable internal parallelism to prevent subprocess spawning
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import navis

navis.set_pbars(hide=True)  # Disable progress bars
navis.config.parallel = False 

# Clear navis cache if it exists
if hasattr(navis, 'cache'):
    navis.cache.clear()
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind, pearsonr
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines

# The six classes for which histograms/plots get the afferent (sensory, optic,
# ascending) vs efferent (descending, endocrine, motor) treatment.
AFFERENT_EFFERENT_ORDER = ["sensory", "optic", "ascending", "descending", "endocrine", "motor"]

# The five "intrinsic" classes. Neurons classified as sensory don't belong among
# them, so whenever exactly this set is being plotted, sensory neurons are dropped
# completely rather than just hidden/unlabeled.
INTRINSIC_CLASSES = ["optic", "central", "visual_projection", "visual_centrifugal", "sensory"]


def is_intrinsic_class_set(super_classes):
    """True when `super_classes`' keys are exactly the five intrinsic classes."""
    if not super_classes:
        print("Warning: super_classes is empty or None. Cannot determine if it's the intrinsic class set.")
        return False
    normalized = {str(c).strip().lower() for c in super_classes.keys()}
    print("Normalized super classes:", normalized)
    return normalized == set(INTRINSIC_CLASSES)


def drop_sensory_if_intrinsic(super_classes):
    """
    When `super_classes` is exactly the five intrinsic classes, drop 'sensory'
    from the dict entirely. Every function here builds its per-class point
    groups (and any pooled correlation/regression/clustering) by filtering
    trees down to `super_classes.keys()`, so removing the key here means
    sensory neurons are excluded from all of those calculations, not just
    from the legend/plotted classes.
    """
    if not is_intrinsic_class_set(super_classes):
        return super_classes
    normalized = {str(c).strip().lower(): c for c in super_classes.keys()}
    sensory_key = normalized["sensory"]
    print("Dropping sensory neurons from intrinsic class set for plotting and statistics.")
    return {k: v for k, v in super_classes.items() if k != sensory_key}


def _is_sensory_tree(tree):
    return str(getattr(tree, 'class_', '')).strip().lower() == 'sensory'


def generate_savefig_path(property_name, tree_type, plot_type, save_dir, html=False, property_name2=None, axon_or_dendrite=None):
    os.makedirs(save_dir, exist_ok=True)
    if property_name2 is not None:
        if axon_or_dendrite is not None:
            propname = f'{property_name}_vs_{property_name2}_{axon_or_dendrite}'
        else:
            propname = f'{property_name}_vs_{property_name2}'
    else:
        if axon_or_dendrite is not None:
            propname = f'{property_name}_{axon_or_dendrite}'
        else:
            propname = f'{property_name}'
    if html:
        filename = f"{propname}_{tree_type}_{plot_type}.html"
    else:
        filename = f"{propname}_{tree_type}_{plot_type}.png"
    return os.path.join(save_dir, filename)


class CompareProperties:
    """Plots and statistics comparing axon/dendrite tree properties for an AxonDendriteForest."""

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
        return [max(0, vmin - margin), vmax + margin]

    def _group_boundary_line(self, center_a, center_b, x_corr, y_corr, log_scale_x, log_scale_y, n_points=200):
        """
        Perpendicular bisector between two centers in correlation (log) space —
        the set of points exactly mid-distance between them — converted back to
        data space for plotting. Returns None if the centers coincide.
        """
        center_a = np.asarray(center_a, dtype=float)
        center_b = np.asarray(center_b, dtype=float)
        diff = center_b - center_a
        if np.linalg.norm(diff) == 0:
            return None

        midpoint = (center_a + center_b) / 2.0
        perp_direction = np.array([-diff[1], diff[0]])
        perp_direction = perp_direction / np.linalg.norm(perp_direction)

        x_span = x_corr.max() - x_corr.min()
        y_span = y_corr.max() - y_corr.min()
        half_len = max(np.hypot(x_span, y_span), 1e-6)
        t_vals = np.linspace(-half_len, half_len, n_points)

        line_x_corr = midpoint[0] + t_vals * perp_direction[0]
        line_y_corr = midpoint[1] + t_vals * perp_direction[1]

        line_x = 10 ** line_x_corr if log_scale_x else line_x_corr
        line_y = 10 ** line_y_corr if log_scale_y else line_y_corr
        return line_x, line_y

    def calculate_cliff_delta(self, x, y):
        """Calculate Cliff's Delta effect size between two samples."""
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        # Broadcasting: creates n_x × n_y comparison matrix
        comparisons = x_arr[:, np.newaxis] - y_arr
        greater = np.sum(comparisons > 0)
        less = np.sum(comparisons < 0)
        # Calculate Cliff's Delta
        delta = (greater - less) / (len(x_arr) * len(y_arr))
        return delta

    def plot_multi_class_violin(self, property_name, bins=100, log_scale_y=False, save_path=None, y_min=None, y_max=None, super_classes: dict = None, plot_multiple: bool = False, given_super_class: str = None):
        """
        Plots the distribution (violin plot) of a specified tree property
        for multiple neuronal classes, showing Axon and Dendrite trees separately.
        """
        super_classes = drop_sensory_if_intrinsic(super_classes)

        # --- 1. Data Aggregation ---
        data = []
        full_data = []
        for i, class_name in enumerate(super_classes.keys()):
            try:
                axon_props = list(super_classes[class_name]['axon_props'])
                dendrite_props = list(super_classes[class_name]['dendrite_props'])
                full_props = list(super_classes[class_name]['full_props'])
            except KeyError:
                print(f"Warning: Missing axon/dendrite data for class '{class_name}'. Skipping.")
                continue

            for val in axon_props:
                if type(class_name) == str:
                    class_name = class_name.replace("_", " ")
                else:
                    class_name = str(class_name)
                data.append({
                    'Class': class_name,
                    'TreeType': 'Axon',
                    'Full': False,
                    property_name: val
                })
            for val in dendrite_props:
                if type(class_name) == str:
                    class_name = class_name.replace("_", " ")
                else:
                    class_name = str(class_name)
                data.append({
                    'Class': class_name,
                    'TreeType': 'Dendrite',
                    property_name: val
                })
            for val in full_props:
                if type(class_name) == str:
                    class_name = class_name.replace("_", " ")
                else:
                    class_name = str(class_name)
                full_data.append({
                    'Class': class_name,
                    property_name: val
                })

        if not data:
            print("No valid data found for violin plot.")
            return

        df = pd.DataFrame(data)
        df_full = pd.DataFrame(full_data)

        df_hist = df_full.copy()

        if log_scale_y:
            df_hist = df_hist[df_hist[property_name] > 0].copy()
            df = df[df[property_name] > 0].copy()
            df_full = df_full[df_full[property_name] > 0].copy()

        original_property_name = property_name
        if log_scale_y:
            df[property_name] = np.log10(df[property_name])
            df_full[property_name] = np.log10(df_full[property_name])
            y_label = original_property_name
        else:
            y_label = property_name

        classes_hist = df_hist["Class"].unique()
        classes = df_full["Class"].unique()

        # Special-case ordering: when the full afferent/efferent class set is present,
        # force this fixed order and label the first three "Afferent", the last three "Efferent".
        normalized_to_class = {str(c).strip().lower(): c for c in classes_hist}
        use_afferent_efferent_grouping = set(normalized_to_class.keys()) == set(AFFERENT_EFFERENT_ORDER)

        if use_afferent_efferent_grouping:
            classes_hist = np.array([normalized_to_class[name] for name in AFFERENT_EFFERENT_ORDER])
            classes = classes_hist

        print("Classes:", classes)
        num_classes = len(classes_hist)

        def get_stars(delta):
            if delta is None:
                return None, None
            abs_d = abs(delta)
            if abs_d > 0.42:
                return '***', '#cc0000'
            elif abs_d > 0.23:
                return '**', '#e69138'
            elif abs_d > 0.11:
                return '*', "#1bb654"
            else:
                return None, None

        def get_badge_color(delta):
            if delta is None:
                return 'white'
            abs_d = abs(delta)
            if abs_d < 0.11:
                return 'white'
            elif abs_d < 0.23:
                return '#1bb654'
            elif abs_d < 0.42:
                return '#e69138'
            else:
                return '#cc0000'

        # --- Calculate Cliff's Delta ---
        delta_values = {}
        print("\n" + "="*60)
        print(f"Cliff's Delta for {property_name} (Axon vs Dendrite):")
        print("="*60)
        for cls in classes:
            cls_axon = df[(df["Class"] == cls) & (df["TreeType"] == "Axon")][property_name]
            cls_dend = df[(df["Class"] == cls) & (df["TreeType"] == "Dendrite")][property_name]
            if len(cls_axon) > 0 and len(cls_dend) > 0:
                delta = self.calculate_cliff_delta(cls_axon.values, cls_dend.values)
                delta_values[cls] = delta
                print(f"  {cls:30s}: δ = {delta:7.4f}")
            else:
                delta_values[cls] = None
                print(f"  {cls:30s}: N/A (insufficient data)")

        overall_delta = self.calculate_cliff_delta(
            df[df["TreeType"] == "Axon"][property_name].values,
            df[df["TreeType"] == "Dendrite"][property_name].values
        )
        print(f"\n  {'OVERALL':30s}: δ = {overall_delta:7.4f}")
        print("="*60 + "\n")

        # --- Layout: 2 rows ---
        fig = plt.figure(figsize=(16, 9.0) if use_afferent_efferent_grouping else (16, 8))
        gs_top = 0.80 if use_afferent_efferent_grouping else 0.95
        gs = gridspec.GridSpec(
            2, num_classes,
            height_ratios=[0.6, 1.2],
            hspace=0.15,
            top=gs_top, bottom=0.05, left=0.08, right=0.98
        )

        # --- Top histogram row ---
        top_axes = []
        if plot_multiple:
            for idx, cls in enumerate(classes_hist):
                ax_h = fig.add_subplot(gs[0, idx])
                top_axes.append(ax_h)
                subset = df_hist[df_hist["Class"] == cls][property_name].dropna()

                if len(subset) == 0:
                    continue

                if log_scale_y and len(subset) > 0 and subset.min() > 0:
                    log_min = np.log10(subset.min())
                    log_max = np.log10(subset.max())
                    if log_max > log_min:
                        hist_bins = np.logspace(log_min, log_max, 30)
                    else:
                        hist_bins = 30
                    ax_h.hist(subset, bins=hist_bins, alpha=0.8, edgecolor='black')
                    ax_h.set_xscale('log')
                else:
                    num_bins = self._compute_bins(subset, 'auto')
                    bin_edges = np.linspace(subset.min(), subset.max(), num_bins + 1)
                    ax_h.hist(subset, bins=bin_edges, alpha=0.8, edgecolor='black')

                wrapped_cls = cls.replace(' ', '\n') if len(cls) > 14 else cls
                if given_super_class is None:
                    ax_h.set_title(wrapped_cls, fontsize=25)
                else:
                    ax_h.set_title(wrapped_cls, fontsize=22)
                ax_h.tick_params(axis="x", rotation=45, labelsize=9)
                ax_h.tick_params(axis="y", labelsize=9)

                if idx == 0:
                    ax_h.set_ylabel('Count', fontsize=16)

            if use_afferent_efferent_grouping and len(top_axes) == 6:
                fig.canvas.draw()

                def _draw_group_bracket(x_left, x_right, label):
                    title_clearance = 0.045  # space above axes reserved for the class-name title
                    tick_drop = 0.02
                    label_gap = 0.02
                    bracket_y = gs_top + title_clearance + tick_drop
                    text_y = bracket_y + label_gap

                    fig.add_artist(mlines.Line2D(
                        [x_left, x_right], [bracket_y, bracket_y],
                        transform=fig.transFigure, color='black', linewidth=1.5
                    ))
                    fig.add_artist(mlines.Line2D(
                        [x_left, x_left], [bracket_y - tick_drop, bracket_y],
                        transform=fig.transFigure, color='black', linewidth=1.5
                    ))
                    fig.add_artist(mlines.Line2D(
                        [x_right, x_right], [bracket_y - tick_drop, bracket_y],
                        transform=fig.transFigure, color='black', linewidth=1.5
                    ))
                    fig.text((x_left + x_right) / 2, text_y, label,
                             ha='center', va='bottom', fontsize=22, fontweight='bold')

                afferent_axes = top_axes[:3]
                efferent_axes = top_axes[3:]
                afferent_left = min(ax.get_position().x0 for ax in afferent_axes)
                afferent_right = max(ax.get_position().x1 for ax in afferent_axes)
                efferent_left = min(ax.get_position().x0 for ax in efferent_axes)
                efferent_right = max(ax.get_position().x1 for ax in efferent_axes)

                _draw_group_bracket(afferent_left, afferent_right, 'Afferent')
                _draw_group_bracket(efferent_left, efferent_right, 'Efferent')

        # --- Bottom violin plot ---
        ax_violin = fig.add_subplot(gs[1, :])

        sns.violinplot(
            x='Class',
            y=property_name,
            hue='TreeType',
            data=df,
            order=list(classes_hist),
            split=True,
            inner='quartile',
            palette=["#fca311", "#695535"],
            cut=0,
            ax=ax_violin
        )

        # --- Compute y limits first ---
        # We need to extend the bottom to make room for badges
        y_data_min = df[property_name].min()
        y_data_max = df[property_name].max()
        y_data_range = y_data_max - y_data_min

        # Reserve 12% of the data range at the bottom for badges
        badge_reserved = y_data_range * 0.12
        y_axis_min = y_data_min - badge_reserved - 0.05
        y_axis_max = y_data_max + 0.1

        # Apply y limits (override if user provided them)
        if y_min is not None and y_max is not None:
            if log_scale_y:
                y_min_log = np.log10(y_min) if y_min is not None and y_min > 0 else y_axis_min
                y_max_log = np.log10(y_max) if y_max is not None and y_max > 0 else y_axis_max
                ax_violin.set_ylim(y_min_log, y_max_log)
                badge_y = y_min_log + (y_max_log - y_min_log) * 0.02
            else:
                ax_violin.set_ylim(y_min, y_max)
                badge_y = y_min + (y_max - y_min) * 0.02
        else:
            ax_violin.set_ylim(y_axis_min, y_axis_max)
            # Place badges in the reserved zone, centred vertically within it
            badge_y = y_axis_min + badge_reserved * 0.4

        # --- Delta badges in the reserved space below violin bodies ---
        violin_classes = classes_hist

        for idx, cls in enumerate(violin_classes):
            delta = delta_values.get(cls, None)
            delta_text = f"δ = {delta:.4f}" if delta is not None else "δ = N/A"
            bg_color = get_badge_color(delta)

            ax_violin.text(
                idx,
                badge_y,
                delta_text,
                ha='center', va='center',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.9,
                          edgecolor='black', linewidth=0.5)
            )

        # Remove x-axis tick labels
        ax_violin.set_xticklabels([])

        if original_property_name == "Fractal Dimension":
            line_y = 0 if log_scale_y else 1
            ax_violin.axhline(y=line_y, color='red', linestyle='--', linewidth=2, label='y=1')

        ax_violin.set_ylabel(y_label, fontsize=16)

        # --- Format Y-axis ticks ---
        if log_scale_y:
            def format_log_tick(x):
                if x == int(x):
                    return f'$10^{{{int(x)}}}$'
                else:
                    return f'$10^{{{x:.1f}}}$'

            # Only show ticks within the actual data range, not in badge zone
            tick_min = np.ceil(y_data_min - 0.1)
            tick_max = np.floor(y_data_max + 0.1)
            ticks = np.arange(tick_min, tick_max + 1)
            ax_violin.set_yticks(ticks)
            ax_violin.set_yticklabels([format_log_tick(t) for t in ticks], fontsize=12)
        else:
            plt.setp(ax_violin.get_yticklabels(), fontsize=12)

        # --- Output ---
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print("Fig saved to ", save_path)
        else:
            plt.show()

        plt.close()

    def _compute_bins(self, values, bins_spec, widen_factor: float = 1.0, clamp_max: int = 500) -> int:
        """
        Determine number of histogram bins.

        - values: iterable of numeric values
        - bins_spec: int or method name ('auto'|'fd'|'sturges'|'scott'|'rice'|'sqrt'|'doane')
        - widen_factor: >1 -> fewer (wider) bins; 1.0 keeps default width
        - clamp_max: upper cap on returned bins

        Returns an integer number of bins (safe for plt.hist or np.logspace).
        """
        arr = np.asarray(values)
        arr = arr[np.isfinite(arr)]
        n = len(arr)
        if n == 0:
            return 10

        # sanitize widen_factor
        try:
            widen_factor = float(widen_factor)
        except Exception:
            widen_factor = 1.0
        if widen_factor <= 0:
            widen_factor = 1.0

        # integer requested
        if isinstance(bins_spec, int):
            base = max(3, int(bins_spec))
            effective = max(3, int(np.ceil(base / widen_factor)))
            return min(effective, clamp_max)

        method = str(bins_spec).lower()

        # Freedman–Diaconis (and 'auto' as FD here)
        if method in ('auto', 'fd', 'freedman-diaconis'):
            if n < 2:
                return 10
            q75, q25 = np.percentile(arr, [75, 25])
            iqr = q75 - q25
            if iqr == 0:
                base = max(5, int(np.sqrt(n)))
                effective = max(3, int(np.ceil(base / widen_factor)))
                return min(effective, clamp_max)
            bw = 2.0 * iqr / (n ** (1.0/3.0))
            if bw <= 0 or arr.max() == arr.min():
                base = max(5, int(np.sqrt(n)))
                effective = max(3, int(np.ceil(base / widen_factor)))
                return min(effective, clamp_max)
            # make bins wider by multiplying bandwidth
            bw *= widen_factor
            nbins = int(np.ceil((arr.max() - arr.min()) / bw)) if bw > 0 else max(5, int(np.sqrt(n)))
            effective = max(3, nbins)
            return min(effective, clamp_max)

        # Other named methods: compute base nbins then reduce by widen_factor
        if method in ('sturges', 'scott', 'rice', 'sqrt', 'doane'):
            try:
                edges = np.histogram_bin_edges(arr, bins=method)
                base = max(3, len(edges) - 1)
            except Exception:
                base = max(5, int(np.sqrt(n)))
            effective = max(3, int(np.ceil(base / widen_factor)))
            return min(effective, clamp_max)

        # Unknown spec -> fallback to sqrt rule, then widen
        base = max(5, int(np.sqrt(n)))
        effective = max(3, int(np.ceil(base / widen_factor)))
        return min(effective, clamp_max)

    def _as_positive_float_list(self, values):
        """Filter None and cast to float."""
        return [float(v) for v in values if v is not None]

    def _overlay_distribution_fit_counts(
        self,
        data,
        bin_edges,
        dist_name="lognorm",
        line_kwargs=None,
        label=None,
        x_points=400,
        floc0=False,
        density=False,
        log_scale_x=False,
        mean=None,
        sigma=None
    ):
        """
        Overlay a fitted distribution on a COUNT histogram (density=False),
        correctly scaled for variable-width bins (e.g., log-spaced bins).

        expected curve at x: pdf(x) * N * (local bin width at x)
        """
        data = np.asarray(data, dtype=float)
        if data.size == 0:
            return None

        # must be positive for log-space and for lognorm/gamma with floc=0 assumptions
        # data = data[data > 0]
        if data.size == 0:
            return None

        bin_edges = np.asarray(bin_edges, dtype=float)
        if bin_edges.size < 2:
            return None

        # x range based on bin edges (matches what you plotted)
        x_min, x_max = bin_edges[0], bin_edges[-1]
        if not (x_min > 0 and x_max > 0 and x_max > x_min):
            return None
        # x grid in log space
        if log_scale_x:
            x = np.logspace(np.log10(x_min), np.log10(x_max), int(x_points))
        else:
            x = np.linspace(x_min, x_max, int(x_points))

        # choose distribution
        dist = getattr(stats, dist_name)

        # fit
        if floc0:
            loc0 = np.min(data) * 0.999  # tiny epsilon below min
            params = dist.fit(data, floc=loc0)
        else:
            params = dist.fit(data)

        if mean is not None and sigma is not None:
            if dist_name == "lognorm":
                s = sigma
                scale = np.exp(mean)
                params = (s, 0, scale)
            elif dist_name == "norm":
                params = (mean, sigma)
            else:
                print(f"Warning: mean/sigma override not implemented for {dist_name}")
        
        pdf = dist.pdf(x, *params)

        # Scale appropriately based on density parameter
        if density:
            # For density=True: just plot the PDF
            y = pdf
        else:
            # For density=False (counts): scale by N and bin width
            bin_widths = np.diff(bin_edges)
            bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
            width_at_x = np.interp(
                x,
                bin_centers,
                bin_widths,
                left=bin_widths[0],
                right=bin_widths[-1],
            )
            y = pdf * data.size * width_at_x

        line_kwargs = line_kwargs or {}
        if label is not None:
            line_kwargs = dict(line_kwargs)
            line_kwargs["label"] = label

        plt.plot(x, y, **line_kwargs)
        return params
    

    def plot_histogram(self, property, property_name, tree_type, bins=100, log_scale_x=False, log_scale_y=False, save_path=None, x_min=None, x_max=None, super_classes: dict = None, plot_multiple: bool = False, filter_by_fd: bool = False):
        # When plotting the five intrinsic classes, sensory neurons are dropped
        # completely from every histogram/statistic below, not just unlabeled.
        exclude_sensory_neurons = is_intrinsic_class_set(super_classes)
        super_classes = drop_sensory_if_intrinsic(super_classes)

        print("Axon trees:",len(self.axon_dendrite_forest.axon_tree))
        print("Dendrite trees:",len(self.axon_dendrite_forest.dendrite_tree))


        plt.figure(figsize=(12, 8))
        
        axon_props = []
        dendrite_props = []
        if tree_type == "both":
            axon_color = "#fca311"
            dendrite_color = "#695535"
            if property == "cable_length":
                density = True
                cnt = 0
                for a_tree, d_tree in zip(self.axon_dendrite_forest.axon_tree, self.axon_dendrite_forest.dendrite_tree):
                    if exclude_sensory_neurons and _is_sensory_tree(a_tree):
                        continue
                    a_prop = a_tree.get_property(property)
                    d_prop = d_tree.get_property(property)
                    if a_prop is not None and d_prop is not None:
                        if a_prop > 0 and d_prop > 0:
                            if filter_by_fd:
                                a_fd = a_tree.get_property("fractal_dimension")
                                d_fd = d_tree.get_property("fractal_dimension")
                                if a_fd is not None and d_fd is not None and a_fd > 1 and d_fd > 1:
                                    axon_props.append(a_prop)
                                    dendrite_props.append(d_prop)
                                else:
                                    cnt += 1
                            else:
                                axon_props.append(a_prop)
                                dendrite_props.append(d_prop)
                print(f"Filtered out {cnt} pairs due to FD <= 1 (non-fractal) in cable_length histogram.")
                axon_props = np.log(axon_props)
                dendrite_props = np.log(dendrite_props)
                # with open("dendrite_histogram_log_cable_length.txt", "w") as f:
                #     for val in dendrite_props:
                #         f.write(f"{val}\n")
                all_log_props = np.concatenate([axon_props, dendrite_props])
                current_bins = np.linspace(all_log_props.min(), all_log_props.max(), 50)
                n_axon, bin_edges, _ = plt.hist(axon_props, bins=current_bins, color=axon_color, alpha=0.6, label='Axon', density=density)
                n_dend, bin_edges, _ = plt.hist(dendrite_props, bins=current_bins, color=dendrite_color, alpha=0.5, label='Dendrite',   density=density)

                # Fit normal distribution
                params = self._overlay_distribution_fit_counts(
                    axon_props,
                    bin_edges,
                    dist_name="norm",
                    floc0=False,
                    x_points=400,
                    label="Axon normal fit",
                    line_kwargs=dict(color="darkblue", linewidth=2.5, linestyle="--"),
                    density=density,
                    log_scale_x=False  # Already in log space
                )
                print("Axon normal fit params:", params)
                params = self._overlay_distribution_fit_counts(
                    dendrite_props,
                    bin_edges,
                    dist_name="norm",
                    floc0=False,
                    x_points=400,
                    label="Dendrite normal fit",
                    line_kwargs=dict(color="darkgreen", linewidth=2.5, linestyle="--"),
                    density=density,
                    log_scale_x=False
                )
                print("Dendrite normal fit params:", params)

                # Update x-axis label
                plt.xlabel('ln(Cable Length)', fontsize=20)
            else:
                density = False
                for a_tree, d_tree in zip(self.axon_dendrite_forest.axon_tree, self.axon_dendrite_forest.dendrite_tree):
                    if exclude_sensory_neurons and _is_sensory_tree(a_tree):
                        continue
                    a_prop = a_tree.get_property(property)
                    d_prop = d_tree.get_property(property)
                    if a_prop is not None and d_prop is not None and a_prop > 0 and d_prop > 0:
                        axon_props.append(a_prop)
                        dendrite_props.append(d_prop)
                plt.xlabel(property_name, fontsize=20)
            
                all_values = axon_props + dendrite_props
                num_bins = self._compute_bins(all_values, bins)
                if log_scale_x and all_values:
                    if all_values:
                        log_min = np.log10(x_min) if x_min is not None and x_min > 0 else np.log10(min(all_values))
                        log_max = np.log10(x_max) if x_max is not None and x_max > 0 else np.log10(max(all_values))
                        if log_max <= log_min:
                            log_max = log_min + 1
                        current_bins = np.logspace(log_min, log_max, 30)
                    else:
                        current_bins = np.linspace(min(all_values), max(all_values), 30)
                        # current_bins = num_bins
                else:
                    current_bins = np.linspace(min(all_values), max(all_values), 30)
                    # current_bins = num_bins

                n_axon, bin_edges, _ = plt.hist(axon_props, bins=current_bins, color=axon_color, alpha=0.6, label='Axon', density=density)
                n_dend, bin_edges, _ = plt.hist(dendrite_props, bins=current_bins, color=dendrite_color, alpha=0.5, label='Dendrite', density=density)

            print(f"Cliff's Delta between Axon and Dendrite for {property_name}:")
            print(self.calculate_cliff_delta(axon_props, dendrite_props))
        else:
            density = True
            props = []
            for tree in self.axon_dendrite_forest.full_tree:
                if exclude_sensory_neurons and _is_sensory_tree(tree):
                    continue
                prop = tree.get_property(property)
                if prop is not None and prop > 0:
                    props.append(prop)

            props = np.array(props)

                
            # num_bins = self._compute_bins(props, bins)

            # NOTE this is how we did the binning originally, especially relevant for fractal dimension
            if log_scale_x and props is not None and len(props) > 0:
                if props is not None and len(props) > 0:
                    log_min = np.log10(x_min) if x_min is not None and x_min > 0 else np.log10(min(props))
                    log_max = np.log10(x_max) if x_max is not None and x_max > 0 else np.log10(max(props))
                    if log_max <= log_min:
                        log_max = log_min + 1
                    current_bins = np.logspace(log_min, log_max, 50)
                else:
                    current_bins = num_bins
            else:
            #     current_bins = num_bins


                current_bins = np.linspace(props.min(), props.max(), 50)
            
            n_axon, bin_edges, _ = plt.hist(props, bins=current_bins, alpha=0.6, density=density)
            plt.xlabel(property_name, fontsize=20)

            if property == "cable_length":
                props = np.array([p for p in props if p > 0])
                print("log_scale_x in histogram:", log_scale_x)
                print(props[:5], "props after filtering non-positive values for log transform")
                props = np.log(props)
                print(props[:5], "props after filtering non-positive values for log transform")
                current_bins = np.linspace(props.min(), props.max(), 50)
                print(current_bins[:5], "current_bins for log-transformed cable length")
                print(current_bins[-5:], "current_bins for log-transformed cable length")
                n_axon, bin_edges, _ = plt.hist(props, bins=current_bins, alpha=0.6, density=density, color='steelblue')
                plt.xlabel(property_name, fontsize=20)
                params = self._overlay_distribution_fit_counts(
                    props,
                    bin_edges,
                    dist_name="norm",
                    floc0=False,
                    x_points=400,
                    label="Normal fit",
                    line_kwargs=dict(color="darkblue", linewidth=2.5, linestyle="--"),
                    density=density,
                    log_scale_x=log_scale_x
                )
                plt.xlabel(f"ln({property_name})", fontsize=20)
                print("Normal fit params:", params)

            if x_min is not None or x_max is not None:
                plt.xlim(x_min, x_max)
            elif not log_scale_x:
                plt.xlim(self._axis_range(list(props)))
            

        # --- Scaling and Labels ---
        if log_scale_x:
            plt.xscale('log')

        if log_scale_y:
            try:
                plt.yscale('log', nonpositive='clip')
            except TypeError:
                plt.yscale('log')
            ymin, ymax = plt.ylim()
            if ymin <= 0:
                plt.ylim(bottom=0.5)


        if density:
            print("Density=True: y-axis represents probability density, not raw counts.")
            plt.ylabel('Probability Density', fontsize=20)
        else:
            print("Density=False: y-axis represents raw counts, not probability density.")
            plt.ylabel('Count', fontsize=20)
        plt.legend(fontsize=16)
        plt.legend(fontsize=14)

        plt.tick_params(axis='both', which='major', labelsize=16)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()


    def _get_property_color_map(self, super_classes, skip_first=False):
        super_class_names = list(super_classes.keys())
        colors = [
            '#cb2d28', "#ffd900", '#2c9f2a', '#1f77b4', '#91fd0d', '#bf13aa', '#0dbdfd', '#fde90d', '#0dfdf9',
            '#8c564b', '#e377c2', '#17becf', '#bcbd22', '#7f7f7f', '#ff1493', '#00a651', '#ff6f61', '#4b0082',
            "#03817b", '#b22222', '#ffd700', '#6a3d9a', "#075901", '#a6cee3', '#fb9a99', '#fdbf6f', '#cab2d6'
        ]
        if skip_first:
            colors = colors[1:]
        return {sc: colors[i % len(colors)] for i, sc in enumerate(super_class_names)}

    def _get_color_for_class(self, super_classes, skip_first=False):
        return self._get_property_color_map(super_classes, skip_first=skip_first)
    
    def plot_pairwise_different_properties(
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
        axon_or_dendrite='both',
        super_classes: dict = None,
        forest=None,
        given_super_class: str = None,
        interactive: bool = False
    ):
        super_classes = drop_sensory_if_intrinsic(super_classes)
        print("Remaining super_classes after dropping sensory neurons (if intrinsic):", super_classes.keys())

        # --- Data Retrieval ---
        if axon_or_dendrite == 'both':
            axon_or_dendrite = 'full'
        props_x = self.axon_dendrite_forest.get_property_list(property_x, tree_type=axon_or_dendrite)
        props_y = self.axon_dendrite_forest.get_property_list(property_y, tree_type=axon_or_dendrite)
        x_raw, y_raw, ids_raw, trees_raw = self.axon_dendrite_forest.get_property_pairs(
                                                                                            property_x, property_y, tree_type=axon_or_dendrite
                                                                                        )# Get trees if needed
        if forest is not None:
            if axon_or_dendrite == 'axon':
                trees = forest.axon_tree
            elif axon_or_dendrite == 'dendrite':
                trees = forest.dendrite_tree
            else:
                trees = forest.full_tree
        else:
            trees = None

        # Ensure all lists have same length
        min_len = min(len(props_x), len(props_y))
        if trees is not None:
            min_len = min(min_len, len(trees))
        
        x_vals = list(x_raw)
        y_vals = list(y_raw)
        ids_filtered = list(ids_raw)
        trees_filtered = list(trees_raw)  # same length, aligned

        if trees_raw is not None:
            # Always filter None values AND sync ids/trees with x/y
            filtered = []
            for x, y, id_val, t in zip(x_vals, y_vals, ids_filtered, trees_filtered):
                if x is None or y is None:
                    continue
                if log_scale_x and x <= 0:
                    continue
                if log_scale_y and y <= 0:
                    continue
                filtered.append((x, y, id_val, t))

            if filtered:
                x_vals, y_vals, ids_filtered, trees_filtered = zip(*filtered)
                x_vals, y_vals = list(x_vals), list(y_vals)
                ids_filtered = list(ids_filtered)
                trees_filtered = list(trees_filtered)
            else:
                x_vals, y_vals, ids_filtered, trees_filtered = [], [], [], []
        else:
            # No trees provided
            if log_scale_x or log_scale_y:
                filtered = [(x, y, id_val) for x, y, id_val in zip(x_raw, y_raw, ids_raw)
                        if x is not None and y is not None and x > 0 and y > 0]
            else:
                filtered = [(x, y, id_val) for x, y, id_val in zip(x_raw, y_raw, ids_raw)
                        if x is not None and y is not None]

            if filtered:
                x_vals, y_vals, ids_filtered = zip(*filtered)
                x_vals, y_vals = list(x_vals), list(y_vals)
                ids_filtered = list(ids_filtered)
            else:
                x_vals, y_vals, ids_filtered = [], [], []
            trees_filtered = None

        print(len(trees_filtered) if trees_filtered is not None else "No trees", "data points after filtering for correlation and plotting.")

        # Add validation
        if trees_filtered is not None:
            assert len(x_vals) == len(y_vals) == len(ids_filtered) == len(trees_filtered), \
                f"Length mismatch after filtering: x={len(x_vals)}, y={len(y_vals)}, ids={len(ids_filtered)}, trees={len(trees_filtered)}"

        # The 2-cluster split is only drawn when the classes being plotted are exactly
        # the six afferent/efferent classes.
        use_two_cluster_split = False
        if super_classes is not None:
            normalized_classes = {str(c).strip().lower() for c in super_classes.keys()}
            use_two_cluster_split = normalized_classes == set(AFFERENT_EFFERENT_ORDER)

        # Fit/correlation should use the same subset that is actually plotted.
        x_fit = list(x_vals)
        y_fit = list(y_vals)
        filter_nans = False
        if super_classes is not None and forest is not None and trees_filtered:
            super_class_to_color = self._get_color_for_class(super_classes)
            class_points = {cls: ([], []) for cls in super_classes.keys()}
            plot_keys = set(super_classes.keys())
            x_fit, y_fit = [], []
            for x, y, tree in zip(x_vals, y_vals, trees_filtered):
                sc = tree.class_ if given_super_class is not None else tree.super_class
                if sc in plot_keys:
                    x_fit.append(x)
                    y_fit.append(y)
            if not x_fit:
                # Safety fallback: avoid empty fit arrays if key mismatch occurs.
                x_fit = list(x_vals)
                y_fit = list(y_vals)
            
        if log_scale_x:
            x_corr = np.log10(np.array(x_fit, dtype=float))
        else:
            x_corr = np.array(x_fit, dtype=float)
        if log_scale_y:
            y_corr = np.log10(np.array(y_fit, dtype=float))
        else:            
            y_corr = np.array(y_fit, dtype=float)

        if len(x_corr) in (0, 1) or len(y_corr) in (0, 1):
            print("Not enough data points for correlation.")
            print("x_corr:", x_corr)
            print("y_corr:", y_corr)
            print("x_vals:", x_vals)
            print("y_vals:", y_vals)

        r, p_value = pearsonr(x_corr, y_corr) if len(x_corr) > 1 else (np.nan, np.nan)
        print("Pearson r:", r, "p-value:", p_value)

        fit_coeffs = None
        if len(x_corr) > 1:
            fit_coeffs = np.polyfit(x_corr, y_corr, 1)
            print("Fit line in correlation space: slope=", fit_coeffs[0], "intercept=", fit_coeffs[1])

        if interactive:
            return self._plot_pairwise_interactive(
                x_vals=x_vals, y_vals=y_vals, ids_filtered=ids_filtered, trees_filtered=trees_filtered,
                property_name_x=property_name_x, property_name_y=property_name_y,
                log_scale_x=log_scale_x, log_scale_y=log_scale_y,
                save_path=save_path,
                super_classes=super_classes, forest=forest, given_super_class=given_super_class,
                x_fit=x_fit, x_corr=x_corr, y_corr=y_corr, fit_coeffs=fit_coeffs,
                use_two_cluster_split=use_two_cluster_split,
            )

        # --- Plotting ---
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        center_distance_info = []

        x_label = f'{property_name_x}'
        y_label = f'{property_name_y}'

        centers = {}
        centers_corr = {}
        afferent_efferent_boundary = None

        if super_classes is not None and forest is not None and trees_filtered:

            # Now indices match correctly!
            for i, tree in enumerate(trees_filtered):
                if given_super_class is not None:
                    sc = tree.class_
                else:
                    sc = tree.super_class
                if sc in class_points:
                    class_points[sc][0].append(x_vals[i])
                    class_points[sc][1].append(y_vals[i])
            
            # Sort classes by number of data points (descending: most data first)
            sorted_classes = sorted(class_points.items(), key=lambda item: len(item[1][0]), reverse=True)
            
            # Plot each super class separately for legend (in sorted order)
            max_points_per_class = 5000000
    
            for class_name, (cx, cy) in sorted_classes:
                if filter_nans:
                    class_name = str(class_name) if class_name is not None else "Unknown"
                if not filter_nans or (class_name != "nan" and not "unknown" in class_name.lower()):
                    if cx:
                        if len(cx) > max_points_per_class:
                            idx = np.linspace(0, len(cx) - 1, max_points_per_class, dtype=int)
                            cx_plot = np.asarray(cx)[idx]
                            cy_plot = np.asarray(cy)[idx]
                            label = f"{class_name} ({len(cx)}; shown {max_points_per_class})"
                        else:
                            cx_plot = cx
                            cy_plot = cy
                            label = f"{class_name} ({len(cx)})"

                        ax.scatter(
                            cx_plot,
                            cy_plot,
                            alpha=0.7,
                            s=10,
                            color=super_class_to_color[class_name],
                            label=label,
                            edgecolors='none',
                            linewidths=0,
                            rasterized=True,
                        )

                        # Draw class center as a large, nearly transparent marker.
                        cx_arr = np.asarray(cx, dtype=float)
                        cy_arr = np.asarray(cy, dtype=float)

                        center_x_corr = float(np.mean(np.log10(cx_arr))) if log_scale_x else float(np.mean(cx_arr))
                        center_y_corr = float(np.mean(np.log10(cy_arr))) if log_scale_y else float(np.mean(cy_arr))

                        center_x_plot = 10 ** center_x_corr if log_scale_x else center_x_corr
                        center_y_plot = 10 ** center_y_corr if log_scale_y else center_y_corr

                        centers[class_name] = (center_x_plot, center_y_plot)
                        centers_corr[class_name] = (center_x_corr, center_y_corr)

                        print(f"Class '{class_name}' center (plot space): ({center_x_plot}, {center_y_plot})")
                        print(f"Class '{class_name}' center (log space): ({center_x_corr}, {center_y_corr})")

                        ax.scatter(
                            center_x_plot,
                            center_y_plot,
                            s=260,
                            alpha=0.7,
                            color=super_class_to_color[class_name],
                            edgecolors='black',
                            linewidths=1.1,
                            zorder=4,
                        )

                        if fit_coeffs is not None:
                            m, b = fit_coeffs
                            center_x_corr = float(np.mean(np.log10(cx_arr))) if log_scale_x else float(np.mean(cx_arr))
                            center_y_corr = float(np.mean(np.log10(cy_arr))) if log_scale_y else float(np.mean(cy_arr))

                            center_x = 10 ** center_x_corr if log_scale_x else center_x_corr
                            center_y = 10 ** center_y_corr if log_scale_y else center_y_corr
                            center_dist = abs(m * center_x_corr - center_y_corr + b) / np.sqrt(m * m + 1.0)
                            center_distance_info.append((class_name, float(center_dist)))

                # Avoid the expensive auto-placement algorithm used by loc='best'
            ax.legend(fontsize=11, loc='best', markerscale=2)

            # Afferent (sensory, optic, ascending) vs efferent (descending, endocrine,
            # motor) boundary: perpendicular bisector between the mean of each group's
            # three class centers, only when exactly these six classes are being plotted.
            if use_two_cluster_split:
                normalized_to_class_name = {str(c).strip().lower(): c for c in centers_corr.keys()}
                afferent_names, efferent_names = AFFERENT_EFFERENT_ORDER[:3], AFFERENT_EFFERENT_ORDER[3:]
                afferent_centers = [centers_corr[normalized_to_class_name[n]] for n in afferent_names if n in normalized_to_class_name]
                efferent_centers = [centers_corr[normalized_to_class_name[n]] for n in efferent_names if n in normalized_to_class_name]
                if len(afferent_centers) == 3 and len(efferent_centers) == 3:
                    afferent_center = np.mean(np.asarray(afferent_centers), axis=0)
                    efferent_center = np.mean(np.asarray(efferent_centers), axis=0)
                    afferent_efferent_boundary = (afferent_center, efferent_center)
                    print("Afferent center (correlation space):", afferent_center)
                    print("Efferent center (correlation space):", efferent_center)
        else:
            ax.scatter(
                x_vals,
                y_vals,
                alpha=0.7,
                s=8,
                edgecolors='none',
                linewidths=0,
                rasterized=True,
            )

        # Draw regression line using the same transformed space used for Pearson r.
        if fit_coeffs is not None and len(x_vals) > 1:
            x_arr = np.asarray(x_fit, dtype=float)
            x_min_plot = np.min(x_arr)
            x_max_plot = np.max(x_arr)

            if x_max_plot > x_min_plot:
                if log_scale_x:
                    x_line = np.logspace(np.log10(x_min_plot), np.log10(x_max_plot), 200)
                else:
                    x_line = np.linspace(x_min_plot, x_max_plot, 200)

                m, b = fit_coeffs
                if log_scale_x and log_scale_y:
                    y_line = 10 ** (m * np.log10(x_line) + b)
                elif log_scale_x and not log_scale_y:
                    y_line = m * np.log10(x_line) + b
                elif not log_scale_x and log_scale_y:
                    y_line = 10 ** (m * x_line + b)
                else:
                    y_line = m * x_line + b

                valid = np.isfinite(x_line) & np.isfinite(y_line)
                if log_scale_y:
                    valid = valid & (y_line > 0)

                if np.any(valid):
                    ax.plot(
                        x_line[valid],
                        y_line[valid],
                        color='black',
                        linestyle='--',
                        linewidth=2.2,
                        alpha=0.9,
                        zorder=3,
                    )

        if center_distance_info:
            print("Center distances to regression line:")
            for class_name, dist in center_distance_info:
                print(f"  {class_name}: {dist:.6f}")

        # Scales must be set before we read/restore axis limits below, otherwise
        # the pre-log linear limits get reapplied on a log axis and break it.
        if log_scale_x:
            plt.xscale('log')
        if log_scale_y:
            plt.yscale('log')

        # Draw the afferent/efferent separating boundary (perpendicular bisector
        # between the two group centers, transformed back out of correlation/log
        # space). Axis limits are captured and restored since it's drawn far past
        # the visible data.
        if afferent_efferent_boundary is not None:
            line = self._group_boundary_line(afferent_efferent_boundary[0], afferent_efferent_boundary[1], x_corr, y_corr, log_scale_x, log_scale_y)
            if line is not None:
                boundary_x, boundary_y = line
                orig_xlim, orig_ylim = ax.get_xlim(), ax.get_ylim()
                ax.plot(
                    boundary_x, boundary_y,
                    linestyle=':', color='black', linewidth=2.2, alpha=0.9,
                    zorder=5, label='Afferent/Efferent boundary',
                )
                ax.set_xlim(orig_xlim)
                ax.set_ylim(orig_ylim)
                ax.legend(fontsize=11, loc='best', markerscale=2)

        plt.xlabel(x_label, fontsize=20)
        plt.ylabel(y_label, fontsize=20)
        plt.gca().tick_params(axis='both', which='major', labelsize=16)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def _plot_pairwise_interactive(
        self,
        x_vals, y_vals, ids_filtered, trees_filtered,
        property_name_x, property_name_y,
        log_scale_x, log_scale_y,
        save_path,
        super_classes, forest, given_super_class,
        x_fit, x_corr, y_corr, fit_coeffs,
        use_two_cluster_split,
    ):
        """
        Plotly/HTML counterpart of plot_pairwise_different_properties: same grouping,
        centroids, regression line and afferent/efferent boundary, but as an
        interactive scatter whose hover text shows the neuron id and the x/y values.
        """
        import plotly.graph_objects as go

        fig = go.Figure()

        def hover_text(id_val, x_val, y_val):
            return f"ID: {id_val}<br>{property_name_x}: {x_val:.4g}<br>{property_name_y}: {y_val:.4g}"

        centers_corr = {}

        if super_classes is not None and forest is not None and trees_filtered:
            super_class_to_color = self._get_color_for_class(super_classes)
            class_points = {cls: ([], [], []) for cls in super_classes.keys()}  # x, y, id

            for x, y, id_val, tree in zip(x_vals, y_vals, ids_filtered, trees_filtered):
                sc = tree.class_ if given_super_class is not None else tree.super_class
                if sc in class_points:
                    class_points[sc][0].append(x)
                    class_points[sc][1].append(y)
                    class_points[sc][2].append(id_val)

            # Most data first, matching the static plot's legend order.
            sorted_classes = sorted(class_points.items(), key=lambda item: len(item[1][0]), reverse=True)

            for class_name, (cx, cy, cids) in sorted_classes:
                if not cx:
                    continue

                fig.add_trace(go.Scatter(
                    x=cx, y=cy,
                    mode='markers',
                    marker=dict(size=6, color=super_class_to_color[class_name], opacity=0.7),
                    name=f"{class_name} ({len(cx)})",
                    text=[hover_text(i, x, y) for i, x, y in zip(cids, cx, cy)],
                    hoverinfo='text',
                ))

                # Class centroid, mirroring the large translucent marker in the static plot.
                cx_arr = np.asarray(cx, dtype=float)
                cy_arr = np.asarray(cy, dtype=float)
                center_x_corr = float(np.mean(np.log10(cx_arr))) if log_scale_x else float(np.mean(cx_arr))
                center_y_corr = float(np.mean(np.log10(cy_arr))) if log_scale_y else float(np.mean(cy_arr))
                center_x_plot = 10 ** center_x_corr if log_scale_x else center_x_corr
                center_y_plot = 10 ** center_y_corr if log_scale_y else center_y_corr
                centers_corr[class_name] = (center_x_corr, center_y_corr)

                fig.add_trace(go.Scatter(
                    x=[center_x_plot], y=[center_y_plot],
                    mode='markers',
                    marker=dict(size=22, color=super_class_to_color[class_name], opacity=0.7,
                                line=dict(color='black', width=1.5)),
                    name=f"{class_name} centroid",
                    text=[f"{class_name} centroid<br>{property_name_x}: {center_x_plot:.4g}<br>{property_name_y}: {center_y_plot:.4g}"],
                    hoverinfo='text',
                    showlegend=False,
                ))

            # Afferent/efferent boundary, only for the six afferent/efferent classes
            # (mirrors the static plot's black dotted line).
            if use_two_cluster_split:
                normalized_to_class_name = {str(c).strip().lower(): c for c in centers_corr.keys()}
                afferent_names, efferent_names = AFFERENT_EFFERENT_ORDER[:3], AFFERENT_EFFERENT_ORDER[3:]
                afferent_centers = [centers_corr[normalized_to_class_name[n]] for n in afferent_names if n in normalized_to_class_name]
                efferent_centers = [centers_corr[normalized_to_class_name[n]] for n in efferent_names if n in normalized_to_class_name]
                if len(afferent_centers) == 3 and len(efferent_centers) == 3:
                    afferent_center = np.mean(np.asarray(afferent_centers), axis=0)
                    efferent_center = np.mean(np.asarray(efferent_centers), axis=0)
                    line = self._group_boundary_line(afferent_center, efferent_center, x_corr, y_corr, log_scale_x, log_scale_y)
                    if line is not None:
                        boundary_x, boundary_y = line
                        fig.add_trace(go.Scatter(
                            x=boundary_x, y=boundary_y,
                            mode='lines',
                            line=dict(color='black', dash='dot', width=2.2),
                            name='Afferent/Efferent boundary',
                            hoverinfo='skip',
                        ))
        else:
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode='markers',
                marker=dict(size=6, color='steelblue', opacity=0.7),
                name='Data points',
                text=[hover_text(i, x, y) for i, x, y in zip(ids_filtered, x_vals, y_vals)],
                hoverinfo='text',
            ))

        # Regression line, in the same transformed space used for Pearson r.
        if fit_coeffs is not None and len(x_vals) > 1:
            x_arr = np.asarray(x_fit, dtype=float)
            x_min_plot, x_max_plot = np.min(x_arr), np.max(x_arr)

            if x_max_plot > x_min_plot:
                if log_scale_x:
                    x_line = np.logspace(np.log10(x_min_plot), np.log10(x_max_plot), 200)
                else:
                    x_line = np.linspace(x_min_plot, x_max_plot, 200)

                m, b = fit_coeffs
                if log_scale_x and log_scale_y:
                    y_line = 10 ** (m * np.log10(x_line) + b)
                elif log_scale_x and not log_scale_y:
                    y_line = m * np.log10(x_line) + b
                elif not log_scale_x and log_scale_y:
                    y_line = 10 ** (m * x_line + b)
                else:
                    y_line = m * x_line + b

                valid = np.isfinite(x_line) & np.isfinite(y_line)
                if log_scale_y:
                    valid = valid & (y_line > 0)

                if np.any(valid):
                    fig.add_trace(go.Scatter(
                        x=x_line[valid], y=y_line[valid],
                        mode='lines',
                        line=dict(color='black', dash='dash', width=2.2),
                        name='Regression line',
                        hoverinfo='skip',
                    ))

        fig.update_layout(
            xaxis=dict(title=property_name_x, type='log' if log_scale_x else 'linear'),
            yaxis=dict(title=property_name_y, type='log' if log_scale_y else 'linear'),
            width=1000,
            height=800,
            showlegend=True,
        )

        if save_path:
            if not save_path.endswith('.html'):
                save_path = save_path.rsplit('.', 1)[0] + '.html'
            fig.write_html(save_path)
            print(f"Interactive plot saved to: {save_path}")
        else:
            fig.show()

    def plot_pairwise_3d_properties(
        self,
        property_x,
        property_y,
        property_z,
        property_name_x,
        property_name_y,
        property_name_z,
        log_scale_x=False,
        log_scale_y=False,
        log_scale_z=False,
        save_path=None,
        axon_or_dendrite='both',
        super_classes: dict = None,
        forest=None,
        given_super_class: str = None
    ):
        """
        Creates an interactive 3D scatter plot using Plotly.
        
        Parameters
        ----------
        property_x, property_y, property_z : str
            Property names for x, y, z axes
        property_name_x, property_name_y, property_name_z : str
            Display names for axes labels
        log_scale_x, log_scale_y, log_scale_z : bool
            Whether to use log scale for each axis
        save_path : str
            Path to save HTML file (must end in .html)
        axon_or_dendrite : str
            'axon', 'dendrite', or 'both' (converted to 'full')
        super_classes : dict
            Dictionary of super classes for coloring
        forest : AxonDendriteForest
            Forest object containing tree data
        given_super_class : str
            Filter to specific super class
        """
        import plotly.graph_objects as go

        super_classes = drop_sensory_if_intrinsic(super_classes)

        # --- Data Retrieval ---
        if axon_or_dendrite == 'both':
            axon_or_dendrite = 'full'

        props_x = self.axon_dendrite_forest.get_property_list(property_x, tree_type=axon_or_dendrite)
        props_y = self.axon_dendrite_forest.get_property_list(property_y, tree_type=axon_or_dendrite)
        props_z = self.axon_dendrite_forest.get_property_list(property_z, tree_type=axon_or_dendrite)
        props_x, props_y, props_z, _, trees_raw = self.axon_dendrite_forest.get_property_pairs3d(property_x, property_y, property_z, tree_type=axon_or_dendrite)
        
        # Get trees if needed
        if forest is not None:
            if axon_or_dendrite == 'axon':
                trees = forest.axon_tree
            elif axon_or_dendrite == 'dendrite':
                trees = forest.dendrite_tree
            else:
                trees = forest.full_tree
        else:
            trees = None

        # Ensure all lists have same length
        min_len = min(len(props_x), len(props_y), len(props_z))
        if trees is not None:
            min_len = min(min_len, len(trees))
        
        x_raw = list(props_x)
        y_raw = list(props_y)
        z_raw = list(props_z)

        # --- Filtering: remove None values and apply log filtering ---
        if log_scale_x or log_scale_y or log_scale_z:
            if trees_raw is not None:
                filtered = [(x, y, z, t) for x, y, z, t in zip(x_raw, y_raw, z_raw, trees_raw) 
                        if x is not None and y is not None and z is not None 
                        and x > 0 and y > 0 and z > 0]
                if filtered:
                    x_vals, y_vals, z_vals, trees_filtered = zip(*filtered)
                    x_vals, y_vals, z_vals = list(x_vals), list(y_vals), list(z_vals)
                    trees_filtered = list(trees_filtered)
                else:
                    x_vals, y_vals, z_vals, trees_filtered = [], [], [], []
            else:
                filtered = [(x, y, z) for x, y, z in zip(x_raw, y_raw, z_raw) 
                        if x is not None and y is not None and z is not None
                        and x > 0 and y > 0 and z > 0]
                if filtered:
                    x_vals, y_vals, z_vals = zip(*filtered)
                    x_vals, y_vals, z_vals = list(x_vals), list(y_vals), list(z_vals)
                else:
                    x_vals, y_vals, z_vals = [], [], []
                trees_filtered = None
        else:
            # Still filter out None values
            if trees_raw is not None:
                filtered = [(x, y, z, t) for x, y, z, t in zip(x_raw, y_raw, z_raw, trees_raw) 
                        if x is not None and y is not None and z is not None]
                if filtered:
                    x_vals, y_vals, z_vals, trees_filtered = zip(*filtered)
                    x_vals, y_vals, z_vals = list(x_vals), list(y_vals), list(z_vals)
                    trees_filtered = list(trees_filtered)
                else:
                    x_vals, y_vals, z_vals, trees_filtered = [], [], [], []
            else:
                filtered = [(x, y, z) for x, y, z in zip(x_raw, y_raw, z_raw) 
                        if x is not None and y is not None and z is not None]
                if filtered:
                    x_vals, y_vals, z_vals = zip(*filtered)
                    x_vals, y_vals, z_vals = list(x_vals), list(y_vals), list(z_vals)
                else:
                    x_vals, y_vals, z_vals = [], [], []
                trees_filtered = None

        if not x_vals:
            print("No valid data points for 3D plot.")
            return

        # --- Create Plotly Figure ---
        fig = go.Figure()

        # --- Add traces with coloring by super class ---
        if super_classes is not None and forest is not None and trees_filtered:
            # super_class_to_color = self._get_color_for_class(forest, axon_or_dendrite, super_classes, given_super_class=given_super_class)
            
            # Group points by super class
            class_points = {cls: ([], [], []) for cls in super_classes.keys()}

            # Decide whether keys represent class names or super-class names.
            # In many call sites, `super_classes` keys are actually class labels.
            key_mode = None
            for tree in trees_filtered:
                if tree.class_ in class_points:
                    key_mode = 'class'
                    break
                if tree.super_class in class_points:
                    key_mode = 'super_class'
                    break
            if key_mode is None:
                key_mode = 'class' if given_super_class is not None else 'super_class'
            
            for i, tree in enumerate(trees_filtered):
                sc = tree.class_ if key_mode == 'class' else tree.super_class
                if sc in class_points:
                    class_points[sc][0].append(x_vals[i])
                    class_points[sc][1].append(y_vals[i])
                    class_points[sc][2].append(z_vals[i])
            
            # Plot each super class separately for legend
            added_trace = False
            for class_name, (cx, cy, cz) in class_points.items():
                if cx:
                    fig.add_trace(go.Scatter3d(
                        x=cx,
                        y=cy,
                        z=cz,
                        mode='markers',
                        marker=dict(
                            size=3,
                            color=super_class_to_color[class_name],
                            opacity=0.7
                        ),
                        name=f"{class_name} ({len(cx)})"
                    ))
                    added_trace = True

            # Fallback to plotting all points if grouping produced no traces.
            if not added_trace:
                fig.add_trace(go.Scatter3d(
                    x=x_vals,
                    y=y_vals,
                    z=z_vals,
                    mode='markers',
                    marker=dict(
                        size=3,
                        color='steelblue',
                        opacity=0.7
                    ),
                    name='Data points'
                ))
        else:
            # Single trace without class coloring
            fig.add_trace(go.Scatter3d(
                x=x_vals,
                y=y_vals,
                z=z_vals,
                mode='markers',
                marker=dict(
                    size=3,
                    color='steelblue',
                    opacity=0.7
                ),
                name='Data points'
            ))

        # --- Configure axes labels and log scales ---
        x_label = property_name_x
        y_label = property_name_y
        z_label = property_name_z
        
        # Build title
        if given_super_class is not None:
            title = f"3D Scatter: {property_name_x} vs {property_name_y} vs {property_name_z}<br>for {given_super_class} ({axon_or_dendrite})"
        else:
            title = f"3D Scatter: {property_name_x} vs {property_name_y} vs {property_name_z} ({axon_or_dendrite})"

        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(
                    title=x_label,
                    type='log' if log_scale_x else 'linear'
                ),
                yaxis=dict(
                    title=y_label,
                    type='log' if log_scale_y else 'linear'
                ),
                zaxis=dict(
                    title=z_label,
                    type='log' if log_scale_z else 'linear'
                )
            ),
            width=1000,
            height=800,
            showlegend=True
        )

        # --- Save or Show ---
        if save_path:
            if not save_path.endswith('.html'):
                save_path = save_path.replace('.png', '.html')
            fig.write_html(save_path)
            print(f"3D plot saved to: {save_path}")
        else:
            fig.show()

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
        props_x, props_y, _, _ = self.axon_dendrite_forest.get_property_pairs(property_x, property_y, tree_type=axon_or_dendrite)
        
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
