import os
# Disable internal parallelism to prevent subprocess spawning
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import json
import pandas as pd
import numpy as np


def _get_filtered_skeleton_ids():
    file_w_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"../../filtered_skeleton_ids(025)_new_intrinsic.txt"))
    with open(file_w_path, 'r') as f:
        ids = [line.strip() for line in f if line.strip()]
    return ids


class AxonDendriteTree:
    def __init__(self, id, tree_properties=None, c_values=None, tree_type=None, super_class=None, class_=None):
        self.id = id
        self.tree_properties = tree_properties
        self.c_values = c_values
        self.tree_type = tree_type
        self.super_class = super_class
        self.class_ = class_


    @property
    def fractal_dimension(self):
        try:
            fractal_dimension = self.tree_properties['fractal_dimension'] if self.tree_properties else None
        except:
            fractal_dimension = -1
        return fractal_dimension
    

    @property
    def num_nodes(self):
        return self.tree_properties['num_nodes'] if self.tree_properties else None
        

    @property
    def num_edges(self):
        return self.tree_properties['num_edges'] if self.tree_properties else None
        

    @property
    def num_leaf_nodes(self):
        return self.tree_properties['num_leaf_nodes'] if self.tree_properties else None


    @property
    def height(self):
        return self.tree_properties['height'] if self.tree_properties else None
        

    @property
    def max_width(self):
        return self.tree_properties['max_width'] if self.tree_properties else None
        

    @property
    def cable_length(self):
        return self.tree_properties['cable_length'] if self.tree_properties else None


    @property
    def num_filtered_pre_synapses(self):
        return self.tree_properties['num_filtered_pre_synapses'] if self.tree_properties else None
        

    @property
    def num_filtered_post_synapses(self):
        return self.tree_properties['num_filtered_post_synapses'] if self.tree_properties else None


    @property
    def c_value(self):
        return self.c_values['eigenvalue_geometric_multiplicity'] if self.c_values else None
    

    @classmethod
    def from_properties_csv(cls, tree_filepath, c_filepath, id=None, tree_type=None, super_class=None, class_=None):
        if id is None:
            id = tree_filepath.split('/')[-1].split('.')[0].split('_')[0]
        with open(tree_filepath, 'r') as f:
            tree_json = f.read()
        with open(c_filepath, 'r') as f:
            c_json = f.read()
        instance = cls(id, json.loads(tree_json), json.loads(c_json), tree_type=tree_type, super_class=super_class, class_=class_)
        return instance
    

    def get_all_properties(self):
        properties = [
            name for name, value in AxonDendriteTree.__dict__.items()
            if isinstance(value, property)
        ]
        return properties

    def get_property_values(self):
        return [getattr(self, prop) for prop in self.get_all_properties()]


    def get_property(self, property_name):
        return getattr(self, property_name, None)


class AxonDendriteForest():
    def __init__(self, tree_type):
        self.axon_tree = []
        self.dendrite_tree = []
        self.full_tree = []
        self.tree_type = tree_type


    def add_axon_tree(self, axon_tree):
        self.axon_tree.append(axon_tree)


    def add_dendrite_tree(self, dendrite_tree):
        self.dendrite_tree.append(dendrite_tree)


    def add_full_tree(self, full_tree):
        self.full_tree.append(full_tree)

    
    def get_property_list(self, property_name, tree_type='axon'):
        property_list = []
        if tree_type == 'axon':
            for tree in self.axon_tree:
                x = getattr(tree, property_name) 
                if x is not None and not np.isnan(x):
                    property_list.append(getattr(tree, property_name))
        elif tree_type == 'dendrite':
            for tree in self.dendrite_tree:
                x = getattr(tree, property_name) 
                if x is not None and not np.isnan(x):
                    property_list.append(getattr(tree, property_name))
        elif tree_type == 'full':
            for tree in self.full_tree:
                x = getattr(tree, property_name) 
                if x is not None and not np.isnan(x):
                    property_list.append(getattr(tree, property_name))
        return property_list
    
    def get_property_pairs(self, prop_x, prop_y, tree_type="full"):
        if tree_type == "axon":
            trees = self.axon_tree
        elif tree_type == "dendrite":
            trees = self.dendrite_tree
        elif tree_type == "full":
            trees = self.full_tree
        elif tree_type == "both":
            trees = self.full_tree
        else:
            raise ValueError(tree_type)

        xs, ys, ids, kept_trees = [], [], [], []
        if not tree_type == "both":
            for t in trees:
                x = getattr(t, prop_x, None)
                y = getattr(t, prop_y, None)

                # drop None
                if x is None or y is None:
                    continue

                # drop NaN safely (works for numpy floats; avoids np.isnan on non-floats)
                if isinstance(x, (float, np.floating)) and np.isnan(x):
                    continue
                if isinstance(y, (float, np.floating)) and np.isnan(y):
                    continue

                xs.append(x)
                ys.append(y)
                ids.append(t.id)
                kept_trees.append(t)
        else:
            for t_full, t_axon, t_dendrite in zip(self.full_tree, self.axon_tree, self.dendrite_tree):
                x = getattr(t_axon, prop_x, None)
                y = getattr(t_dendrite, prop_y, None)

                # drop None
                if x is None or y is None:
                    continue

                # drop NaN safely (works for numpy floats; avoids np.isnan on non-floats)
                if isinstance(x, (float, np.floating)) and np.isnan(x):
                    continue
                if isinstance(y, (float, np.floating)) and np.isnan(y):
                    continue

                xs.append(x)
                ys.append(y)
                ids.append(t_full.id)
                kept_trees.append((t_full, t_axon, t_dendrite))

        return xs, ys, ids, kept_trees
    
    def get_property_pairs3d(self, prop_x, prop_y, prop_z, tree_type="full"):
        if tree_type == "axon":
            trees = self.axon_tree
        elif tree_type == "dendrite":
            trees = self.dendrite_tree
        elif tree_type == "full":
            trees = self.full_tree
        else:
            raise ValueError(tree_type)

        xs, ys, zs, ids, kept_trees = [], [], [], [], []
        for t in trees:
            x = getattr(t, prop_x, None)
            y = getattr(t, prop_y, None)
            z = getattr(t, prop_z, None)

            # drop None
            if x is None or y is None:
                continue

            # drop NaN safely (works for numpy floats; avoids np.isnan on non-floats)
            if isinstance(x, (float, np.floating)) and np.isnan(x):
                continue
            if isinstance(y, (float, np.floating)) and np.isnan(y):
                continue
            if isinstance(z, (float, np.floating)) and np.isnan(z):
                continue

            xs.append(x)
            ys.append(y)
            zs.append(z)
            ids.append(t.id)
            kept_trees.append(t)

        return xs, ys, zs, ids, kept_trees

    def get_id_list(self, tree_type='full'):
        """
        Returns list of skeleton IDs for the specified tree type.
        
        Parameters
        ----------
        tree_type : str
            'axon', 'dendrite', or 'full'
        
        Returns
        -------
        list
            List of skeleton IDs
        """
        if tree_type == 'axon':
            return [tree.id for tree in self.axon_tree]
        elif tree_type == 'dendrite':
            return [tree.id for tree in self.dendrite_tree]
        elif tree_type == 'full':
            return [tree.id for tree in self.full_tree]
        else:
            raise ValueError(f"Unknown tree_type: {tree_type}. Use 'axon', 'dendrite', or 'full'.")
    

    @classmethod
    def build_forest_from_directory(cls, tree_dir, c_dir, undirected=True, class_fname_w_path=None):
        df = None
        if class_fname_w_path is not None:
            df = pd.read_csv(class_fname_w_path)


        tree_type = 'undirected' if undirected else 'directed'
        forest = cls(tree_type)
        ids = _get_filtered_skeleton_ids()
        if df is not None:
            filtered_df = df[df['root_id'].isin([int(id) for id in ids])]

            filtered_ids = filtered_df['root_id'].tolist()
            super_classes = filtered_df['super_class'].tolist()
            classes = filtered_df['class'].tolist()
        else:
            super_classes = [None for _ in range(len(ids))]
            classes = [None for _ in range(len(ids))]
            filtered_ids = ids
            

        for id, superclass, class_ in zip(filtered_ids, super_classes, classes):
            tree_filepath_axon = f'{tree_dir}/{id}_axon.json'
            c_filepath_axon = f'{c_dir}/{id}_axon.json'
            tree_filepath_dendrite = f'{tree_dir}/{id}_dendrite.json'
            c_filepath_dendrite = f'{c_dir}/{id}_dendrite.json'
            tree_filepath_full = f'{tree_dir}/{id}_full.json'
            c_filepath_full = f'{c_dir}/{id}_full.json'
            if os.path.exists(tree_filepath_axon) and os.path.exists(c_filepath_axon):
                if os.path.exists(tree_filepath_dendrite) and os.path.exists(c_filepath_dendrite):
                    if os.path.exists(tree_filepath_full) and os.path.exists(c_filepath_full):
                        tree = AxonDendriteTree.from_properties_csv(tree_filepath_axon, c_filepath_axon, tree_type=tree_type, id=int(id), super_class=superclass, class_=class_)
                        forest.add_axon_tree(tree)
                        tree = AxonDendriteTree.from_properties_csv(tree_filepath_dendrite, c_filepath_dendrite, tree_type=tree_type, id=int(id), super_class=superclass, class_=class_)
                        forest.add_dendrite_tree(tree)
                        tree = AxonDendriteTree.from_properties_csv(tree_filepath_full, c_filepath_full, tree_type=tree_type, id=int(id), super_class=superclass, class_=class_)
                        forest.add_full_tree(tree)
        return forest

    
if __name__ == "__main__":
    skeleton_id = 720575940603563893
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Axon tree
    tree_filepath = os.path.abspath(os.path.join(script_dir, f"../../data/undirected/tree_properties/{skeleton_id}_axon.json"))
    c_filepath = os.path.abspath(os.path.join(script_dir, f"../../data/undirected/c_values/{skeleton_id}_axon.json"))
    axon_tree = AxonDendriteTree.from_properties_csv(tree_filepath, c_filepath, id=skeleton_id)

    # Dendrite tree
    tree_filepath = os.path.abspath(os.path.join(script_dir, f"../../data/undirected/tree_properties/{skeleton_id}_dendrite.json"))
    c_filepath = os.path.abspath(os.path.join(script_dir, f"../../data/undirected/c_values/{skeleton_id}_dendrite.json"))
    dendrite_tree = AxonDendriteTree.from_properties_csv(tree_filepath, c_filepath, id=skeleton_id)
