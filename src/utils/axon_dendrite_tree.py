import os
import json

def _get_skeleton_ids(filedir):
    ids = []
    for filename in os.listdir(filedir):
        ids.append(filename.split('/')[-1].split('_')[0])

    return list(set(ids))


def _get_filtered_skeleton_ids():
    file_w_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"../../filtered_skeleton_ids(03).txt"))
    with open(file_w_path, 'r') as f:
        ids = [line.strip() for line in f if line.strip()]
    return ids


class AxonDendriteTree:
    def __init__(self, id, tree_properties=None, c_values=None, tree_type=None):
        self.id = id
        self.tree_properties = tree_properties
        self.c_values = c_values
        self.tree_type = tree_type


    @property
    def fractal_dimension(self):
        try:
            fractal_dimension = self.tree_properties['fractal_dimension'] if self.tree_properties else None
        except:
            fractal_dimension = 0
        return fractal_dimension
    

    @property
    def num_nodes(self):
        return self.tree_properties['num_nodes'] if self.tree_properties else None
        

    @property
    def num_edges(self):
        return self.tree_properties['num_edges'] if self.tree_properties else None
        

    # @property
    # def root_node(self):
    #     return self.tree_properties['root_node'] if self.tree_properties else None
        

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
    def num_leaf_nodes(self):
        return self.tree_properties['num_leaf_nodes'] if self.tree_properties else None
        

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
    def from_properties_csv(cls, tree_filepath, c_filepath, id=None, tree_type=None):
        if id is None:
            id = tree_filepath.split('/')[-1].split('.')[0].split('_')[0]
        with open(tree_filepath, 'r') as f:
            tree_json = f.read()
        with open(c_filepath, 'r') as f:
            c_json = f.read()
        instance = cls(id, json.loads(tree_json), json.loads(c_json), tree_type=tree_type)
        return instance
    

    def get_all_properties(self):
        properties = [
            name for name, value in AxonDendriteTree.__dict__.items()
            if isinstance(value, property)
        ]
        return properties

    def get_property_values(self):
        return [getattr(self, prop) for prop in self.get_all_properties()]


class AxonDendriteForest():
    def __init__(self, tree_type):
        self.axon_tree = []
        self.dendrite_tree = []
        self.tree_type = tree_type


    def add_axon_tree(self, axon_tree):
        self.axon_tree.append(axon_tree)


    def add_dendrite_tree(self, dendrite_tree):
        self.dendrite_tree.append(dendrite_tree)

    
    def get_property_list(self, property_name, tree_type='axon'):
        property_list = []
        if tree_type == 'axon':
            for tree in self.axon_tree:
                property_list.append(getattr(tree, property_name))
        elif tree_type == 'dendrite':
            for tree in self.dendrite_tree:
                property_list.append(getattr(tree, property_name))
        return property_list
    
    

    @classmethod
    def build_forest_from_directory(cls, tree_dir, c_dir, undirected=True):
        tree_type = 'undirected' if undirected else 'directed'
        forest = cls(tree_type)
        # ids = _get_skeleton_ids(tree_dir)
        ids = _get_filtered_skeleton_ids()
        for id in ids:
            tree_filepath_axon = f'{tree_dir}/{id}_axon.json'
            c_filepath_axon = f'{c_dir}/{id}_axon.json'
            tree_filepath_dendrite = f'{tree_dir}/{id}_dendrite.json'
            c_filepath_dendrite = f'{c_dir}/{id}_dendrite.json'
            if os.path.exists(tree_filepath_axon) and os.path.exists(c_filepath_axon):
                if os.path.exists(tree_filepath_dendrite) and os.path.exists(c_filepath_dendrite):
                    tree = AxonDendriteTree.from_properties_csv(tree_filepath_axon, c_filepath_axon, tree_type=tree_type, id=id)
                    forest.add_axon_tree(tree)
                    tree = AxonDendriteTree.from_properties_csv(tree_filepath_dendrite, c_filepath_dendrite, tree_type=tree_type, id=id)
                    forest.add_dendrite_tree(tree)
            else:
                print(f"Missing files for skeleton ID {id}, skipping. \naxon files: {os.path.exists(tree_filepath_axon)}, {os.path.exists(c_filepath_axon)}; \n{tree_filepath_axon} and {c_filepath_axon} \ndendrite files: {os.path.exists(tree_filepath_dendrite)}, {os.path.exists(c_filepath_dendrite)}\n{tree_filepath_dendrite} and {c_filepath_dendrite}")
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
