import navis

def split(neuron, node_id):
    return navis.cut_skeleton(neuron, node_id + 1, ret='distal')