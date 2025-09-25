from fafbseg import flywire
from collections import Counter

class Synapses:
    def __init__(self, skeleton_tree):
        self.skeleton_tree = skeleton_tree
        self.synapses = self._read_synapses()
        self.pre_synapses = self._get_pre_synapses()
        self.post_synapses = self._get_post_synapses()
        self.filtered_pre_synapses = self._get_filtered_synapses(self.pre_synapses, threshold=5)
        self.filtered_post_synapses = self._get_filtered_synapses(self.post_synapses, threshold=5)

    def _read_synapses(self):
        return self.skeleton_tree.skeleton.connectors

    def _get_pre_synapses(self):
        """
        Returns a list of pre-synapses.
        """
        pre_synapses = self.synapses[(self.synapses['type'] == 'pre')]['partner_id'].tolist()
        return pre_synapses

    def _get_post_synapses(self):
        """
        Returns a list of post-synapses.
        """
        post_synapses = self.synapses[(self.synapses['type'] == 'post')]['partner_id'].tolist()
        return post_synapses

    def _get_filtered_synapses(self, synapses:list, threshold: int = 5) -> list:
        """
        Filters synapses based on a threshold.

        Args:
            synapses (list): List of synapses to filter.
            threshold (int): Minimum number of connections required to keep a synapse. Default is 5.

        Returns:
            list: Filtered list of synapses.
        """
        all_synapses = self.pre_synapses + self.post_synapses
        counts = Counter(all_synapses)
        return [x for x in synapses if counts[x] >= threshold]