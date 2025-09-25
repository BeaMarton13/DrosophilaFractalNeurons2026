import csv

class HandleFilteredConnector:
    """
    Class to handle filtered connectors in a skeleton tree.
    """

    def __init__(self, skeleton_tree_id: int):
        self.skeleton_tree_id = skeleton_tree_id
        try:
            self.synapses = self.read_synapses()
        except ValueError as e:
            print(f"Error reading synapses: {e}")
            self.synapses = {"pre": [], "post": []}

    @property
    def pre_synapses(self) -> list:
        """
        Returns a list of pre-synapses.
        """
        return self.synapses.get('pre', [])

    @property
    def post_synapses(self) -> list:
        """
        Returns a list of post-synapses.
        """
        return self.synapses.get('post', [])

    def read_synapses(self, filtered_connectors_filelist: str="filtered_connectors/filtered_connectors.filelist") -> dict:
        """
        Reads filtered connectors from a CSV file.

        :param filtered_connectors_filelist: The name of the file containing filtered connectors.
        :return: A dictionary containing filtered connectors.
        """
        filtered_connectors = {}
        with open(filtered_connectors_filelist, 'r') as file:
            filelist = [x.strip('\n') for x in file.readlines()]
            if f"{self.skeleton_tree_id}.csv" not in filelist:
                raise ValueError(f"Skeleton tree ID {self.skeleton_tree_id}.csv not found in the file list.")
            else:
                synapses = self._read_pre_post_synapses(f"filtered_connectors/{self.skeleton_tree_id}.csv")
        return synapses

    def _read_pre_post_synapses(self, filename: str) -> dict:
        """
        Reads a CSV file and returns its content as a dictionary.

        :param filename: The name of the CSV file to read.
        :return: A dictionary containing pre and post synapses.
        """
        with open(filename, 'r') as csvfile:
            pre_synapses = []
            post_synapses = []
            dict_reader = csv.DictReader(csvfile)
            for row in dict_reader:
                if row['type'] == 'pre':
                    pre_synapses.append(int(row['partner_id']))
                elif row['type'] == 'post':
                    post_synapses.append(int(row['partner_id']))
            return {"pre": pre_synapses, "post": post_synapses}