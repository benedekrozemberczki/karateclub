import random
import networkx as nx


class EulerianDiffuser:
    """
    Class to make diffusions for a given graph.

    Args:
        diffusion_number (int): Number of diffusions
        diffusion_cover (int): Number of nodes in diffusion.
    """
    def __init__(self, diffusion_number: int, diffusion_cover: int):
        self.diffusion_number = diffusion_number
        self.diffusion_cover = diffusion_cover


    def _run_diffusion_process(self, node):
        """
        Generating a diffusion tree from a given source node and linearizing it
        with a directed Eulerian tour.

        Arg types:
            * **node** *(int)* - The source node of the diffusion.
        Return types:
            * **euler** *(list of strings)* - The list of nodes in the walk.
        """
        infected = [node]
        sub_graph = nx.DiGraph()
        sub_graph.add_node(node)
        infected_counter = 1
        while infected_counter < self.diffusion_cover:
            end_point = random.sample(infected, 1)[0]
            nebs = [node for node in self.graph.neighbors(end_point)]
            sample = random.choice(nebs)
            if sample not in infected:
                infected_counter = infected_counter + 1
                infected = infected + [sample]
                sub_graph.add_edges_from([(end_point, sample), (sample, end_point)])
                if infected_counter == self.diffusion_cover:
                    break
        euler = [str(u) for u, v in nx.eulerian_circuit(sub_graph, infected[0])]
        return euler


    def do_diffusions(self, graph: nx.classes.graph.Graph):
        """
        Running diffusions from every node.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to run diffusions on.
        """
        self.graph = graph
        self.diffusions = []
        for _ in range(self.diffusion_number):
            for node in self.graph.nodes():
                diffusion_sequence = self._run_diffusion_process(node)
                self.diffusions.append(diffusion_sequence)
