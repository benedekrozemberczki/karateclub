import networkx as nx
from typing import Dict
from karateclub.estimator import Estimator

class SCD(Estimator):
    r"""An implementation of `"SCD" <http://wwwconference.org/proceedings/www2014/proceedings/p225.pdf>`_ from the
    WWW '14 paper "High Quality, Scalable and Parallel Community Detection for 
    Large Real Graphs". The procedure greedily optimizes the approximate weighted
    community clustering metric. First, clusters are built around highly clustered nodes.
    Second, we refine the initial partition by using the approximate WCC. These refinements
    happen for the whole vertex set.

    Args:
        iterations (int): Refinemeent iterations. Default is 25.
        eps (float): Epsilon score for zero division correction. Default is 10**-6.
        seed (int): Random seed value. Default is 42.
    """
    def __init__(self, iterations: int=25, eps: float=10**-6, seed: int=42):
        self.iterations = iterations
        self.eps = eps
        self.seed = seed

    def _set_omega(self):
        """
        Calculating the graph level clustering coefficient.
        """
        self._omega = nx.transitivity(self._graph)

    def _set_nodes(self):
        """
        Creating a list of nodes.
        """
        self._nodes = [node for node in self._graph.nodes()]

    def _create_initial_partition(self):
        """
        Initial local clustering coefficient based cluster membership assignments.
        """
        self._clustering_coefficient = nx.clustering(self._graph)
        self._cc_pairs = [(node_cc**0.5, node) for node, node_cc in self._clustering_coefficient.items()]
        self._cc_pairs = sorted(self._cc_pairs, key=lambda tup: tup[0])[::-1]
        self._do_initial_assignments()

    def _do_initial_assignments(self):
        """
        Creating the cluster membership hash table.
        """
        self._cluster_memberships = {}
        neighbor_memberships = {}
        cluster_index = 0 
        for pair in self._cc_pairs:
            if pair[1] in neighbor_memberships:
                self._cluster_memberships[pair[1]] = neighbor_memberships[pair[1]]
            else:
                self._cluster_memberships[pair[1]] = cluster_index
                for neighbor in self._graph.neighbors(pair[1]):
                    if neighbor not in neighbor_memberships:
                        neighbor_memberships[neighbor] = cluster_index
                cluster_index = cluster_index + 1

    def _create_inverse_community_index(self):
        """
        Creating a community - node list index.
        """
        inverse_community_index = {}
        for node, cluster_membership in self._cluster_memberships.items():
            if cluster_membership in inverse_community_index:
                inverse_community_index[cluster_membership] = inverse_community_index[cluster_membership].union({node})
            else:
                inverse_community_index[cluster_membership] = {node}
        return inverse_community_index

    def _calculate_community_statistics(self, inverse_community_index):
        """
        Calculating the community level statistics used for refinement. 
        """
        community_statistics = {}
        for comm, members in inverse_community_index.items():
            induced_graph = self._graph.subgraph(members)
            size = induced_graph.number_of_nodes()
            density = nx.density(induced_graph)
            edge_out = sum([0 if neighbor in induced_graph else 1 for node in members for neighbor in self._graph.neighbors(node)])
            community_statistics[comm] = {"r":size, "d": density, "b": edge_out}
        return community_statistics

    def _calculate_theta_1(self, r, d, b, q, d_out, d_in):
        """
        Calculating the 1st WCC component.
        """
        theta_1_enum = (r-1)*d+1+q
        theta_1_denom = (r+q)*((r-1)*(r-2)*(d**3)+(d_in-1)*d+q*(q-1)*(d+1)*self._omega+d_out*self._omega)+self.eps
        theta_1_multi = (d_in-1)*d
        theta_1 = (theta_1_enum / theta_1_denom)*theta_1_multi
        return theta_1

    def _calculate_theta_2(self, r, d, b, q):
        """
        Calculating the 2nd WCC component.
        """
        theta_2_left_enum = (r-1)*(r-2)*(d**3)
        theta_2_left_denom = theta_2_left_enum+q*(q-1)*self._omega+q*(r-1)*self._omega*d+self.eps
        theta_2_right_enum = (r-1)*d+q
        theta_2_right_denom = (r+q)*(r-1+q)+self.eps
        theta_2 = -(theta_2_left_enum/theta_2_left_denom)*(theta_2_right_enum/theta_2_right_denom)
        return theta_2

    def _calculate_theta_3(self, r, d, b, q, d_out, d_in):
        """
        Calculating the 3rd WCC component.
        """
        theta_3_left_enum = d_in*(d_in-1)*d
        theta_3_left_denom = theta_3_left_enum + d_out*(d_out-1)*self._omega+d_out*d_in*self._omega+self.eps
        theta_3_right_enum = d_in+d_out
        theta_3_right_denom = r+d_out+self.eps
        theta_3 = (theta_3_left_enum/theta_3_left_denom)*(theta_3_right_enum/theta_3_right_denom)
        return theta_3

    def _calculate_wcc(self, community_level_stats, d_out, d_in):
        """
        Calculating the WCC.
        """
        r = community_level_stats["r"]
        d = community_level_stats["d"]
        b = community_level_stats["b"]
        q = (b-d_in)/r
        theta_1 = self._calculate_theta_1(r, d, b, q, d_out, d_in)
        theta_2 = self._calculate_theta_2(r, d, b, q)
        theta_3 = self._calculate_theta_3(r, d, b, q, d_out, d_in)
        wcc = d_in*theta_1 + (r-d_in)*theta_2+theta_3
        return wcc

    def _find_community_index(self, community_statistics):
        """
        Finding the current community index.
        """
        return max(community_statistics.keys())+1

    def _do_refinement(self):
        new_memberships = {}
        inverse_community_index = self._create_inverse_community_index()
        community_statistics = self._calculate_community_statistics(inverse_community_index)
        community_index = self._find_community_index(community_statistics)
        for node in self._nodes:
            community_level_stats = community_statistics[self._cluster_memberships[node]]
            neighbors = set([neighbor for neighbor in self._graph.neighbors(node)])
            candidate_communities = [self._cluster_memberships[neighbor] for neighbor in neighbors]
            d_out = len(set(neighbors).difference(inverse_community_index[self._cluster_memberships[node]]))
            d_in = len(neighbors) - d_out
            WCC_r = -self._calculate_wcc(community_level_stats, d_out, d_in)
            WCC_t = 0
            best_community = None
            for comm in candidate_communities:
                 community_level_stats = community_statistics[comm]
                 d_out = len(set(neighbors).difference(inverse_community_index[comm]))
                 d_in = len(neighbors) - d_out
                 WCC_aux = self._calculate_wcc(community_level_stats, d_out, d_in)+WCC_r
                 if WCC_aux > WCC_t:
                     WCC_t = WCC_aux
                     best_community = comm
            if WCC_r > WCC_t and WCC_r > 0: 
                new_memberships[node] = community_index
                community_index = community_index + 1
            elif WCC_t > WCC_r and WCC_t > 0:
                new_memberships[node] = best_community
            else:
                new_memberships[node] = self._cluster_memberships[node]
        self._cluster_memberships = new_memberships

    def fit(self, graph: nx.classes.graph.Graph):
        """
        Fitting a Label Propagation clustering model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be clustered.
        """
        self._set_seed()
        self._check_graph(graph)
        self._graph = graph
        self._create_initial_partition()
        self._set_omega()
        self._set_nodes()
        for _ in range(self.iterations):
            self._do_refinement()

    def get_memberships(self) -> Dict[int, int]:
        r"""Getting the cluster membership of nodes.

        Return types:
            * **memberships** *(dict)* - Node cluster memberships.
        """
        memberships = self._cluster_memberships
        return memberships
