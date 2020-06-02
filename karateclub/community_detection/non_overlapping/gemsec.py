import random
import numpy as np
import networkx as nx
from karateclub.utils.walker import RandomWalker
from karateclub.estimator import Estimator

class GEMSEC(Estimator):
    r"""An implementation of `"GEMSEC" <https://arxiv.org/abs/1802.03997>`_
    from the ASONAM '19 paper "GEMSEC: Graph Embedding with Self Clustering".
    The procedure uses random walks to approximate the pointwise mutual information
    matrix obtained by pooling normalized adjacency matrix powers. This matrix
    is decomposed by an approximate factorization technique which is combined
    with a k-means like clustering cost. A node embedding and clustering are
    learned jointly.

    Args:
        walk_number (int): Number of random walks. Default is 5.
        walk_length (int): Length of random walks. Default is 80.
        dimensions (int): Dimensionality of embedding. Default is 32.
        negative_samples (int): Number of negative samples. Default is 5.
        window_size (int): Matrix power order. Default is 5.
        learning_rate (float): Gradient descent learning rate. Default is 0.1.
        clusters (int): Number of cluster centers. Default is 10.
        gamma (float): Clustering cost weight coefficient. Default is 0.1.
        seed (int): Random seed value. Default is 42.
    """
    def __init__(self, walk_number=5, walk_length=80, dimensions=32, negative_samples=5,
                 window_size=5, learning_rate=0.1, clusters=10, gamma=0.1, seed=42):

        self.walk_number = walk_number
        self.walk_length = walk_length
        self.dimensions = dimensions
        self.negative_samples = negative_samples
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.clusters = clusters
        self.gamma = gamma
        self.seed = seed


    def _setup_sampling_weights(self, graph):
        """
        Creating a negative sampling table.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph for negative sampling.
        """
        self._sampler = {}
        index = 0
        for node in graph.nodes():
            for _ in range(graph.degree(node)):
                self._sampler[index] = node
                index = index + 1
        self._global_index = index-1


    def _initialize_node_embeddings(self, graph):
        """
        Creating a node embedding array.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph for negative sampling.
        """
        shape = (graph.number_of_nodes(), self.dimensions)
        self._base_embedding = np.random.normal(0, 1.0/self.dimensions, shape)


    def _initialize_cluster_centers(self, graph):
        """
        Creating a cluster center array.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph for negative sampling.
        """
        shape = (self.dimensions, self.clusters)
        self._cluster_centers = np.random.normal(0, 1.0/self.dimensions, shape)


    def _sample_negative_samples(self):
        """
        Sampling a batch of nodes as negative samples.

        Return types:
            * **negative_samples** *(list)*: List of negative sampled nodes.
        """
        negative_samples = [self._sampler[random.randint(0,self._global_index)] for _ in range(self.negative_samples)]
        return negative_samples


    def _calculcate_noise_vector(self, negative_samples, source_node):
        """
        Getting the noise vector for the weight update.

        Arg types:
            * **negative_samples** *(list)*: List of negative sampled nodes.
            * **source_node** *(int)* - Source node in the walk.

        Return types:
            * **noise_vector** *(NumPy array) - Noise update vector.
        """
        noise_vectors = self._base_embedding[negative_samples, :]
        source_vector = self._base_embedding[int(source_node), :]
        raw_scores = noise_vectors.dot(source_vector.T)
        raw_scores = np.exp(np.clip(raw_scores, -15, 15))
        scores = raw_scores/np.sum(raw_scores)
        scores = scores.reshape(-1,1)
        noise_vector = np.sum(scores*noise_vectors,axis=0)
        return noise_vector


    def _calculate_cluster_vector(self, source_node):
        """
        Getting the cluster vector for the weight update.

        Arg types:
            * **source_node** *(int)* - Source node in the walk.

        Return types:
            * **cluster_vector** *(NumPy array) - Cluster update vector.
            * **cluster_index** *(int)*: Node cluster membership index.
        """
        distances = self._base_embedding[int(source_node), :].reshape(-1,1) - self._cluster_centers
        scores = np.power(np.sum(np.power(distances,2),axis=0),0.5)
        cluster_index = np.argmin(scores)
        cluster_vector = distances[:,cluster_index]/scores[cluster_index]
        return cluster_vector, cluster_index


    def _do_descent_for_pair(self, negative_samples, source_node, target_node):
        """
        Updating the cluster center and the node embedding.

        Arg types:
            * **negative_samples** *(list)* - Negative samples.
            * **source_node** *(int)* - Source node in the walk.
            * **target_node** *(int)* - Target node in the walk.
        """
        noise_vector = self._calculcate_noise_vector(negative_samples, source_node)
        target_vector = self._base_embedding[int(target_node), :]
        cluster_vector, cluster_index = self._calculate_cluster_vector(source_node)
        node_gradient = noise_vector - target_vector + self.gamma*cluster_vector
        node_gradient = node_gradient / np.linalg.norm(node_gradient)
        self._base_embedding[int(source_node), :] += -self.learning_rate*node_gradient
        self._cluster_centers[:, cluster_index] += self.learning_rate*self.gamma*cluster_vector 


    def _update_a_weight(self, source_node, target_node):
        """
        Updating the weights for a pair of nodes.

        Arg types:
            * **source_node** *(int)* - Source node in the walk.
            * **target_node** *(int)* - Target node in the walk.
        """
        negative_samples = self._sample_negative_samples()
        self._do_descent_for_pair(negative_samples, source_node, target_node)
        self._do_descent_for_pair(negative_samples, target_node, source_node)


    def _do_gradient_descent(self):
        """
        Updating the embedding weights and cluster centers with gradient descent.
        """
        random.shuffle(self._walker.walks)
        for walk in self._walker.walks:
            for i, source_node in enumerate(walk[:self.walk_length-self.window_size]):
                for step in range(1, self.window_size+1):
                    target_node = walk[i+step]
                    self._update_a_weight(source_node, target_node)


    def fit(self, graph):
        """
        Fitting a GEMSEC model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        """
        self._set_seed()
        self._check_graph(graph)
        self._setup_sampling_weights(graph)
        self._walker = RandomWalker(self.walk_length, self.walk_number)
        self._walker.do_walks(graph)
        self._initialize_node_embeddings(graph)
        self._initialize_cluster_centers(graph)
        self._do_gradient_descent()


    def get_embedding(self):
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)*: The embedding of nodes.
        """
        return np.array(self._base_embedding)


    def _get_membership(self, node):
        """Getting the cluster membership of a node.

        Arg types:
            * **node** *(int)* - The graph to be clustered.

        Return types:
            * **cluster_index** *(int)*: Node cluster membership index.
        """
        distances = self._base_embedding[node, :].reshape(-1, 1) - self._cluster_centers
        scores = np.power(np.sum(np.power(distances,2), axis=0), 0.5)
        cluster_index = np.argmin(scores)   
        return cluster_index


    def get_memberships(self):
        r"""Getting the cluster membership of nodes.

        Return types:
            * **memberships** *(dict)*: Node cluster memberships.
        """
        memberships = {node:self._get_membership(node) for node in range(self._base_embedding.shape[0])}
        return memberships 
