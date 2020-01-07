Introduction by Example
=======================

We shortly overview the fundamental concepts and features of Karate Club through simple examples. These are the following:

.. contents::
    :local:

Standardized Dataset Ingestion
------------------------------


Community Detection
-------------------

The first machine learning task that we will do is the clustering of pages on Facebook. In this network
nodes represent official verified Facebook pages and the links between them are mutual likes. The pages
have categories and we will look how well the cluster and group memberships are aligned. For details
about the dataset `see this paper <https://arxiv.org/abs/1909.13021>`_.

We first need to load the Facebook page-page network dataset. We will use the page-page graph and the 
page category vector. These are returned as a ``NetworkX`` graph and ``numpy`` array respectively.

.. code-block:: python

    from karateclub.dataset import GraphReader

    reader = GraphReader("facebook")

    graph = reader.get_graph()
    target = reader.get_target()

The constructor defines the graphreader object and the methods ``get_graph`` and ``get_target`` read the data.

Now let's use the ``Label Propagation`` community detection method from `Near Linear Time Algorithm to Detect Community Structures in Large-Scale Networks <https://arxiv.org/abs/0709.2938>`_. 

.. code-block:: python

    from karateclub.community_detection.non_overlapping import LabelPropagation
    
    model = LabelPropagation()
    model.fit(graph)
    cluster_membership = model.get_memberships()

The constructor defines a model, we fit the model on the Facebook graph with the ``fit`` method and return the cluster memberships
with the ``get_memberships`` method as a dictionary.


Finally we can evaluate the clustering using normalized mutual information. First we need to create an ordered list of the node memberships.
We use the ground truth about the cluster memberships for calculating the NMI.


.. code-block:: python

    from sklearn.metrics.cluster import normalized_mutual_info_score

    cluster_membership = [cluster_membership[node] for node in range(len(cluster_membership))]

    nmi = normalized_mutual_info_score(target, cluster_membership)
    print('NMI: {:.4f}'.format(nmi))
    >>> NMI: 0.34374

It is worth noting that the clustering methods in Karate Club work on arbitrary NetworkX graphs that follow the 
dataset formatting requirements. One could simply cluster a randomly generated Watts-Strogatz graph just like this.

.. code-block:: python

    import networkx as nx
    from karateclub.community_detection.non_overlapping import LabelPropagation
    
    graph = nx.newman_watts_strogatz_graph(100, 20, 0.05)

    model = LabelPropagation()
    model.fit(graph)
    cluster_membership = model.get_memberships()  


Node Embedding
--------------

Graph Embedding
--------------


Benchmark Datasets
------------------


