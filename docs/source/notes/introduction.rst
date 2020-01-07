Introduction by Example
=======================

We shortly overview the fundamental concepts and features of KarateClub through simple examples. These are the following:

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

Now let's use the ``Label Propagation`` community detection method from `Near Linear Time Algorithm to Detect
 Community Structures in Large-Scale Networks <https://arxiv.org/abs/0709.2938>`_. 
.. code-block:: python

    from karateclub.community_detection.non_overlapping import LabelPropagation
    
    model = LabelPropagation()
    model.fit(graph)
    cluster_membership = model.get_memberships()

The constructor defines a model, we fit the model on the Facebook graph with the ``fit`` method and return the cluster memberships
with the ``get_memberships`` method as a dictionary.


Finally we can evaluate the clustering using normalized mutual information. First we need to create an ordered list of the node memberships.


.. code-block:: python

    from sklearn.metrics.cluster import normalized_mutual_info_score

    cluster_membership = [cluster_membership[node] for node in range(len(cluster_membership))]

    nmi = normalized_mutual_info_score(target, cluster_membership)
    print('NMI: {:.4f}'.format(nmi))
    >>> NMI: 0.34374


Node Embedding
--------------

Graph Embedding
--------------


Benchmark Datasets
------------------

PyTorch Geometric contains a large number of common benchmark datasets, *e.g.* all Planetoid datasets (Cora, Citeseer, Pubmed), all graph classification datasets from `http://graphkernels.cs.tu-dortmund.de/ <http://graphkernels.cs.tu-dortmund.de/>`_ and their `cleaned versions <https://github.com/nd7141/graph_datasets>`_, the QM7 and QM9 dataset, and a handful of 3D mesh/point cloud datasets like FAUST, ModelNet10/40 and ShapeNet.

Initializing a dataset is straightforward.
An initialization of a dataset will automatically download its raw files and process them to the previously described ``Data`` format.
*E.g.*, to load the ENZYMES dataset (consisting of 600 graphs within 6 classes), type:

.. code-block:: python

    from torch_geometric.datasets import TUDataset

    dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    >>> ENZYMES(600)

    len(dataset)
    >>> 600

    dataset.num_classes
    >>> 6

    dataset.num_node_features
    >>> 3

We now have access to all 600 graphs in the dataset:

.. code-block:: python

    data = dataset[0]
    >>> Data(edge_index=[2, 168], x=[37, 3], y=[1])

    data.is_undirected()
    >>> True

We can see that the first graph in the dataset contains 37 nodes, each one having 3 features.
There are 168/2 = 84 undirected edges and the graph is assigned to exactly one class.
In addition, the data object is holding exactly one graph-level target.

We can even use slices, long or byte tensors to split the dataset.
*E.g.*, to create a 90/10 train/test split, type:

.. code-block:: python

    train_dataset = dataset[:540]
    >>> ENZYMES(540)

    test_dataset = dataset[540:]
    >>> ENZYMES(60)

If you are unsure whether the dataset is already shuffled before you split, you can randomly permutate it by running:

.. code-block:: python

    dataset = dataset.shuffle()
    >>> ENZYMES(600)

This is equivalent of doing:

.. code-block:: python

    perm = torch.randperm(len(dataset))
    dataset = dataset[perm]
    >> ENZYMES(600)

Let's try another one! Let's download Cora, the standard benchmark dataset for semi-supervised graph node classification:

.. code-block:: python

    from torch_geometric.datasets import Planetoid

    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    >>> Cora()

    len(dataset)
    >>> 1

    dataset.num_classes
    >>> 7

    dataset.num_node_features
    >>> 1433

Here, the dataset contains only a single, undirected citation graph:

.. code-block:: python

    data = dataset[0]
    >>> Data(edge_index=[2, 10556], test_mask=[2708],
             train_mask=[2708], val_mask=[2708], x=[2708, 1433], y=[2708])

    data.is_undirected()
    >>> True

    data.train_mask.sum().item()
    >>> 140

    data.val_mask.sum().item()
    >>> 500

    data.test_mask.sum().item()
    >>> 1000

This time, the ``Data`` objects holds a label for each node, and additional attributes: ``train_mask``, ``val_mask`` and ``test_mask``:

- ``train_mask`` denotes against which nodes to train (140 nodes)
- ``val_mask`` denotes which nodes to use for validation, *e.g.*, to perform early stopping (500 nodes)
- ``test_mask`` denotes against which nodes to test (1000 nodes)


