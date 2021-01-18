
 ![Version](https://badge.fury.io/py/karateclub.svg?style=plastic)
 ![License](https://img.shields.io/github/license/benedekrozemberczki/karateclub.svg)
[![repo size](https://img.shields.io/github/repo-size/benedekrozemberczki/karateclub.svg)](https://github.com/benedekrozemberczki/karateclub/archive/master.zip)
 [![Arxiv](https://img.shields.io/badge/ArXiv-2003.04819-orange.svg)](https://arxiv.org/abs/2003.04819)
[![build badge](https://github.com/benedekrozemberczki/karateclub/workflows/CI/badge.svg)](https://github.com/benedekrozemberczki/karateclub/actions?query=workflow%3ACI)
 [![coverage badge](https://codecov.io/gh/benedekrozemberczki/karateclub/branch/master/graph/badge.svg)](https://codecov.io/github/benedekrozemberczki/karateclub?branch=master)
<p align="center">
  <img width="90%" src="https://github.com/benedekrozemberczki/karateclub/blob/master/karatelogo.jpg?sanitize=true" />
</p>

--------------------------------------------------------------------------------


**Karate Club** is an unsupervised machine learning extension library for [NetworkX](https://networkx.github.io/).


Please look at the **[Documentation](https://karateclub.readthedocs.io/)**, relevant **[Paper](https://arxiv.org/abs/2003.04819)**, **[Promo Video](https://www.youtube.com/watch?v=t212-ntxu2U)**, and **[External Resources](https://karateclub.readthedocs.io/en/latest/notes/resources.html)**.

*Karate Club* consists of state-of-the-art methods to do unsupervised learning on graph structured data. To put it simply it is a Swiss Army knife for small-scale graph mining research. First, it provides network embedding techniques at the node and graph level. Second, it includes a variety of overlapping and non-overlapping community detection methods. Implemented methods cover a wide range of network science ([NetSci](https://netscisociety.net/home), [Complenet](https://complenet.weebly.com/)), data mining ([ICDM](http://icdm2019.bigke.org/), [CIKM](http://www.cikm2019.net/), [KDD](https://www.kdd.org/kdd2020/)), artificial intelligence ([AAAI](http://www.aaai.org/Conferences/conferences.php), [IJCAI](https://www.ijcai.org/)) and machine learning ([NeurIPS](https://nips.cc/), [ICML](https://icml.cc/), [ICLR](https://iclr.cc/)) conferences, workshops, and pieces from prominent journals.

The newly introduced graph classification datasets are available at [SNAP](https://snap.stanford.edu/data/#disjointgraphs), [TUD Graph Kernel Datasets](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets), and [GraphLearning.io](https://chrsmrrs.github.io/datasets/).

--------------------------------------------------------------------------------

**Citing**

If you find *Karate Club* and the new datasets useful in your research, please consider citing the following paper:

```bibtex
@inproceedings{karateclub,
       title = {{Karate Club: An API Oriented Open-source Python Framework for Unsupervised Learning on Graphs}},
       author = {Benedek Rozemberczki and Oliver Kiss and Rik Sarkar},
       year = {2020},
       pages = {3125–3132},
       booktitle = {Proceedings of the 29th ACM International Conference on Information and Knowledge Management (CIKM '20)},
       organization = {ACM},
}
```
--------------------------------------------------------------------------------

**A simple example**

*Karate Club* makes the use of modern community detection techniques quite easy (see [here](https://karateclub.readthedocs.io/en/latest/notes/introduction.html) for the accompanying tutorial). For example, this is all it takes to use on a Watts-Strogatz graph [Ego-splitting](https://www.eecs.yorku.ca/course_archive/2017-18/F/6412/reading/kdd17p145.pdf):

```python
import networkx as nx
from karateclub import EgoNetSplitter

g = nx.newman_watts_strogatz_graph(1000, 20, 0.05)

splitter = EgoNetSplitter(1.0)

splitter.fit(g)

print(splitter.get_memberships())
```

--------------------------------------------------------------------------------

**Models included**

In detail, the following community detection and embedding methods were implemented.

**Overlapping Community Detection**

* **[DANMF](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.community_detection.overlapping.danmf.DANMF)** from Ye *et al.*: [Deep Autoencoder-like Nonnegative Matrix Factorization for Community Detection](https://github.com/benedekrozemberczki/DANMF/blob/master/18DANMF.pdf) (CIKM 2018)

* **[M-NMF](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.community_detection.overlapping.mnmf.M_NMF)** from Wang *et al.*: [Community Preserving Network Embedding](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14589) (AAAI 2017)

* **[Ego-Splitting](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.community_detection.overlapping.ego_splitter.EgoNetSplitter)** from Epasto *et al.*: [Ego-splitting Framework: from Non-Overlapping to Overlapping Clusters](https://www.eecs.yorku.ca/course_archive/2017-18/F/6412/reading/kdd17p145.pdf) (KDD 2017)

* **[NNSED](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.community_detection.overlapping.nnsed.NNSED)** from Sun *et al.*: [A Non-negative Symmetric Encoder-Decoder Approach for Community Detection](http://www.bigdatalab.ac.cn/~shenhuawei/publications/2017/cikm-sun.pdf) (CIKM 2017)

* **[BigClam](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.community_detection.overlapping.bigclam.BigClam)** from Yang and Leskovec: [Overlapping Community Detection at Scale: A Nonnegative Matrix Factorization Approach](http://infolab.stanford.edu/~crucis/pubs/paper-nmfagm.pdf) (WSDM 2013)

* **[SymmNMF](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.community_detection.overlapping.symmnmf.SymmNMF)** from Kuang *et al.*: [Symmetric Nonnegative Matrix Factorization for Graph Clustering](https://www.cc.gatech.edu/~hpark/papers/DaDingParkSDM12.pdf) (SDM 2012)

**Non-Overlapping Community Detection**

* **[GEMSEC](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.community_detection.non_overlapping.gemsec.GEMSEC)** from Rozemberczki *et al.*: [GEMSEC: Graph Embedding with Self Clustering](https://arxiv.org/abs/1802.03997) (ASONAM 2019)

* **[EdMot](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.community_detection.non_overlapping.edmot.EdMot)** from Li *et al.*: [EdMot: An Edge Enhancement Approach for Motif-aware Community Detection](https://arxiv.org/abs/1906.04560) (KDD 2019)

* **[SCD](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.community_detection.non_overlapping.scd.SCD)** from Prat-Perez *et al.*: [High Quality, Scalable and Parallel Community Detectionfor Large Real Graphs](http://wwwconference.org/proceedings/www2014/proceedings/p225.pdf) (WWW 2014)

* **[Label Propagation](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.community_detection.non_overlapping.label_propagation.LabelPropagation)** from Raghavan *et al.*: [Near Linear Time Algorithm to Detect Community Structures in Large-Scale Networks](https://arxiv.org/abs/0709.2938) (Physics Review E 2007)

**Neighbourhood-Based Node Level Embedding**

* **[SocioDim](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.node_embedding.neighbourhood.sociodim.SocioDim)** from Tang *et al.*: [Relational Learning via Latent Social Dimensions](ttp://www.public.asu.edu/~huanliu/papers/kdd09.pdf) (KDD 2009)

* **[GLEE](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.node_embedding.neighbourhood.geometriclaplacianeigenmaps.GLEE)** from Torres *et al.*: [GLEE: Geometric Laplacian Eigenmap Embedding](https://arxiv.org/abs/1905.09763) (Journal of Complex Networks 2020)

* **[BoostNE](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.node_embedding.neighbourhood.boostne.BoostNE)** from Li *et al.*: [Multi-Level Network Embedding with Boosted Low-Rank Matrix Approximation](https://arxiv.org/abs/1808.08627) (ASONAM 2019)

* **[NodeSketch](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.node_embedding.neighbourhood.nodesketch.NodeSketch)**  from Yang *et al.*: [NodeSketch: Highly-Efficient Graph Embeddings via Recursive Sketching](https://exascale.info/assets/pdf/yang2019nodesketch.pdf) (KDD 2019)

* **[Diff2Vec](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.node_embedding.neighbourhood.diff2vec.Diff2Vec)** from Rozemberczki and Sarkar: [Fast Sequence Based Embedding with Diffusion Graphs](https://arxiv.org/abs/2001.07463) (CompleNet 2018)

* **[NetMF](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.node_embedding.neighbourhood.netmf.NetMF)** from Qiu *et al.*: [Network Embedding as Matrix Factorization: Unifying DeepWalk, LINE, PTE, and Node2Vec](https://keg.cs.tsinghua.edu.cn/jietang/publications/WSDM18-Qiu-et-al-NetMF-network-embedding.pdf) (WSDM 2018)

* **[RandNE](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.node_embedding.neighbourhood.randne.RandNE)** from Zhang *et al.*: [Billion-scale Network Embedding with Iterative Random Projection](https://arxiv.org/abs/1805.02396) (ICDM 2018)

* **[Walklets](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.node_embedding.neighbourhood.walklets.Walklets)** from Perozzi *et al.*: [Don't Walk, Skip! Online Learning of Multi-scale Network Embeddings](https://arxiv.org/abs/1605.02115) (ASONAM 2017)

* **[HOPE](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.node_embedding.neighbourhood.hope.HOPE)** from Ou *et al.*: [Asymmetric Transitivity Preserving Graph Embedding](https://dl.acm.org/doi/abs/10.1145/2939672.2939751) (KDD 2016)

* **[GraRep](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.node_embedding.neighbourhood.grarep.GraRep)** from Cao *et al.*: [GraRep: Learning Graph Representations with Global Structural Information](https://dl.acm.org/citation.cfm?id=2806512) (CIKM 2015)

* **[DeepWalk](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.node_embedding.neighbourhood.deepwalk.DeepWalk)** from Perozzi *et al.*: [DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652) (KDD 2014)

* **[Node2Vec](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.node_embedding.neighbourhood.node2vec.Node2Vec)** from Grover *et al.*: [node2vec: Scalable Feature Learning for Networks](https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf) (KDD 2016)

* **[NMF-ADMM](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.node_embedding.neighbourhood.nmfadmm.NMFADMM)** from Sun and Févotte: [Alternating Direction Method of Multipliers for Non-Negative Matrix Factorization with the Beta-Divergence](http://statweb.stanford.edu/~dlsun/papers/nmf_admm.pdf) (ICASSP 2014)

* **[Laplacian Eigenmaps](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.node_embedding.neighbourhood.laplacianeigenmaps.LaplacianEigenmaps)** from Belkin and Niyogi: [Laplacian Eigenmaps and Spectral Techniques for Embedding and Clustering](https://papers.nips.cc/paper/1961-laplacian-eigenmaps-and-spectral-techniques-for-embedding-and-clustering) (NIPS 2001)

**Structural Node Level Embedding**

* **[GraphWave](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.node_embedding.structural.graphwave.GraphWave)** from Donnat *et al.*: [Learning Structural Node Embeddings via Diffusion Wavelets](https://arxiv.org/abs/1710.10321) (KDD 2018)

* **[Role2Vec](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.node_embedding.structural.role2vec.Role2vec)** from Ahmed *et al.*: [Learning Role-based Graph Embeddings](https://arxiv.org/abs/1802.02896) (IJCAI StarAI 2018)

**Attributed Node Level Embedding**

* **[FEATHER-N](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.node_embedding.attributed.feathernode.FeatherNode)** from Rozemberczki *et al.*: [Characteristic Functions on Graphs: Birds of a Feather, from Statistical Descriptors to Parametric Models](https://arxiv.org/abs/2005.07959) (CIKM 2020)

* **[AE](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.node_embedding.attributed.ae.AE)** from Rozemberczki *et al.*: [Multi-Scale Attributed Node Embedding](https://arxiv.org/abs/1909.13021) (Arxiv 2019)

* **[MUSAE](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.node_embedding.attributed.musae.MUSAE)** from Rozemberczki *et al.*: [Multi-Scale Attributed Node Embedding](https://arxiv.org/abs/1909.13021) (Arxiv 2019)

* **[FSCNMF](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.node_embedding.attributed.fscnmf.FSCNMF)** from Bandyopadhyay *et al.*: [Fusing Structure and Content via Non-negative Matrix Factorization for Embedding Information Networks](https://arxiv.org/pdf/1804.05313.pdf) (ArXiV 2018)

* **[SINE](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.node_embedding.attributed.sine.SINE)** from Zhang *et al.*: [SINE: Scalable Incomplete Network Embedding](https://arxiv.org/pdf/1810.06768.pdf) (ICDM 2018)

* **[BANE](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.node_embedding.attributed.bane.BANE)** from Yang *et al.*: [Binarized Attributed Network Embedding](https://ieeexplore.ieee.org/document/8626170) (ICDM 2018)

* **[TENE](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.node_embedding.attributed.tene.TENE)** from Yang *et al.*: [Enhanced Network Embedding with Text Information](https://ieeexplore.ieee.org/document/8545577) (ICPR 2018)

* **[ASNE](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.node_embedding.attributed.asne.ASNE)** from Liao *et al.*: [Attributed Social Network Embedding](https://arxiv.org/abs/1705.04969) (TKDE 2018)

* **[TADW](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.node_embedding.attributed.tadw.TADW)** from Yang *et al.*: [Network Representation Learning with Rich Text Information](https://www.ijcai.org/Proceedings/15/Papers/299.pdf) (IJCAI 2015)

**Meta Node Embedding**

* **[NEU](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.node_embedding.meta.neu.NEU)** from Yang *et al.*: [Fast Network Embedding Enhancement via High Order Proximity Approximation](https://www.ijcai.org/Proceedings/2017/0544.pdf) (IJCAI 2017)

**Graph Level Embedding**

* **[FEATHER-G](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.graph_embedding.feathergraph.FeatherGraph)** from Rozemberczki *et al.*: [Characteristic Functions on Graphs: Birds of a Feather, from Statistical Descriptors to Parametric Models](https://arxiv.org/abs/2005.07959) (CIKM 2020)

* **[IGE](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.graph_embedding.ige.IGE)** from Galland *et al.*: [Invariant Embedding for Graph Classification](https://graphreason.github.io/papers/16.pdf) (ICML 2019 LRGSD Workshop)

* **[LDP](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.graph_embedding.ldp.LDP)** from Cai *et al.*: [A Simple Yet Effective Baseline for Non-Attributed Graph Classification](https://arxiv.org/abs/1811.03508) (ICLR 2019)

* **[GeoScattering](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.graph_embedding.geoscattering.GeoScattering)** from Gao *et al.*: [Geometric Scattering for Graph Data Analysis](http://proceedings.mlr.press/v97/gao19e.html) (ICML 2019)

* **[GL2Vec](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.graph_embedding.gl2vec.GL2Vec)** from Chen and Koga: [GL2Vec: Graph Embedding Enriched by Line Graphs with Edge Features](https://link.springer.com/chapter/10.1007/978-3-030-36718-3_1) (ICONIP 2019)

* **[NetLSD](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.graph_embedding.netlsd.NetLSD)** from Tsitsulin *et al.*: [NetLSD: Hearing the Shape of a Graph](https://arxiv.org/abs/1805.10712) (KDD 2018)

* **[SF](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.graph_embedding.sf.SF)** from de Lara and Pineau: [A Simple Baseline Algorithm for Graph Classification](https://arxiv.org/abs/1810.09155) (NeurIPS RRL Workshop 2018) 

* **[FGSD](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.graph_embedding.fgsd.FGSD)** from Verma and Zhang: [Hunt For The Unique, Stable, Sparse And Fast Feature Learning On Graphs](https://papers.nips.cc/paper/6614-hunt-for-the-unique-stable-sparse-and-fast-feature-learning-on-graphs.pdf) (NeurIPS 2017)

* **[Graph2Vec](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.graph_embedding.graph2vec.Graph2Vec)** from Narayanan *et al.*: [Graph2Vec: Learning Distributed Representations of Graphs](https://arxiv.org/abs/1707.05005) (MLGWorkshop 2017)


Head over to our [documentation](https://karateclub.readthedocs.io) to find out more about installation and data handling, a full list of implemented methods, and datasets. For a quick start, check out our [examples](https://github.com/benedekrozemberczki/karateclub/tree/master/examples.py).

If you notice anything unexpected, please open an [issue](https://github.com/benedekrozemberczki/karateclub/issues) and let us know. If you are missing a specific method, feel free to open a [feature request](https://github.com/benedekrozemberczki/karateclub/issues).
We are motivated to constantly make Karate Club even better.


--------------------------------------------------------------------------------

**Installation**

Karate Club can be installed with the following pip command.

```sh
$ pip install karateclub
```

As we create new releases frequently, upgrading the package casually might be beneficial.

```sh
$ pip install karateclub --upgrade
```

--------------------------------------------------------------------------------

**Running examples**

As part of the documentation we provide a number of use cases to show how the clusterings and embeddings can be utilized for downstream learning. These can accessed [here](https://karateclub.readthedocs.io/en/latest/notes/introduction.html) with detailed explanations.


Besides the case studies we provide synthetic examples for each model. These can be tried out by running the example scripts. In order to run one of the examples, the Graph2Vec snippet:

```sh
$ cd examples/whole_graph_embedding/
$ python graph2vec_example.py
```

--------------------------------------------------------------------------------

**Running tests**

```sh
$ python setup.py test
```

--------------------------------------------------------------------------------

**License**

- [GNU General Public License v3.0](https://github.com/benedekrozemberczki/karateclub/blob/master/LICENSE)
