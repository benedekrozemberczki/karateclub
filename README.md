
 ![Version](https://badge.fury.io/py/karateclub.svg?style=plastic)
 ![GitHub stars](https://img.shields.io/github/stars/benedekrozemberczki/karateclub.svg?style=plastic) ![GitHub forks](https://img.shields.io/github/forks/benedekrozemberczki/karateclub.svg?color=blue&style=plastic) ![License](https://img.shields.io/github/license/benedekrozemberczki/karateclub.svg?color=blue&style=plastic) [![PyPI download month](https://img.shields.io/pypi/dm/karateclub.svg?color=blue&style=plastic)](https://pypi.python.org/pypi/karateclub/)

<p align="center">
  <img width="90%" src="https://github.com/benedekrozemberczki/karateclub/blob/master/karatelogo.jpg?sanitize=true" />
</p>

--------------------------------------------------------------------------------


**[Documentation](https://karateclub.readthedocs.io/)**

*Karate Club* is an unsupervised machine learning extension library for [NetworkX](https://networkx.github.io/).


*Karate Club* consists of state-of-the-art methods to do unsupervised learning on graph structured data. To put it simply it is a Swiss Army knife for small-scale graph mining research. First, it provides network embedding techniques at the node and graph level. Second, it includes a variety of overlapping and non-overlapping community detection methods. Implemented methods cover a wide range of network science ([NetSci](https://netscisociety.net/home), [Complenet](https://complenet.weebly.com/)), data mining ([ICDM](http://icdm2019.bigke.org/), [CIKM](http://www.cikm2019.net/), [KDD](https://www.kdd.org/kdd2020/)), artificial intelligence ([AAAI](http://www.aaai.org/Conferences/conferences.php), [IJCAI](https://www.ijcai.org/)) and machine learning ([NeurIPS](https://nips.cc/), [ICML](https://icml.cc/), [ICLR](https://iclr.cc/)) conferences, workshops, and pieces from prominent journals.  

--------------------------------------------------------------------------------

**Citing**

If you find *Karate Club* useful in your research, please consider citing the following paper:

```bibtex
>@misc{rozemberczki2020karateclub,    
       title = {Karate Club: An open-source Python framework for unsupervised learning on graphs},   
       author = {Benedek Rozemberczki and Rik Sarkar},   
       year = {2020},   
       publisher = {GitHub},   
       journal = {GitHub repository},   
       howpublished = {\url{https://github.com/benedekrozemberczki/karateclub}}   
       }
```
--------------------------------------------------------------------------------

**A simple example**

*Karate Club* makes the use of modern community detection tecniques quite easy (see [here](https://karateclub.readthedocs.io/en/latest/notes/introduction.html) for the accompanying tutorial).
For example, this is all it takes to use on a Watts-Strogatz graph [Ego-splitting](https://www.eecs.yorku.ca/course_archive/2017-18/F/6412/reading/kdd17p145.pdf):

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

* **[BigClam](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.community_detection.overlapping.bigclam.BigClam)** from Yang and Leskovec: [Overlapping Community Detection at Scale:A Nonnegative Matrix Factorization Approach](http://infolab.stanford.edu/~crucis/pubs/paper-nmfagm.pdf) (WSDM 2013)

* **[SymmNMF](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.community_detection.overlapping.symmnmf.SymmNMF)** from Kuang *et al.*: [Symmetric Nonnegative Matrix Factorization for Graph Clustering](https://www.cc.gatech.edu/~hpark/papers/DaDingParkSDM12.pdf) (SDM 2012)

**Non-Overlapping Community Detection**

* **[EdMot](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.community_detection.non_overlapping.edmot.EdMot)** from Li *et al.*: [EdMot: An Edge Enhancement Approach for Motif-aware Community Detection](https://arxiv.org/abs/1906.04560) (KDD 2019)

* **[SCD](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.community_detection.non_overlapping.scd.SCD)** from Prat-Perez *et al.*: [High Quality, Scalable and Parallel Community Detectionfor Large Real Graphs](http://wwwconference.org/proceedings/www2014/proceedings/p225.pdf) (WWW 2014)

* **[Label Propagation](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.community_detection.non_overlapping.label_propagation.LabelPropagation)** from Raghavan *et al.*: [Near Linear Time Algorithm to Detect Community Structures in Large-Scale Networks](https://arxiv.org/abs/0709.2938) (Physics Review E 2007)

**Neighbourhood-Based Node Level Embedding**

* **[BoostNE](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.node_embedding.neighbourhood.boostne.BoostNE)** from Li *et al.*: [Multi-Level Network Embedding with Boosted Low-Rank Matrix Approximation](https://arxiv.org/abs/1808.08627) (ASONAM 2019)

* **[Diff2Vec](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.node_embedding.neighbourhood.diff2vec.Diff2Vec)** from Rozemberczki and Sarkar: [Fast Sequence Based Embedding with Diffusion Graphs](https://arxiv.org/abs/2001.07463) (CompleNet 2018)

* **[NetMF](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.node_embedding.neighbourhood.netmf.NetMF)** from Qui *et al.*: [Network Embedding as Matrix Factorization: Unifying DeepWalk, LINE, PTE, and Node2Vec](https://keg.cs.tsinghua.edu.cn/jietang/publications/WSDM18-Qiu-et-al-NetMF-network-embedding.pdf) (WSDM 2018)

* **[Walklets](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.node_embedding.neighbourhood.walklets.Walklets)** from Perozzi *et al.*: [Don't Walk, Skip! Online Learning of Multi-scale Network Embeddings](https://arxiv.org/abs/1605.02115) (ASONAM 2017)

* **[GraRep](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.node_embedding.neighbourhood.grarep.GraRep)** from Cao *et al.*: [GraRep: Learning Graph Representations with Global Structural Information](https://dl.acm.org/citation.cfm?id=2806512) (CIKM 2015)

* **[DeepWalk](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.node_embedding.neighbourhood.deepwalk.DeepWalk)** from Perozzi *et al.*: [DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652) (KDD 2014)

* **[NMF-ADMM](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.node_embedding.neighbourhood.nmfadmm.NMFADMM)** from Sun and Févotte: [Alternating Direction Method of Multipliers for Non-Negative Matrix Factorization with the Beta-Divergence](http://statweb.stanford.edu/~dlsun/papers/nmf_admm.pdf) (ICASSP 2014)

**Structural Node Level Embedding**

* **[GraphWave](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.node_embedding.structural.graphwave.GraphWave)** from Donnat *et al.*: [Learning Structural Node Embeddings via Diffusion Wavelets](https://arxiv.org/abs/1710.10321) (KDD 2018)

* **[Role2Vec](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.node_embedding.structural.role2vec.Role2vec)** from Ahmed *et al.*: [Learning Role-based Graph Embeddings](https://arxiv.org/abs/1802.02896) (IJCAI StarAI 2018)

**Attributed Node Level Embedding**

* **[MUSAE](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.node_embedding.attributed.musae.MUSAE)** from Rozemberczki *et al.*: [Multi-Scale Attributed Node Embedding](https://arxiv.org/abs/1909.13021) (Arxiv 2019)

* **[FSCNMF](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.node_embedding.attributed.fscnmf.FSCNMF)** from Bandyopadhyay *et al.*: [Fusing Structure and Content via Non-negative Matrix Factorization for Embedding Information Networks](https://arxiv.org/pdf/1804.05313.pdf) (ArXiV 2018)

* **[SINE](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.node_embedding.attributed.sine.SINE)** from Zhang *et al.*: [SINE: Scalable Incomplete Network Embedding](https://arxiv.org/pdf/1810.06768.pdf) (ICDM 2018)

* **[BANE](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.node_embedding.attributed.bane.BANE)** from Yang *et al.*: [Binarized Attributed Network Embedding](https://ieeexplore.ieee.org/document/8626170) (ICDM 2018)

* **[TENE](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.node_embedding.attributed.tene.TENE)** from Yang *et al.*: [Enhanced Network Embedding with Text Information](https://ieeexplore.ieee.org/document/8545577) (ICPR 2018)

* **[TADW](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.node_embedding.attributed.tadw.TADW)** from Yang *et al.*: [Network Representation Learning with Rich Text Information](https://www.ijcai.org/Proceedings/15/Papers/299.pdf) (IJCAI 2015)

**Graph Level Embedding**

* **[GL2Vec](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.graph_embedding.gl2vec.GL2Vec)** from Chen and Koga: [GL2Vec: Graph Embedding Enriched by Line Graphs with Edge Features](https://link.springer.com/chapter/10.1007/978-3-030-36718-3_1) (ICONIP 2019)

* **[SF](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.graph_embedding.sf.SF)** from de Lara and Pineau: [A Simple Baseline Algorithm for Graph Classification](https://arxiv.org/abs/1810.09155) (NeurIPS RRL Workshop 2018) 

* **[FGSD](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.graph_embedding.fgsd.FGSD)** from Verma and Zhang: [Hunt For The Unique, Stable, Sparse And Fast Feature Learning On Graphs](https://papers.nips.cc/paper/6614-hunt-for-the-unique-stable-sparse-and-fast-feature-learning-on-graphs.pdf) (NeurIPS 2017)

* **[Graph2Vec](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.graph_embedding.graph2vec.Graph2Vec)** from Narayanan *et al.*: [Graph2Vec: Learning Distributed Representations of Graphs](https://arxiv.org/abs/1707.05005) (MLGWorkshop 2017)


Head over to our [documentation](https://karateclub.readthedocs.io) to find out more about installation and data handling, a full list of implemented methods, and datasets.
For a quick start, check out our [examples](https://github.com/benedekrozemberczki/karateclub/tree/master/examples.py).

If you notice anything unexpected, please open an [issue](https://github.com/benedekrozemberczki/karateclub/issues) and let us know.
If you are missing a specific method, feel free to open a [feature request](https://github.com/benedekrozemberczki/karateclub/issues).
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


Besides the case studies we provide synthetic examples for each model. These can be tried out by running the examples script.

```sh
$ python examples.py
```
