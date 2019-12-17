 ![GitHub stars](https://img.shields.io/github/stars/benedekrozemberczki/karateclub.svg?style=plastic) ![GitHub forks](https://img.shields.io/github/forks/benedekrozemberczki/karateclub.svg?color=blue&style=plastic) ![License](https://img.shields.io/github/license/benedekrozemberczki/karateclub.svg?color=blue&style=plastic)

<p align="center">
  <img width="90%" src="https://github.com/benedekrozemberczki/karateclub/blob/master/karatelogo.jpg?sanitize=true" />
</p>

--------------------------------------------------------------------------------


**[Documentation](https://karateclub.readthedocs.io/)** | **[Paper](https://arxiv.org/abs/1802.03997)** 

*karateclub* is an unsupervised machine learning extension library for [NetworkX](https://networkx.github.io/).


Karate Club consists of state-of-the-art methods to do unsupervised learning on graph structured data. To put it simply it is a Swiss Army knife for small-scale graph mining research. First, it provides network embedding techniques at the node and graph level. Second, it includes a variety of overlapping and non-overlapping commmunity detection methods. The package also includes methods that can deal with bipartite, temporal and heterogeneous graphs. Implemented methods cover a wide range of network science ([NetSci](https://netscisociety.net/home), [Complenet](https://complenet.weebly.com/)), data mining ([ICDM](http://icdm2019.bigke.org/), [CIKM](http://www.cikm2019.net/), [KDD](https://www.kdd.org/kdd2020/)), artificial intelligence ([AAAI](http://www.aaai.org/Conferences/conferences.php), [IJCAI](https://www.ijcai.org/)) and machine learning ([NeurIPS](https://nips.cc/), [ICML](https://icml.cc/), [ICLR](https://iclr.cc/)) conferences and pieces from prominent journals.  

--------------------------------------------------------------------------------

Karate Club makes the use of modern community detection tecniques quite easy (see [here](link_here) for the accompanying tutorial).
For example, this is all it takes to use on a Watts-Strogatz graph [Ego-splitting](https://www.eecs.yorku.ca/course_archive/2017-18/F/6412/reading/kdd17p145.pdf):

```python
import networkx as nx
from karateclub import EgoNetSplitter

g = nx.newman_watts_strogatz_graph(1000, 20, 0.05)

splitter = EgoNetSplitter(1.0)

splitter.fit(g)

print(splitter.overlapping_partitions)
```

In detail, the following methods are currently implemented.

**Community Detection**

* **[Ego-Splitting](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.ego_splitter.EgoNetSplitter)** from Epasto *et al.*: [Ego-splitting Framework: from Non-Overlapping to Overlapping Clusters](https://www.eecs.yorku.ca/course_archive/2017-18/F/6412/reading/kdd17p145.pdf) (KDD 2017)

* **[EdMot](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.edmot.EdMot)** from Li *et al.*: [EdMot: An Edge Enhancement Approach for Motif-aware Community Detection](https://arxiv.org/abs/1906.04560) (KDD 2019)

* **[DANMF](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.danmf.DANMF)** from Ye *et al.*: [Deep Autoencoder-like Nonnegative Matrix Factorization for Community Detection](https://github.com/benedekrozemberczki/DANMF/blob/master/18DANMF.pdf) (CIKM 2018)

* **[M-NMF](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.mnmf.M_NMF)** from Wang *et al.*: [Community Preserving Network Embedding](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14589) (AAAI 2017)

* **[Label Propagation](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.label_propagation.LabelPropagation)** from Raghavan *et al.*: [Near Linear Time Algorithm to Detect Community Structures in Large-Scale Networks](https://arxiv.org/abs/0709.2938) (Physics Review E 2007)

--------------------------------------------------------------------------------

Head over to our [documentation](https://karateclub.readthedocs.io) to find out more about installation and data handling, a full list of implemented methods, and datasets.
For a quick start, check out our [examples](https://github.com/benedekrozemberczki/karateclub/tree/master/examples.py).

If you notice anything unexpected, please open an [issue](https://github.com/benedekrozemberczki/karateclub/issues) and let us know.
If you are missing a specific method, feel free to open a [feature request](https://github.com/benedekrozemberczki/karateclub/issues).
We are motivated to constantly make Karate Club even better.

## Installation

```sh
$ pip install karateclub
```


## Running examples

```
$ python examples.py
```
