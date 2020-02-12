"""Example runs with Karate Club."""

import networkx as nx

import community
import numpy as np

from karateclub.node_embedding.neighbourhood import GraRep, DeepWalk, Walklets, NMFADMM, Diff2Vec, BoostNE, NetMF
from karateclub.community_detection.overlapping import EgoNetSplitter, NNSED, DANMF, MNMF, BigClam
from karateclub.community_detection.non_overlapping import EdMot, LabelPropagation
from karateclub.graph_embedding import Graph2Vec, FGSD, GL2Vec, SF
from karateclub.node_embedding.attributed import BANE, TENE, TADW, FSCNMF
from karateclub.node_embedding.structural import GraphWave, Role2Vec
from karateclub.dataset import GraphReader, GraphSetReader

#-----------------------------------
# Role2vec example
#-----------------------------------

g = nx.newman_watts_strogatz_graph(100, 20, 0.05)

model = Role2Vec()

model.fit(g)
model.get_embedding()

#-----------------------------------
# SF example
#-----------------------------------

graphs = [nx.newman_watts_strogatz_graph(50, 5, 0.3) for _ in range(100)]

model = SF()

model.fit(graphs)
model.get_embedding()

#-----------------------------------
# FSCNMF example
#-----------------------------------

g = nx.newman_watts_strogatz_graph(200, 20, 0.05)

x = np.random.uniform(0, 1, (200, 200))

model = FSCNMF()

model.fit(g, x)

#-----------------------------------
# TADW example
#-----------------------------------

g = nx.newman_watts_strogatz_graph(200, 20, 0.05)

x = np.random.uniform(0, 1, (200, 200))

model = TADW()

model.fit(g, x)

#-----------------------------------
# GL2Vec example
#-----------------------------------

graphs = [nx.newman_watts_strogatz_graph(50, 5, 0.3) for _ in range(100)]

model = GL2Vec()

model.fit(graphs)
model.get_embedding()

#-----------------------------------
# FGSD example
#-----------------------------------

graphs = [nx.newman_watts_strogatz_graph(50, 5, 0.3) for _ in range(100)]

model = FGSD()

model.fit(graphs)
model.get_embedding()

#-----------------------------------
# NetMF example
#-----------------------------------

g = nx.newman_watts_strogatz_graph(100, 20, 0.05)

model = NetMF()

model.fit(g)
model.get_embedding()

#-----------------------------------
# GraphSet reader example
#-----------------------------------

reader = GraphSetReader("reddit10k")

graphs = reader.get_graphs()
y = reader.get_target()

#-----------------------------------
# Graph reader example
#-----------------------------------

reader = GraphReader("facebook")

graph = reader.get_graph()
target = reader.get_target()

#----------------------------------
# Graph2Vec attributed example
#----------------------------------

graphs = []

for i in range(50):
    graph = nx.newman_watts_strogatz_graph(50, 5, 0.3)
    nx.set_node_attributes(graph, {j: str(j) for j in range(50)}, "feature")
    graphs.append(graph)
model = Graph2Vec(attributed=True)

model.fit(graphs)
model.get_embedding()

#-----------------------------------
# Graph2Vec example
#-----------------------------------

graphs = [nx.newman_watts_strogatz_graph(50, 5, 0.3) for _ in range(1000)]

model = Graph2Vec()

model.fit(graphs)
model.get_embedding()

#-----------------------------------
# BoostNE example
#-----------------------------------

g = nx.newman_watts_strogatz_graph(100, 20, 0.05)

model = BoostNE()

model.fit(g)
model.get_embedding()

#-----------------------------------
# Diff2Vec example
#-----------------------------------

g = nx.newman_watts_strogatz_graph(100, 20, 0.05)

model = Diff2Vec()

model.fit(g)
model.get_embedding()

#-----------------------------------
# NMF ADMM example
#-----------------------------------

g = nx.newman_watts_strogatz_graph(100, 20, 0.05)

model = NMFADMM()

model.fit(g)
model.get_embedding()

#-----------------------------------
# TENE example
#-----------------------------------

g = nx.newman_watts_strogatz_graph(100, 20, 0.05)

t = np.random.uniform(0,1,(100,2000))

tp = nx.newman_watts_strogatz_graph(100, 20, 0.05)

tp = nx.adjacency_matrix(tp)

model = TENE()

model.fit(g, t)
model.get_embedding()

model.fit(g, tp)
model.get_embedding()

#-----------------------------------
# BANE example
#-----------------------------------

g = nx.newman_watts_strogatz_graph(100, 20, 0.05)

x = np.random.uniform(0,1,(100,2000))

p = nx.newman_watts_strogatz_graph(100, 20, 0.05)

x = nx.adjacency_matrix(p)
model = BANE()

model.fit(g, x)

#------------------------------------
# BigClam example
#------------------------------------

g = nx.newman_watts_strogatz_graph(100, 20, 0.05)

model = BigClam()

model.fit(g)

membership = model.get_memberships()

#------------------------------------
# Walklets example
#------------------------------------

g = nx.newman_watts_strogatz_graph(100, 20, 0.05)

model = Walklets()

model.fit(g)

emb = model.get_embedding()

#------------------------------------
# DeepWalk example
#------------------------------------

g = nx.newman_watts_strogatz_graph(100, 20, 0.05)

model = DeepWalk()

model.fit(g)

#------------------------------------
# Splitter example
#------------------------------------

g = nx.newman_watts_strogatz_graph(100, 20, 0.05)

model = EgoNetSplitter(1.0)

model.fit(g)

#------------------------------------
# Edmot example
#------------------------------------

g = nx.newman_watts_strogatz_graph(100, 10, 0.9)

model = EdMot(3, 0.5)

model.fit(g)

#------------------------------------
# DANMF example
#------------------------------------

g = nx.newman_watts_strogatz_graph(100, 10, 0.02)

model = DANMF()

model.fit(g)

#------------------------------------
# M-NMF example
#------------------------------------

g = nx.newman_watts_strogatz_graph(100, 10, 0.02)

model = MNMF()

model.fit(g)

#------------------------------------
# Label Propagation example
#------------------------------------

g = nx.newman_watts_strogatz_graph(100, 10, 0.02)

model = LabelPropagation()

model.fit(g)

#------------------------------------
# GraRep example
#------------------------------------

g = nx.newman_watts_strogatz_graph(100, 10, 0.02)

model = GraRep()

model.fit(g)

embedding = model.get_embedding()

#------------------------------------
# GraphWave example
#------------------------------------

g = nx.newman_watts_strogatz_graph(100, 10, 0.02)

model = GraphWave()

model.fit(g)

embedding = model.get_embedding()

#------------------------------------
# NNSED example
#------------------------------------

g = nx.newman_watts_strogatz_graph(100, 10, 0.02)

model = NNSED()

model.fit(g)

embedding = model.get_embedding()

memberships = model.get_memberships()

