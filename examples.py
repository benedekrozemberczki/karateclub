"""Example runs with Karate Club."""

import networkx as nx

import community
import numpy as np

from karateclub.community_detection.overlapping import EgoNetSplitter, NNSED, DANMF, MNMF, BigClam
from karateclub.community_detection.non_overlapping import EdMot, LabelPropagation
from karateclub.node_embedding.neighbourhood import GraRep, DeepWalk, Walklets
from karateclub.node_embedding.structural import GraphWave
from karateclub.node_embedding.attributed import BANE


#-----------------------------------
# BANE example
#-----------------------------------

g = nx.newman_watts_strogatz_graph(100, 20, 0.05)

x = np.random.uniform(0,1,(100,2000))
model = BANE()

model.fit(g, x)

#------------------------------------
# BigClam example
#------------------------------------

g = nx.newman_watts_strogatz_graph(100, 20, 0.05)

model = BigClam()

model.fit(g)

membership = model.get_memberships()

print(community.modularity(membership, g))

quit()

#------------------------------------
# Walklets example
#------------------------------------



g = nx.newman_watts_strogatz_graph(100, 20, 0.05)


model = Walklets()

model.fit(g)

emb = model.get_embedding()

print(emb.shape)


#------------------------------------
# DeepWalk example
#------------------------------------


g = nx.newman_watts_strogatz_graph(100, 20, 0.05)


model = DeepWalk()

model.fit(g)

print(model.get_embedding())


#------------------------------------
# Splitter example
#------------------------------------

g = nx.newman_watts_strogatz_graph(100, 20, 0.05)


model = EgoNetSplitter(1.0)

model.fit(g)

print(model.get_memberships())

#------------------------------------
# Edmot example
#------------------------------------

g = nx.newman_watts_strogatz_graph(100, 10, 0.9)

model = EdMot(3, 0.5)

model.fit(g)

print(model.get_memberships())


#------------------------------------
# DANMF example
#------------------------------------

g = nx.newman_watts_strogatz_graph(100, 10, 0.02)

model = DANMF()

model.fit(g)

print(model.get_memberships())
print(model.get_embedding())

#------------------------------------
# M-NMF example
#------------------------------------

g = nx.newman_watts_strogatz_graph(100, 10, 0.02)

model = MNMF()

model.fit(g)

print(model.get_memberships())
print(model.get_embedding())
print(model.get_cluster_centers())

#------------------------------------
# Label Propagation example
#------------------------------------


g = nx.newman_watts_strogatz_graph(100, 10, 0.02)
model = LabelPropagation()

model.fit(g)

print(model.get_memberships())



#------------------------------------
# GraRep example
#------------------------------------

g = nx.newman_watts_strogatz_graph(100, 10, 0.02)

model = GraRep()

model.fit(g)
embedding = model.get_embedding()

print(embedding)

#------------------------------------
# GraphWave example
#------------------------------------


g = nx.newman_watts_strogatz_graph(100, 10, 0.02)

model = GraphWave()

model.fit(g)
embedding = model.get_embedding()

print(embedding)


#------------------------------------
# NNSED example
#------------------------------------


g = nx.newman_watts_strogatz_graph(100, 10, 0.02)

model = NNSED()

model.fit(g)
embedding = model.get_embedding()

print(embedding)

memberships = model.get_memberships()
print(memberships)
