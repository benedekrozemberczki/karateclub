"""Example runs with Karate Club."""


import networkx as nx
from karateclub import EgoNetSplitter, EdMot, DANMF, M_NMF, LabelPropagation
from karateclub import GraRep, GraphWave


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

model = M_NMF()

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
