"""Example runs with Karate Club."""

import networkx as nx
from karateclub import EgoNetSplitter, EdMot

g = nx.newman_watts_strogatz_graph(1000, 20, 0.05)

#------------------------------------
# Splitter example
#------------------------------------

splitter = EgoNetSplitter(1.0)

splitter.fit(g)

print(splitter.get_memberships())

#------------------------------------
# Edmot example
#------------------------------------

edmot = EdMot(2, 0.5)

edmot.fit(g)

print(edmot.get_memberships())
