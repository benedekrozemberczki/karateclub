"""GraphSetReader example."""

from karateclub.dataset import GraphSetReader

reader = GraphSetReader("reddit10k")

graphs = reader.get_graphs()
y = reader.get_target()

