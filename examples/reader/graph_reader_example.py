"""GraphReader example."""

from karateclub.dataset import GraphReader

reader = GraphReader("facebook")

graph = reader.get_graph()
target = reader.get_target()

