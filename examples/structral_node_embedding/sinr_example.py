"""SINr illustrative example.
Nodes in both cliques (barbell graph) will get the same embedding vectors,
except for those connected to the path.
Nodes in the path are in distinct communities with a high-enough gamma,
and will thus get distinct vectors.
"""

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from sklearn.decomposition import PCA
from karateclub.node_embedding.structural import SINr


def embed_and_plot(graph: nx.Graph, gamma: int, ax: Axes):
    """Embed the graph using SINr and plot the 2D PCA projection.

    Args:
        graph (nx.Graph): The graph to embed.
        gamma (int): The modularity multi-resolution parameter.
        ax (Axes): The matplotlib axis to plot the graph on.
    """
    model = SINr(gamma=gamma)
    model.fit(graph)
    embedding = model.get_embedding()

    pca_embedding = PCA(n_components=2).fit_transform(embedding)

    ax.scatter(pca_embedding[:, 0], pca_embedding[:, 1])
    for idx, x in enumerate(pca_embedding):
        ax.annotate(idx, (x[0], x[1]))


if __name__ == "__main__":

    barbell = nx.barbell_graph(4, 8)
    fig, axs = plt.subplots(3)

    nx.draw_kamada_kawai(barbell, with_labels=True, ax=axs[0])

    embed_and_plot(barbell, 0.5, axs[1])
    embed_and_plot(barbell, 10, axs[2])

    plt.show()
