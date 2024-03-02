"""SINr illustrative example.
Nodes in both cliques will get the same embedding vectors, except for the one connected to the path.
Nodes in the paths are in distinct communities with sufficient gamma, and get thus distinct vectors.
"""

import networkx as nx
from karateclub.node_embedding.structural import SINr
import matplotlib.pyplot as plt

def embed_and_plot(g, gamma, ax):
    model = SINr(gamma=gamma)
    model.fit(g)
    X = model.get_embedding()


    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_2 = pca.fit_transform(X)

    ax.scatter(X_2[:,0], X_2[:,1])
    for idx, x in enumerate(X_2):
        ax.annotate(idx, (x[0], x[1]))
    
    

g = nx.barbell_graph(4,8)
fig, axs = plt.subplots(3)

nx.draw_kamada_kawai(g, with_labels=True, ax=axs[0])

embed_and_plot(g,0.5, axs[1])
embed_and_plot(g,10, axs[2])

plt.show()