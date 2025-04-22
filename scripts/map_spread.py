# %%
import neural_cone as Ϟnc

# %%
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# %%

num_nodes = 10
G = nx.random_geometric_graph(num_nodes, 0.5)
pos = nx.get_node_attributes(G, "pos")
nx.draw(G, pos, node_size=100, node_color="black", with_labels=True)
plt.show()
# %%
# Get the Laplacian
L = nx.laplacian_matrix(G).toarray()
# zero our L diagonal
# L[np.diag_indices_from(L)] = 0
# set an initial node state
assert L.shape[0] == num_nodes
x0 = np.zeros((L.shape[0], 1))
x0[2] = 1

# %%
timesteps = 10
x = x0
for time in range(timesteps):
    x = np.tanh(np.dot(1 / 10 * L, x))
    print(x)
    nx.draw(G, pos, node_size=100, node_color="black", with_labels=True)
    active_nodes = np.where(np.abs(x) > 0)[0]
    nx.draw(
        G, pos, nodelist=active_nodes, node_size=100, node_color="red", with_labels=True
    )
    plt.show()
