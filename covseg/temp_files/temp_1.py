import networkx as nx
import community as community
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# define the graph
edge = [(1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (4, 6), (5, 6)]
G = nx.Graph()
for e in edge:
    G.add_edge(*e, weight=1)
G.edges[1, 2]['weight'] = 100
G.edges[3, 4]['weight'] = 100
G.edges[5, 6]['weight'] = 100
# retrun partition as freq dict
partition = community.best_partition(G)
mod = community.modularity(partition, G)
print(mod)
print(partition)
print(G.edges[1, 2]['weight'])

# visualization
pos = nx.spring_layout(G)
cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=100, cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.show()

m = 304
q = 1/2/m*((100 - 101 * 101 / 608) * 4 + (100 - 102 * 102 / 608)*2 + (0 - 101 * 101 / 608)*4 + (0 - 102 * 102 / 608)*2)
print(q)