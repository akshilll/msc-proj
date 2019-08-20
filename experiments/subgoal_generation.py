import networkx as nx

graph_path = "./experiments/graphs/heart_peg_solitaire_graph.gexf"

graph = nx.read_gexf(graph_path)

def subgoal_extraction(graph, centrality="betweenness"):
    for n in graph.nodes:
        print(nx.algorithms.centrality.betweenness_centrality(n))