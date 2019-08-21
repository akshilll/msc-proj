import networkx as nx

graph_path = "./experiments/graphs/heart_peg_solitaire_graph.gexf"

def extract_subgoals(path=graph_path, centrality="betweenness", n_subgoals=5):

    # Read in graph
    graph = nx.read_gexf(graph_path)

    if centrality == "betweenness":
        metric_values = nx.algorithms.centrality.betweenness_centrality(graph, normalized=False)
    
    elif centrality == "closeness":
        metric_values = nx.algorithms.centrality.closeness_centrality(graph)

    elif centrality == "degree":
        metric_values = nx.algorithms.centrality.degree_centrality(graph)

    elif centrality == "eigenvector":
        metric_values = nx.algorithms.centrality.eigenvector_centrality(graph, max_iter=10000)

    elif centrality == "katz":
        metric_values = nx.algorithms.centrality.katz_centrality(graph, normalized=False)

    elif centrality == "load":
        metric_values = nx.algorithms.centrality.load_centrality(graph, normalized=False)

    assert len(metric_values) == len(graph)


