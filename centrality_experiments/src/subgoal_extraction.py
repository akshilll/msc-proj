import networkx as nx
from heapq import nlargest
from centrality_experiments.environments.heart_peg_solitaire import heart_peg_state
import numpy as np
import re
import json
import pickle


def extract_subgoals(path, out_dir, centrality="betweenness", n_subgoals=5):
    
    out_path = out_dir + "/subgoals/" +  centrality + ".txt"

    # Read in graph
    graph = nx.read_gexf(graph_path)

    # Choose centrality measure
    if centrality == "betweenness":
        metric_values = nx.algorithms.centrality.betweenness_centrality(graph)
    
    elif centrality == "closeness":
        metric_values = nx.algorithms.centrality.closeness_centrality(graph.reverse())

    elif centrality == "degree":
        metric_values = nx.algorithms.centrality.degree_centrality(graph)

    elif centrality == "eigenvector":
        metric_values = nx.algorithms.centrality.eigenvector_centrality(graph.reverse(), max_iter=10000)

    elif centrality == "katz":
        metric_values = nx.algorithms.centrality.katz_centrality(graph.reverse(), max_iter=10000)

    elif centrality == "load":
        metric_values = nx.algorithms.centrality.load_centrality(graph)

    elif centrality == "pagerank":
        metric_values = nx.algorithms.link_analysis.pagerank_alg.pagerank(graph.reverse())

    assert len(metric_values) == len(graph)

    # Get local maxima 
    subgoals = []

    for node in nx.nodes(graph):
        # Get neighbours of node
        neighbours = list(nx.all_neighbors(graph, node))
        
        assert len(neighbours) > 0

        # get centrality values for the neighbourhood
        nbhd_metrics = [metric_values[n] for n in neighbours]
        
        # If it's the local max then it's a subgoal!
        centrality_val = metric_values[node]    
        if centrality_val >= max(nbhd_metrics):
            subgoals.append(node)
    
    print(len(subgoals))

    # We only want a certain number of subgoals so get the best ones
    if len(subgoals) > n_subgoals:
        subgoal_dict = {s: metric_values[s] for s in subgoals}
        
        # Get subgoals with highest value
        subgoals = [key for key in sorted(subgoal_dict, key=subgoal_dict.get, reverse=True)[:n_subgoals]]


    print(centrality, subgoals)


    # Write subgoals to a txt file
    with open(out_path, "wb+") as f:
        pickle.dump(subgoals, f)
    
    return subgoals


if __name__=='__main__':
    # Run subgoal extraction for each centrality from graph
    graph_path = "./centrality_experiments/graphs/heart_peg_solitaire_without_.gexf"
    out_dir = "./centrality_experiments/"
    centralities = ["betweenness", "closeness", "degree", "eigenvector", "katz", "load", "pagerank"]
    for c in centralities:
        extract_subgoals(graph_path, out_dir, centrality=c)