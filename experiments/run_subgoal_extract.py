from subgoal_extraction import extract_subgoals, string_to_list
from heart_peg_solitaire import heart_peg_state

# Run subgoal extraction for each centrality from graph
graph_path = "graphs/heart_peg_solitaire_graph.gexf"
centralities = ["betweenness", "closeness", "degree", "eigenvector", "katz", "load", "pagerank"]
for c in centralities:
    extract_subgoals(graph_path, centrality=c)