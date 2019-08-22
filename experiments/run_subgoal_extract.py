from subgoal_extraction import extract_subgoals, string_to_list
from heart_peg_solitaire import heart_peg_state


graph_path = "graphs/heart_peg_solitaire_graph.gexf"
centralities = ["betweenness", "closeness", "degree", "eigenvector", "katz", "load"]
for c in centralities:
    extract_subgoals(graph_path, centrality=c)