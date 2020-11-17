import sys
from centrality_experiments.environments.heart_peg_solitaire import heart_peg_env, heart_peg_state
from centrality_experiments.src.graph_generation import generate_interaction_graph, write_graph, add_all_graph_attrs


# Generate graph
file_path = "./centrality_experiments/graphs/heart_peg_solitaire.gexf"
write_graph(file_path=file_path)
add_all_graph_attrs(file_path=file_path)

