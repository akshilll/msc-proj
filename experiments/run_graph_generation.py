import sys
from heart_peg_solitaire import heart_peg_env, heart_peg_state
from graph_generation import generate_interaction_graph, write_graph


# Generate graph
file_path = "graphs/heart_peg_solitaire_graph_without_symm.gexf"
write_graph(file_path=file_path)
