import sys
from centrality_experiments.environments.heart_peg_solitaire import heart_peg_env, heart_peg_state
from centrality_experiments.src.graph_generation import *
from networkx.drawing.nx_pydot import write_dot

# Generate graph
file_path = "./centrality_experiments/graphs/heart_peg_solitaire.gexf"
write_graph(file_path=file_path)
add_all_graph_attrs(file_path=file_path)
gexf_to_dot(file_path)


