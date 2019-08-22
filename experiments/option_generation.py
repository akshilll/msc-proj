import numpy as np
from barl_simpleoptions.option import PrimitiveOption, SubgoalOption
from heart_peg_solitaire import heart_peg_state
from subgoal_extraction import string_to_list
import json


# Primitive Options
def generate_primitive_options(graph_path="graphs/heart_peg_solitaire_graph.gexf"):
    
    graph = nx.read_gexf(graph_path)
    primitive_options = []
    for node in nx.nodes(graph):
        s = heart_peg_state(string_to_list(node))
        




