import numpy as np
from copy import deepcopy
import networkx as nx 
from centrality_experiments.environments.heart_peg_solitaire import heart_peg_env, heart_peg_state
from typing import List
from barl_simpleoptions.state import State


# Josh's code, slightly optimised by me found https://github.com/Ueva/BaRL-SimpleOptions/blob/master/barl_simpleoptions/state.py
def generate_interaction_graph(initial_states : List['State']) :
	"""
	Generates a directed state-transition graph for this environment.
	"""
	states = []
	current_successor_states = []

	# Add initial states to current successor list.
	current_successor_states = deepcopy(initial_states)

	# While we have new successor states to process.
	while (len(current_successor_states) != 0) :
		# Add each current new successor to our list of processed states.
		
		next_successor_states = []
		for successor_state in current_successor_states :
			if (successor_state not in states) :
				states.append(successor_state)

				# Add this state's successors to the successor list.
				if (not successor_state.is_terminal_state()) : 
					for new_successor_state in successor_state.get_successors() :
						next_successor_states.append(new_successor_state)

		current_successor_states = deepcopy(next_successor_states)
		
	print(len(states))
	# Create graph from state list.
	interaction_graph = nx.DiGraph()
	for state in states :
		
		# Add state to interaction graph.
		interaction_graph.add_node(str(state))

		for successor_state in state.get_successors() :
			interaction_graph.add_node(str(successor_state))
			interaction_graph.add_edge(str(state), str(successor_state))
	
	return interaction_graph

		
def write_graph(file_path):
	""" Wrapper function of generate_interaction_graph
        Outputs interaction graph of heart_peg_solitaire to file

        Arguments:
            file_path -- String containing file path to output file where graph is written
    
    """
	start_state = [1] * 16
	start_state[9] = 0
	init_state = heart_peg_state(state = start_state)	

	interaction_graph = generate_interaction_graph(initial_states = [init_state])
	
	
	nx.write_gexf(interaction_graph, file_path)

def add_centrality_attr(centrality, file_path="./centrality_experiments/graphs/heart_peg_solitaire_graph.gexf"):
	graph = nx.read_gexf(file_path)
	
	assert type(centrality) == str

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
		metric_values = nx.algorithms.link_analysis.pagerank_alg.pagerank(graph)
	
	elif centrality == "out_degree":
		metric_values = nx.algorithms.centrality.out_degree_centrality(graph)

	nx.set_node_attributes(graph, metric_values, centrality)
	
	nx.write_gexf(graph, file_path)
	
	return True

def add_all_graph_attrs(file_path="./centrality_experiments/graphs/heart_peg_solitaire_graph.gexf"):
	centralities = ["betweenness", "closeness", "degree", "eigenvector", "katz", "load", "pagerank", "out_degree"]

	for c in centralities:
		add_centrality_attr(c, file_path)

