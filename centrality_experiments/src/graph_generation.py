import numpy as np
from copy import deepcopy
import networkx as nx 
from centrality_experiments.environments.heart_peg_solitaire import heart_peg_env, heart_peg_state
from centrality_experiments.environments.rooms import rooms_state, rooms_environment
from typing import List
from barl_simpleoptions.state import State
from networkx.drawing.nx_pydot import write_dot
import functools
import operator


# Josh's code, slightly optimised by me found https://github.com/Ueva/BaRL-SimpleOptions/blob/master/barl_simpleoptions/state.py
def generate_interaction_graph(initial_states : List['State']) :
	"""
	Generates a directed state-transition graph for this environment.
	"""
	states = []
	current_successor_states = []
	steps_from_initial = []
	steps_tmp = 0

	# Add initial states to current successor list.
	current_successor_states = deepcopy(initial_states)

	# While we have new successor states to process.
	while (len(current_successor_states) != 0) :
		# Add each current new successor to our list of processed states.
		steps_tmp += 1
		next_successor_states = []
		for successor_state in current_successor_states :
			if (successor_state not in states) :
				states.append(successor_state)
				steps_from_initial.append(steps_tmp)

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

# TODO: change this to have a dict of str:function for centrality metrics
def add_centrality_attr(centrality, graph_path):
	graph = nx.read_gexf(graph_path)
	
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
	
	nx.write_gexf(graph, graph_path)
	
	return True

def add_all_graph_attrs(graph_path):
	centralities = ["betweenness", "closeness", "degree", "eigenvector", "katz", "load", "pagerank", "out_degree"]
	
	for c in centralities:
		add_centrality_attr(c, graph_path)


def extract_win_subgraph(graph_path, out_path):
	g = nx.read_gexf(graph_path)
	winning_nodes = np.eye(16, dtype=int).tolist()
	winning_nodes = [str(i) for i in winning_nodes if str(i) in g]
	init_node = '[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]'
	
	subgraph_nodes = [[init_node]]

	for w_node in winning_nodes:
		tmp = nx.all_simple_paths(g, source=init_node, target=w_node)
		subgraph_nodes = subgraph_nodes + list(tmp)

	sg_nodes_unique = np.unique(functools.reduce(operator.iconcat, subgraph_nodes, []))

	subgraph = g.subgraph(sg_nodes_unique)

	nx.write_gexf(subgraph, out_path)




	


if __name__=="__main__":
	
	# Generate graph
	# graph_path = "./centrality_experiments/graphs/heart_peg_solitaire.gexf"
	# start_state = [1] * 16
	# start_state[9] = 0
	# init_state = heart_peg_state(state = start_state)	

	graph_dir = "./centrality_experiments/graphs/"
	layout_dir =  "./centrality_experiments/environments/rooms_layouts/"

	room_envs = ["two_rooms", "four_rooms", "six_rooms"]

	for env in room_envs:
		graph_path = graph_dir + env + ".gexf" 
		layout_path = layout_dir + env + ".txt"
		init_state = rooms_state(layout_path, (2, 2))
	
		interaction_graph = generate_interaction_graph(initial_states = [init_state])

		nx.write_gexf(interaction_graph, graph_path)

		add_all_graph_attrs(graph_path=graph_path)
