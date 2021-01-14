import numpy as np
from copy import deepcopy
import functools
import operator

import networkx as nx 
from networkx.algorithms.centrality import *
from networkx.algorithms.link_analysis.pagerank_alg import pagerank
from networkx.drawing.nx_pydot import write_dot

from centrality_experiments.environments.heart_peg_solitaire import heart_peg_env, heart_peg_state
from centrality_experiments.environments.rooms import rooms_state, rooms_environment
from centrality_experiments.environments.peg_solitaire import peg_solitaire_env, peg_solitaire_state

from typing import List
from barl_simpleoptions.state import State



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
	i = 0
	# While we have new successor states to process.
	while (len(current_successor_states) != 0) :
		i += 1
		print(i)
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
def add_centrality_attr(centrality, graph, win_only, winning_nodes=None, init_node=None):
	centrality_fns = {
				"betweenness": betweenness_centrality,
				"closeness": closeness_centrality,
				"out_degree": out_degree_centrality,
				"load": load_centrality,
				"pagerank": pagerank
				}
	
	assert type(centrality) == str
	

	if not win_only:
		if centrality == "eigenvector":
			metric_values = eigenvector_centrality(graph.reverse(), max_iter=10000)
		elif centrality == "katz":
			metric_values = katz_centrality(graph.reverse(), max_iter=10000)
		else:
			centrality_fn = centrality_fns[centrality]
			metric_values = centrality_fn(graph)

	else:
		# Get all shortest paths from init state to goal states
		# For every node on that path, calculate centrality for that node using the subgraph of winning shortest path
		# All other nodes have a value of 0
		
		subgraph_nodes = [[init_node]]

		for w_node in winning_nodes:
			tmp = nx.all_shortest_paths(graph, source=init_node, target=w_node) # shortest or simple?
			subgraph_nodes = subgraph_nodes + list(tmp)

		subgraph_nodes_unique = np.unique(functools.reduce(operator.iconcat, subgraph_nodes, []))
		subgraph = graph.subgraph(subgraph_nodes_unique)
		
		if centrality == "eigenvector":
			subgraph_metric_values = eigenvector_centrality(subgraph.reverse(), max_iter=10000)
		elif centrality == "katz":
			subgraph_metric_values = katz_centrality(subgraph.reverse(), max_iter=10000)
		else:
			centrality_fn = centrality_fns[centrality]
			subgraph_metric_values = centrality_fn(subgraph)
		
		
		metric_values = {node: 0. for node in graph.nodes}
		for n in subgraph_nodes_unique:
			metric_values[n] = subgraph_metric_values[n]

	nx.set_node_attributes(graph, metric_values, centrality)
	
	return graph

def add_all_graph_attrs(graph_path, out_path, win_only, winning_nodes, init_node):
	centralities = ["betweenness", "closeness", "eigenvector", "katz", "load", "pagerank", "out_degree"]
	graph = nx.read_gexf(graph_path)

	for c in centralities:
		graph = add_centrality_attr(c, graph, win_only, winning_nodes, init_node)
	
	nx.write_gexf(graph, out_path)
	
	return True


def generate_rooms_stg():
	pass


if __name__=="__main__":
	
	# Generate graph
	graph_path = "./centrality_experiments/graphs/five_square_solitaire.gexf"
	out_path = "./centrality_experiments/graphs/five_square_solitaire_win.gexf"
	
	layout_dir = './centrality_experiments/environments/peg_solitaire_layouts/'
	layout1 = layout_dir + '4square.txt'
	layout2 = layout_dir + 'heart.txt'
	layout3 = layout_dir + '5square.txt'
	layout4 = layout_dir + 'cross_1111.txt'

    # Initial state
	gap_coord_4square = [(1, 2)]
	gap_coord_heart = [(2, 2)]
	gap_coord_5square = [(1, 2)]
	gap_coord_cross1111 = [(2, 2)]

	four_square_init_state = peg_solitaire_state(layout1, gap_coord_4square)
	five_square_init_state = peg_solitaire_state(layout3, gap_coord_5square)
	heart_init_state = peg_solitaire_state(layout2, gap_coord_heart)
	cross1111_init_state = peg_solitaire_state(layout4, gap_coord_cross1111)


	# Generate graph
	# graph_path = "./centrality_experiments/graphs/heart_peg_solitaire.gexf"
	# out_path = "./centrality_experiments/graphs/hps_win.gexf"
	# start_state = [1] * 16
	# start_state[9] = 0
	# init_state = heart_peg_state(state = start_state)	


	print('Time to make the graph')
	interaction_graph = generate_interaction_graph(initial_states = [cross1111_init_state])
	
	nx.write_gexf(interaction_graph, graph_path)
	
	# four_square_winning_idc = [(i, j) for i in range(4) for j in range(4)]
	# four_square_winning_nodes = []
	
	# for idx in four_square_winning_idc:
	# 	tmp = deepcopy(four_square_winning_idc)
	# 	tmp.remove(idx)
	# 	four_square_winning_nodes.append(tmp)

	# winning_nodes = [str(i) for i in winning_nodes if str(i) in interaction_graph]
	# init_node = str(heart_init_state)
	# add_all_graph_attrs(graph_path=graph_path, out_path=out_path, win_only=False, winning_nodes=None, init_node=init_node)

	
	