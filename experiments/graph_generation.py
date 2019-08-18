import numpy as np
from copy import deepcopy
import networkx as nx 
from heart_peg_solitaire import heart_peg_env, heart_peg_state
from typing import List
from barl_simpleoptions.state import State


# Josh's code, slightly optimised by me
def generate_interaction_graph(initial_states : List['State']) :
	"""
	Generates the state-transition graph for this environment.
	"""
	states = []

	# Add initial states to current successor list.
	current_successor_states = deepcopy(initial_states)
	print(current_successor_states)

	# While we have no new successor states to process.
	while (not len(current_successor_states) == 0) :
		# Add each current new successor to our list of processed states.
		
		next_successor_states = []
		for successor_state in current_successor_states :
			if (successor_state not in current_successor_states) :
				states.append(successor_state)

				# Add this state's successors to the successor list.
				if (not successor_state.is_terminal_state()) : 
					for new_successor_state in successor_state.get_successors() :
						next_successor_states.append(new_successor_state)

		current_successor_states = deepcopy(next_successor_states)
		print(current_successor_states, states)
	
	# Create graph from state list.
	interaction_graph = nx.DiGraph()
	for state in states :
		
		# Add state to interaction graph.
		interaction_graph.add_node(state)

		for successor_state in state.get_successors() :
			interaction_graph.add_node(successor_state)
			interaction_graph.add_edge(state, successor_state)
	
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
	print(interaction_graph)
	nx.write_gexf(interaction_graph, file_path)


