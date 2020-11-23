import numpy as np
from barl_simpleoptions.option import PrimitiveOption, SubgoalOption
from centrality_experiments.environments.heart_peg_solitaire import heart_peg_state
from centrality_experiments.src.subgoal_extraction import string_to_list

import networkx as nx
import functools
import operator
import os
import pickle


# Subgoal Option class based mostly on Josh's code from https://github.com/Ueva/Betweenness-Project
class sg_option(SubgoalOption):
    # Slightly altered version of Josh's __init__
    def __init__(self, subgoal, graph, initiation_set_size=20, policy_file_path=None):
        self.graph = graph
        self.subgoal = subgoal
        self.initiation_set_size = initiation_set_size
        self._build_initiation_set()

        # Modification by Akshil  
        if policy_file_path is not None:

            self.policy_file_path = policy_file_path
            
            # Load the policy file for this option.
            with open(policy_file_path, mode = "rb") as f:
                self.policy_dict = pickle.load(f)
        else:

            # Generate policy
            self.policy_dict = self.generate_policy_dict()
            # Write it to a file for later
            with open("./centrality_experiments/subgoals/policies/{}_policy.format".format(str(subgoal)), "wb") as f:
                pickle.dump(self.policy_dict, f)

    # Slightly modified version of Josh's code in github.com/Ueva/Betweenness-Project/Code/Simple%20Numpile%20Experiments/options.py
    def generate_policy_dict(self):
        """
        Generates policy for subgoal option

        Returns: 
        policy_dict : Dictionary mapping states to actions
        """
        subgoal_string = str(self.subgoal)
        
        
        policy_dict = {}

        for state_string in self.graph.nodes:

            # Only do this for states in initiation set
            if state_string in self.initiation_set:

                # If there is a path from that node to the subgoal node.
                if (nx.has_path(self.graph, state_string, subgoal_string)) :
                
                    # Make state from string
                    state = heart_peg_state(string_to_list(state_string))

                    # Get the shortest path to that node. 
                    shortest_path = nx.shortest_path(self.graph, state_string, subgoal_string)
                    
                    # Get next node on the path.
                    next_state = shortest_path[1]

                    # Get action needed to transition to that state.
                    for successor in state.get_successors() :
                        if (str(successor) == next_state) :
                            next_state = successor
                            break

                action = state.get_transition_action(next_state)
                policy_dict[state_string] = action
        return policy_dict
    
    def __str__(self):
        return "SubgoalOption({})".format(str(self.subgoal))
            

def generate_primitive_options(graph_path="./centrality_experiments/graphs/heart_peg_solitaire_graph.gexf"):
    """Gets all possible actions from a graph and generates list of PrimitiveOptions

    Arguments:
    graph_path -- String containing file path to graph relative to graph

    Returns:
    primitive_options -- List of primitive option objects from all available actions of all states in graph
    """
    # Read in graph
    graph = nx.read_gexf(graph_path)
    
    # Build list of lists of available_actions for each state in graph
    primitive_actions = [heart_peg_state(string_to_list(node)).get_available_actions() for node in nx.nodes(graph)]
    
    # Reduce the list to 1d 
    primitive_actions = list(set(functools.reduce(operator.iconcat, primitive_actions, [])))
    
    # Construct list of primitive options
    primitive_options = [PrimitiveOption(a) for a in primitive_actions]

    return primitive_options

def generate_subgoal_options(centrality, graph_path="./centrality_experiments/graphs/heart_peg_solitaire_graph.gexf"):
    print("Generating subgoals for {}".format(centrality))

    if centrality not in ["betweenness", "closeness", "degree", "eigenvector", "katz", "load", "pagerank"]:
        raise Exception("You have not chosen an available centrality measure!")
    assert type(centrality) == str

    subgoals = []
    
    # Path where the subgoals reside
    sg_file_path = "./centrality_experiments/subgoals/{}.txt".format(centrality)

    # Read in graph without symmetry
    graph = nx.read_gexf(graph_path)

    # Open it up
    f = open(sg_file_path, "rb")
    subgoal_strings = pickle.load(f)

    # Generate subgoals and symmetry subgoals and pickle them
    for s in subgoal_strings:
        state = string_to_list(s)
        print(state)
        sg_state = heart_peg_state(state=state)
        symm_state = sg_state.symm_state
        sg_symm_state = heart_peg_state(state=symm_state)

        # If the subgoal is the initial state 
        if sg_state.is_initial_state():
            print(centrality, "INITIAL STATE FOUND AS SUBGOAL")
            continue

        if os.path.exists("./centrality_experiments/subgoals/policies/{}.pickle".format(str(sg_state))):
            policy_fp = "./centrality_experiments/subgoals/policies/{}.pickle".format(str(sg_state))
        else: 
            policy_fp = None
        
        if os.path.exists("./centrality_experiments/subgoals/policies/{}.pickle".format(str(sg_symm_state))):
            policy_fp_symm = "./centrality_experiments/subgoals/policies/{}.pickle".format(str(sg_symm_state))
        else: 
            policy_fp_symm = None

        subgoals.append(sg_option(sg_state, graph=graph, policy_file_path=policy_fp))
        subgoals.append(sg_option(sg_symm_state, graph=graph, policy_file_path=policy_fp_symm))
    
    # Close the file
    f.close()

    return list(set(subgoals))

if __name__ == "__main__":
    centralities = ["betweenness", "degree", "katz", "load", "pagerank", "eigenvector", "closeness"] 

    # Run option generation for each centrality
    for c in centralities:
        sg = generate_subgoal_options(c)
        if len(sg) > 0: print(sg[0].policy_dict)
        print(c, "done, with {} subgoals".format(len(sg)))