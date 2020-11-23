from centrality_experiments.src.option_generation import generate_primitive_options, generate_subgoal_options, sg_option
from centrality_experiments.environments.heart_peg_solitaire import heart_peg_env

from barl_simpleoptions.options_agent import OptionAgent


import networkx as nx
import pickle
import numpy as np 
import matplotlib.pyplot as plt
import time 

import sys
import datetime




# TODO: change this to have all running functionality in this file with if file exists statements to save time. 
def run_experiment(num_agents, num_epi, centrality=None, graph_path="./centrality_experiments/graphs/heart_peg_solitaire_graph.gexf"):
	"""
	TODO: make the function behave as follows by calling other functions if required.
	Input: environment, num_subgoals, centrality, num_agents, num_episode
	1) If the graph does not exist, generate a graph and save it somewhere
	2) If the subgoals do not exist, generate the subgoals and save them somewhere
	3) If the primitive options do not exist generate them and save 
	4) If the subgoal options do not exist, generate them and save
	5) Run the experiments for each agent and save pickled results
	"""

	
	
	# Generate primitive options
	options = generate_primitive_options(graph_path=graph_path)
	print("{} Primitive options generated".format(len(options)))

	results_path  = "./centrality_experiments/results/primitive_results.pickle"
	
	# Add subgoal options
	if centrality is not None:
		subgoal_options = generate_subgoal_options(centrality, graph_path=graph_path)
		print("{} subgoal options generated".format(len(subgoal_options)))
		results_path  = "./centrality_experiments/results/{}_results.pickle".format(centrality)

		options += subgoal_options


	env = heart_peg_env(options=options)

	print("environment generated")

	# Time it
	start_time = time.time()
	
	# Create a list of agents and run the episodes
	agents = [OptionAgent(env) for _ in range(num_agents)]
	episode_returns = [agent.run_agent(num_epi) for agent in agents]

	print("Total time was {}".format(time.time() - start_time))
	
	f = open(results_path, "wb")
	pickle.dump(episode_returns, f)
	f.close()


	return episode_returns

if __name__ == "__main__":
	print(datetime.datetime.now())
	run_experiment(500, 1000, centrality="load")
	print("done load")
	print(datetime.datetime.now())
	run_experiment(500, 1000, centrality="betweenness")
	print("done betweenness")
	print(datetime.datetime.now())
	run_experiment(500, 1000, centrality="katz")
	print("done katz")
	print(datetime.datetime.now())
	run_experiment(500, 1000, centrality="eigenvector")
	print("done eigenvector")
	print(datetime.datetime.now())
	run_experiment(500, 1000)
	print("done primitive")
	print(datetime.datetime.now())
	run_experiment(500, 1000, centrality="degree")
	print("done degree")
	print(datetime.datetime.now())
	run_experiment(500, 1000, centrality="closeness")
	print("done eigenvector")
	print(datetime.datetime.now())
	run_experiment(500, 1000, centrality="pagerank")
	print("done pagerank")

	
