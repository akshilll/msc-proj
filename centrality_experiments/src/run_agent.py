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
def run_experiment(num_agents, num_epi, centrality=None):
	# Generate primitive options
	options = generate_primitive_options()
	print("{} Primitive options generated".format(len(options)))

	results_path  = "./centrality_experiments/results/primitive_results.pickle"
	
	# Add subgoal options
	if centrality is not None:
		subgoal_options = generate_subgoal_options(centrality)
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

	
