from option_generation import generate_primitive_options, generate_subgoal_options, sg_option
from barl_simpleoptions.options_agent import OptionAgent

from heart_peg_solitaire import heart_peg_env
import networkx as nx
import pickle
import numpy as np 
import matplotlib.pyplot as plt
import time 
import multiprocessing as mp
import sys

def run_experiment(num_agents, num_epi, centrality=None):
	# Generate options
	options = generate_primitive_options()
	print("{} Primitive options generated".format(len(options)))

	results_path  = "results/primitive_results.pickle"

	if centrality is not None:
		subgoal_options = generate_subgoal_options(centrality)
		print("{} Betweenness subgoal options generated")
		results_path  = "results/{}_results.pickle".format(centrality)

		options += subgoal_options


	env = heart_peg_env(options=options)

	print("environment generated")

	# Time it
	start_time = time.time()
	
	agents = [OptionAgent(env) for _ in range(num_agents)]
	episode_returns = [agent.run_agent(num_epi) for agent in agents]

	print("Total time was {}".format(time.time() - start_time))

	# Save data
	f = open(results_path, "wb+")
	pickle.dump(episode_returns, f)
	f.close()


	return episode_returns

if __name__ == "__main__":
	ret = run_experiment(250, 1000)