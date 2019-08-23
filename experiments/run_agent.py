from option_generation import generate_primitive_options, generate_subgoal_options, sg_option
from barl_simpleoptions.options_agent import OptionAgent

from heart_peg_solitaire import heart_peg_env
import networkx as nx
import pickle
import numpy as np 
import matplotlib.pyplot as plt

# Read in graphs
graph = nx.read_gexf("graphs/heart_peg_solitaire_graph.gexf")
graph_without_symm = nx.read_gexf("graphs/heart_peg_solitaire_graph_without_symm.gexf")

print("Graphs are in")

# Generate options
primitive_options = generate_primitive_options()
print("{} Primitive options generated".format(len(primitive_options)))

betweenness_subgoal_options = generate_subgoal_options("betweenness")
primitive_options += betweenness_subgoal_options
#print("{} Betweenness subgoal options generated")

env = heart_peg_env(options=primitive_options)

print("environment generated")

primitive_agent = OptionAgent(env)

print("agent made")

num_epi = 500
num_agents = 1
episode_returns = [0] * num_agents 
episode_returns = primitive_agent.run_agent(num_epi)
	

#episode_returns = np.array(episode_returns).reshape((num_agents, num_epi)).mean(axis=0)


# Save Return Graph.
plt.plot(range(1, num_epi + 1), np.cumsum(episode_returns))
plt.title("Agent Training Curve")
plt.xlabel("Episode Number")
plt.ylabel("Episode Return")
plt.savefig("episode_returns.png")