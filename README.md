# Graph Centrality for Option Discovery

This repository contains experiments comparing graph centrality metrics used for in subgoal detection in hierarchical reinforcement learning. I have implemented functionality for:

- [x] State transition graph generation for a given environment (assuming discrete state and action spaces).
- [x] Subgoal detection using node centrality metric
- [x] Visualising the state transition graph and detected subgoals.
- [x] Option generation given a subgoal. 
- [x] A pipeline for generating subgoals, creating options based on those subgoals, and training an agent equipped with those options.

#### Centrality Measures Implemented

- [x] Betweenness
- [x] Load
- [x] Degree
- [x] Closeness
- [x] Katz
- [x] Eigenvector
- [x] PageRank


## Setup
We build on the code from BaRL_SimpleOptions. Running the experiments, therefore, requires installing BaRL_SimpleOptions which can be found [here](https://github.com/Ueva/BaRL-SimpleOptions), as well as **networkx**, **numpy** and **pickle**.


## Running

To run the experiments navigate to **experiments/** and run **run_agent.py**. This will generate options for each centrality metric and run training for an agents using each set of options generated. The results will be pickled and stored in **experiments/results/**. The code in **experiments/results/make_graph.py** can be used to generate figures displaying the results. 
