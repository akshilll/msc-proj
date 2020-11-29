# Graph Centrality for Option Discovery
Experiments comparing graph centrality measures in subgoal detection for option discovery

## Setup

Running the code in this repo requires the installation of BaRL_SimpleOptions which can be found [here](https://github.com/Ueva/BaRL-SimpleOptions) as well as **networkx**, **numpy** and **pickle**.


## Running

To run the experiments navigate to **experiments/** and run **run_agent.py** which will generate options for each centrality from a previously computed graph and run 1000 episodes for 400 agents for each centrality and one for a primitive Q learning agent. The results will be pickled and stored in **experiments/results/**. **experiments/results/make_graph.py** can be used to generate figures displaying the results. 

NOTE: for any environment where symmetry is present, after creating the symmetry-reduced graph which takes into account the symmetry of the state, subgoals must be made for both the subgoal states found on the symmetry-reduced graph and the full graph. We use Josh's code in which the agent cannot account for symmetry when faced with a state. To an agent, the symmetry of a state is not the same as the original state. Therefore, we must generate subgoals for both the original state and the symmetry state. In turn, we must also generate options using the full graph. If we do not, then some of the subgoals will not appear on the graph and so an error will occur. This is not ideal as it means we do not account for symmetry when learning.