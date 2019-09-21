# msc-proj
Experiments for my MSc project in skill acquisition... everything is in the experiments folder!

## Setup

Running the code in this repo requires the installation of BaRL_SimpleOptions which can be found [here](https://github.com/Ueva/BaRL-SimpleOptions) as well as **networkx**, **numpy** and **pickle**.


## Running

To run the experiments navigate to **experiments/** and run **run_agent.py** which will generate options for each centrality from a previously computed graph and run 1000 episodes for 400 agents for each centrality and one for a primitive Q learning agent. The results will be pickled and stored in **experiments/results/**. **experiments/results/make_graph.py** can be used to generate figures displaying the results. 






