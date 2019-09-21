# msc-proj
Experiments for my MSc project in skill acquisition... everything is in the experiments folder!

## Setup

Running the code in this repo requires the installation of BaRL_SimpleOptions which can be found [here](https://github.com/Ueva/BaRL-SimpleOptions) as well as **networkx**, **numpy** and **pickle**.


## Running

To run the experiments navigate to **experiments/** and run **run_agent.py** which will generate options for each centrality from a previously computed graph and run 1000 episodes for 400 agents for each centrality and one for a primitive Q learning agent.



## Folder structure

### experiments/

**experiments/environments/** contains nothing anymore since I chose to rejig the file structure as there was an issue with my VS code

**experiments/tests/** contains the unit tests for my implementation of the environment and state classes for the heart peg solitaire

**experiments/results/** contains pickle files of the episode returns from the experiments run for 400 agents for 1000 episodes each as well as matlplotlib graphs for comparing the results of each agent

**graph_generation.py** contains code for generating an STG for the game which is called using **run_graph_generation.py**

**option_generation.py** contains code for generating options for each centrality which is called using **run_option_gen.py** but this is never called outside of **run_agent.py**

**run_subgoal_extract.py** calls code from **subgoal_extraction.py** which extracts subgoals for each centrality from a previously computed graph





