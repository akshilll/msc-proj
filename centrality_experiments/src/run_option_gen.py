from centrality_experiments.src.option_generation import generate_primitive_options, generate_subgoal_options


centralities = ["betweenness", "degree", "katz", "load", "pagerank", "eigenvector", "closeness"] 

# Run option generation for each centrality
for c in centralities:
    sg = generate_subgoal_options(c)
    if len(sg) > 0: print(sg[0].policy_dict)
    print(c, "done, with {} subgoals".format(len(sg)))
    


