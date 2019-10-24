from option_generation import generate_primitive_options, generate_subgoal_options


centralities = ["betweenness", "closeness", "degree", "eigenvector", "katz", "load", "pagerank"]

# Run option generation for each centrality
for c in centralities:
    sg = generate_subgoal_options(c)
    print(sg[0].policy_dict)
    print(c, "done, with {} subgoals".format( len(sg)))
    


