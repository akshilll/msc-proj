import operator
import numpy
import functools


num_to_coord = [(-1, 2), (1, 2), (-2, 1), (-1, 1), (0, 1), (1, 1), (2, 1), 
(-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
	   			(-1,-1), (0, -1), (1, -1),
	   					 (0, -2)]


symm_num_to_coord = [(-i, j) for (i, j) in num_to_coord] 




class heart_peg_state(State):

	def __init__(self, gap_list):
		"""
	    Instantiate new state
	    Params:
	        - gap_list: indices of holes which are empty. List(Integer)
	     Returns:
	        - State
	    """  
        self.gap_list = gap_list
		self.state = np.array([0 for i in range(15) if i in gap_list else 1])
		self.num_to_coord = [(-1, 2), (1, 2)], (-2, 1), (-1, 1), (0, 1), (1, 1), (2, 1), \ 
							 (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0), (-1,-1), (0, -1), \
							 (1, -1), (0, -2)]
		self.board_directions = np.array([(i, j) for i in range(2) for j in range(2)])

	def __str__(self, state):
		return self.state.tostring()


	def __eq__(self, other_state):
	## DO I DO SYMMETRY HERE? YES
		return (self.symmetry_state() == other_state.state()).all() or (self.state() == other_state.state())

    def symmetry_state(self):
        



	def get_available_actions(self):
	# where does the state itself come from 
		for i in range(len(self.state)):
			# Look for a gap
			if self.state[i] == 0:
				for (x, y) in []
		
		return list of hashable actions


	def take_action(self, action):

		(gap_number, direction_from) = action
		(x, y) = direction_from 

		self.state[] 

		return List(States)

	def is_action_legal(self, action) -> bool :
		# Check that the coordinate is within the board 
		# If the coordinate is on an edge check  that 
		# If there is a gap at the index specified
			# Check the self.gap_list
		# If there are two pegs in the holes in the direction specified


	def get_successor_states(self):
		available_actions = self.get_available_actions()
		
		out = [self.take_action(a) for a in available_actions] 
		
		out = functools.reduce(operator.iconcat, out, [])

		return out
