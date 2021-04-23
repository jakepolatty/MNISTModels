import numpy as np
from simple_rl.mdp.StateClass import State

class ModelSelectionState(State):
    ''' Class for Chain MDP States '''

    def __init__(self, current, visited):
        State.__init__(self, data=[current, visited])
        self.current = current
        self.visited = visited

    def __hash__(self):
        return hash(tuple(self.visited))

    def __str__(self):
        return "Current: " + str(self.num) + " - Visited: " + str(self.visited)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        '''
        Summary:
            Model states are equal when their current models and prior models are the same
        '''
        return isinstance(other, ModelSelectionState) and self.current == other.current and np.array_equal(self.visited, other.visited)