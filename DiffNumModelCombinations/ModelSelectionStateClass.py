import numpy as np
from simple_rl.mdp.StateClass import State

class ModelSelectionState(State):
    ''' Class for Model Selection MDP States '''

    def __init__(self, models):
        State.__init__(self, data=[models])
        self.models = models

    def __hash__(self):
        return hash(tuple(self.models))

    def __str__(self):
        return str(self.models)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        '''
        Summary:
            Model states are equal when their current models and prior models are the same
        '''
        return isinstance(other, ModelSelectionState) and np.array_equal(self.models, other.models)