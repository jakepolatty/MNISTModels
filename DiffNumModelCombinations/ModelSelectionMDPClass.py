# Python imports.
import random
import numpy as np

# Other imports
from simple_rl.mdp.MDPClass import MDP
from ModelSelectionStateClass import ModelSelectionState

class ModelSelectionMDP(MDP):
    ACTIONS = [0, 1, 2]

    def __init__(self, models, y_test, predictions, probs, cutoff):
        self.models = models
        self.model_count = len(models)
        self.y_test = y_test
        self.predictions = predictions
        self.probs = probs
        self.cutoff = cutoff

        self.first_model = random.randint(0, self.model_count - 1)

        model_mask = np.zeros(self.model_count)
        model_mask[self.first_model] = 1

        init_state = ModelSelectionState(model_mask)

        MDP.__init__(self, ModelSelectionMDP.ACTIONS, self._transition_func,
                    self._reward_func, init_state=init_state)

    
    def _transition_func(self, state, action):
        '''
        Args:
            state (State)
            action_dict (str)
        Returns
            (State)
        '''
       	for i in range(self.model_count):
       		if state.models[i] == 1:
       			state.models[i] = -1

       		if i == action and state.models[i] != -1:
       			state.models[i] = 1

       	print(state)
        return state


    def _reward_func(self, state, action_dict, next_state=None):
        '''
        Args:
            state (State)
            action (dict of actions)
        Returns
            (float)
        '''
        return random.randint(0, 1) / 10.0

    def __str__(self):
        return "model_selection"