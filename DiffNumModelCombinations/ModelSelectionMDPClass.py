# Python imports.
import random
import numpy as np

# Other imports
from simple_rl.mdp.MDPClass import MDP
from ModelSelectionStateClass import ModelSelectionState

class ModelSelectionMDP(MDP):
    ACTIONS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def __init__(self, models, y_test, predictions, probs, rankings, cutoff):
        self.models = models
        self.model_count = len(models)
        self.y_test = y_test
        self.predictions = predictions
        self.probs = probs
        self.rankings = rankings
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

       	# if self.get_current_model(state) != -1:
       	# 	print(state, self.get_current_model(state))
        return state


    def _reward_func(self, state, action_dict, next_state):
        '''
        Args:
            state (State)
            action (dict of actions)
        Returns
            (float)
        '''
        current_model = self.get_current_model(state)
        next_model = self.get_current_model(next_state)

        if current_model == -1:
        	return -1
        elif next_model == -1:
        	return -1
        else:
        	print(current_model, next_model)
        	return self.rankings[next_model] - self.rankings[current_model]

    def get_current_model(self, state):
    	models = state.models

    	for i in range(self.model_count):
    		if state.models[i] == 1:
    			return i

    	return -1

    def __str__(self):
        return "model_selection"