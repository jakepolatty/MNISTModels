# Python imports.
import random

# Other imports
from simple_rl.mdp.MDPClass import MDP
from ModelSelectionStateClass import ModelSelectionState

class ModelSelectionMDP(MDP):
    ACTIONS = ["predict", "finish"]

    def __init__(self, models, y_test, predictions, probs, cutoff):
        self.models = models
        self.y_test = y_test
        self.predictions = predictions
        self.probs = probs
        self.cutoff = cutoff

        self.first_model = random.randint(0, len(models))

        init_state = ModelSelectionState(self.first_model, [])

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
        return state


    def _reward_func(self, state, action_dict, next_state=None):
        '''
        Args:
            state (State)
            action (dict of actions)
        Returns
            (float)
        '''
        if 

    def __str__(self):
        return "model_selection"