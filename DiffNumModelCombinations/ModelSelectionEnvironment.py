import random
import numpy as np
from tensorforce.environments import Environment


class ModelSelectionEnvironment(Environment):

    def __init__(self):
        super().__init__()

    ########################
    # Environment Definition
    ########################
    def states(self):
        # (output_size + num_models) size dict
        # first [output_size] values initialized to zeros for first model call
        # these [output_size] values will contain the output of the previous model for all later calls
        # last [num_models] values provide a mask of ones for all previously called models and zeros for uncalled models
        state_size = self.output_size + self.num_models
        return dict(type='float', shape=(state_size,))

    def actions(self):
        # (num_models + 1) action options
        # one action for selecting each of the possible models to call
        # one action for transitioning to terminal state and using the final prediction
        action_count = self.num_models + 1
        return dict(type='int', num_values=action_count)


    ######################
    # Environment Controls
    ######################
    def max_episode_timesteps(self):
        # Can only traverse each state once, so max timesteps will be the same as the possible number of actions
        return self.num_models + 1

    def close(self):
        super().close()

    def reset(self):
        # Initialize a state vector of all zeros
        # The first [output_size] zeros represent the output of the upcoming model calls
        # The last [num_models] zeros represent a mask for all models that have been called (initially, none have)
        state_size = self.output_size + self.num_models
        state = np.zeros(size=(state_size,))
        return state


    #####################
    # Execution Functions
    #####################
    def compute_timestep(self, action):
        # Computes the state vector for the next state based upon the agent's selected action
        # Retrieves the output vector for the chosen model, loads it into the first part of the new
        # state vector, and updates the called model mask in the second half of the vector
        state_size = self.output_size + self.num_models
        state = np.zeros(size=(state_size,))
        return state

    def is_terminal(self):
        # Checks whether the execution has transitioned to a terminal state
        # Returns true if the state is terminal and false otherwise
        return false

    def reward(self, next_state, terminal):
        # Computes the numerical reward value for the current state of the environment
        # Version 1 has 0 reward for all non-terminal states,
        # a positive reward for a terminal state that yields the correct prediction,
        # and a negative reward for a terminal state that yields an incorrect prediction
        # Rewards are scaled based upon the number of model calls that have been made, 
        # with a correct prediction in a single call giving the maximum reward and
        # an incorrect prediction after calling all models giving the minimum reward
        return 0.0

    def execute(self, actions):
        # Utilizes the helper functions to compute the next state,
        # determine whether this state is terminal, and return the reward
        next_state = self.compute_timestep(actions)
        terminal = self.is_terminal()
        reward = self.reward(next_state, terminal)
        return next_state, terminal, reward






