import random
import numpy as np
from tensorforce.environments import Environment


class ModelSelectionEnvironment(Environment):

    def __init__(self, num_models, output_size, model_outputs, y_test, avg_model_costs):
        super().__init__()

        self.num_models = num_models
        self.output_size = output_size

        self.avg_model_costs = avg_model_costs
        self.min_cost = np.amin(avg_model_costs)
        self.max_cost = np.amax(avg_model_costs)

        self.model_outputs = model_outputs
        self.y_test = y_test
        self.test_data_size = len(y_test)

        self.current_point = -1

        state_size = self.output_size + self.num_models
        self.state = np.zeros(shape=(state_size,))

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
        state = np.zeros(shape=(state_size,))

        self.current_point = random.randint(0, self.test_data_size)
        self.state = state

        return state


    #####################
    # Execution Functions
    #####################

    # TODO
    def compute_timestep(self, action):
        # Computes the state vector for the next state based upon the agent's selected action
        # Retrieves the output vector for the chosen model, loads it into the first part of the new
        # state vector, and updates the called model mask in the second half of the vector
        print("TEST", action)

        if action < self.num_models:
            model_output = self.model_outputs[action][self.current_point]
            new_mask = self.update_model_mask(self.state, action)
            return np.append(model_output, new_mask)
        else:
            return self.state
        

    def is_terminal(self, action):
        # Checks whether the execution has transitioned to a terminal state
        # Returns true if the state is terminal and false otherwise
        return action == self.num_models

    def reward(self, next_state, terminal):
        # Computes the numerical reward value for the current state of the environment
        # Version 1 has 0 reward for all non-terminal states,
        # a positive reward for a terminal state that yields the correct prediction,
        # and a negative reward for a terminal state that yields an incorrect prediction
        # Rewards are scaled based upon the number of model calls that have been made, 
        # with a correct prediction in a single call giving the maximum reward and
        # an incorrect prediction after calling all models giving the minimum reward
        if terminal:
            model_cost = self.get_called_model_cost(next_state)

            if self.is_correct_prediction(next_state):
                return model_cost / self.min_cost
            else:
                return -1.0 * model_cost / self.max_cost
        else:
            return 0.0

    def execute(self, actions):
        # Utilizes the helper functions to compute the next state,
        # determine whether this state is terminal, and return the reward
        next_state = self.compute_timestep(actions)
        self.state = next_state

        terminal = self.is_terminal(actions)
        reward = self.reward(next_state, terminal)
        return next_state, terminal, reward


    ##################
    # Helper Functions
    ##################

    def get_model_mask(self, state):
        # Takes in the state dictionary and converts the called model mask segment
        # into an array of zeros and ones
        raw_mask = state[self.output_size:]
        return raw_mask.astype(int)

    def get_called_model_count(self, state):
        # Takes in the state dictionary and returns the number of called models
        model_mask = self.get_model_mask(state)
        return np.sum(model_mask)

    def get_called_model_cost(self, state):
        # Takes in the state dictionary and returns the total cost of the called models
        # based upon the saved average cost array
        model_mask = self.get_model_mask(state)
        return np.dot(model_mask, self.avg_model_costs)

    def get_model_output(self, state):
        # Takes in the state dictionary and converts the previous model output segment
        # into an array of float values for the outputs
        return state[:self.output_size]

    def is_correct_prediction(self, state):
        # Takes in the state dictionary and returns whether the predicted output from the
        # previously called model is the correct prediction for the current datapoint
        # Note that this should only be called in terminal states to prevent early termination
        model_output = self.get_model_output(state)
        pred = np.argmax(model_output)
        return pred == self.y_test[self.current_point]

    def update_model_mask(self, state, action):
        # Takes in the state dictionary and action and returns a new model mask based upon
        # the selected model to call
        old_mask = self.get_model_mask(state)

        if old_mask[action] == 0:
            old_mask[action] = 1
        elif old_mask[action] == 0:
            old_mask[action] = -1

        return old_mask




