import tensorflow as tf
import numpy as np
from collections import namedtuple
from tensorforce import Agent, Environment
from ModelSelectionEnvironmentCurved import ModelSelectionEnvironment
import helpers.helper_funcs as helpers
import matplotlib.pyplot as plt
import pandas as pd
import os

def main():
    num_models, output_size, val_model_outputs, y_val, test_model_outputs, y_test, avg_model_costs, weight_table = data_loader()    
    environment = ModelSelectionEnvironment(num_models, output_size, val_model_outputs,
         y_val, test_model_outputs, y_test, avg_model_costs)
    #environment = ModelSelectionEnvironment(num_models, output_size, val_model_outputs, y_val, test_model_outputs, y_test, avg_model_costs, weight_table)

    # agent = Agent.create(
    #     agent='ppo', environment=environment,
    #     # Automatically configured network
    #     network='auto',
    #     # Optimization
    #     batch_size=10, update_frequency=2, learning_rate=1e-3, subsampling_fraction=0.2,
    #     optimization_steps=5,
    #     # Reward estimation
    #     likelihood_ratio_clipping=0.2, discount=0.99, estimate_terminal=False,
    #     # Critic
    #     critic_network='auto',
    #     critic_optimizer=dict(optimizer='adam', multi_step=10, learning_rate=1e-3),
    #     # Preprocessing
    #     preprocessing=None,
    #     # Exploration
    #     exploration=0.0, variable_noise=0.0,
    #     # Regularization
    #     l2_regularization=0.0, entropy_regularization=0.0,
    #     # TensorFlow etc
    #     name='agent', device=None, parallel_interactions=1, seed=None, execution=None, saver=None,
    #     summarizer=None, recorder=None
    # )

    # agent = Agent.create(
    #     agent='ppo', environment=environment, batch_size=10, learning_rate=1e-3
    # )

    agent = Agent.create(
        agent='tensorforce',
        environment=environment,  # alternatively: states, actions, (max_episode_timesteps)
        memory=10000,
        update=dict(unit='timesteps', batch_size=64),
        optimizer=dict(type='adam', learning_rate=3e-4),
        policy=dict(network='auto'),
        objective='policy_gradient',
        reward_estimation=dict(horizon=num_models+1)
    )

    time_weights = [x / 10000 for x in avg_model_costs]
    runner(environment, agent, n_episodes=5000, n_episodes_test=y_test.shape[0], time_weights=time_weights)


#################
# Data Management
#################
def data_loader():
    print("Loading data...")
    x_train, y_train, x_val, y_val, x_test, y_test = helpers.get_cifar10_data_val()
    y_val = tf.squeeze(y_val)
    y_test = tf.squeeze(y_test)

    print("Loading models...")
    l1_model = tf.keras.models.load_model('models/cifar/l1_model')
    l2_model = tf.keras.models.load_model('models/cifar/l2_model')
    l3_model = tf.keras.models.load_model('models/cifar/l3_model')
    #l4_model = tf.keras.models.load_model('models/cifar/l4_model')
    #l5_model = tf.keras.models.load_model('models/cifar/l5_model')
    #l6_model = tf.keras.models.load_model('models/cifar/l6_model')
    #l7_model = tf.keras.models.load_model('models/cifar/l7_model')
    #l8_model = tf.keras.models.load_model('models/cifar/l8_model')
    #l9_model = tf.keras.models.load_model('models/cifar/l9_model')
    #l10_model = tf.keras.models.load_model('models/cifar/l10_model')

    #models = [l1_model, l2_model, l3_model, l4_model, l5_model, l6_model, l7_model, l8_model, l9_model, l10_model]
    models = [l1_model, l2_model, l3_model]

    #avg_model_costs = [0.2400, 0.2876, 0.3061, 0.3114, 0.3804, 0.4302, 0.3061, 0.3114, 0.3804, 0.4302]
    avg_model_costs = [0.2400, 0.2876, 0.3061]

    num_models = len(models)
    num_samples = x_test.shape[0]
    output_size = 10

    val_model_outputs = np.zeros((num_models, x_val.shape[0], output_size))
    test_model_outputs = np.zeros((num_models, num_samples, output_size))

    for i in range(num_models):
        print("Loading model " + str(i + 1) + " outputs...")
        model = models[i]

        val_model_probs = model.predict(x_val)
        val_model_outputs[i] = val_model_probs

        test_model_probs = model.predict(x_test)
        test_model_outputs[i] = test_model_probs

    print("Loading model weights...")
    weight_table = compute_class_matrix_B(models, output_size, x_val, y_val)

    return num_models, output_size, val_model_outputs, y_val, test_model_outputs, y_test, avg_model_costs, weight_table

def compute_class_matrix_B(models, num_classes, x_test, y_test):
    # Get dictionary of counts of each class in y_test
    y_test_np = y_test.numpy()
    count_dicts = []

    # Set up accuracy grid
    num_models = len(models)
    accuracies = np.zeros((num_models, num_classes))

    # Iterate over all models and get their predicted outputs
    for i in range(num_models):
        model = models[i]

        model_probs = model.predict(x_test)
        model_preds = np.argmax(model_probs, axis=1)

        unique, counts = np.unique(model_preds, return_counts=True)
        count_dicts.append(dict(zip(unique, counts)))

        # Iterate over all 10 classes
        for j in range(num_classes):
            # Compute the number of times where the prediction matches the test output for that class
            class_count = len(np.where((model_preds == j) & (y_test_np == j))[0])
            accuracies[i][j] = class_count / count_dicts[i][j]

    print(accuracies)
    return accuracies

##################
# Running Controls
##################
def run(environment, agent, n_episodes, time_weights, test=False,):
    Score = namedtuple("Score", ["reward", "reward_mean"])
    score = Score([], [])
    correct = 0
    total_time = 0

    # Train for n_episodes
    for i in range(n_episodes):
        # Initialize episode
        if test:
            states = environment.reset(index=i)
        else:
            states = environment.reset()
        internals = agent.initial_internals()
        terminal = False

        while not terminal:
            if test:  # Test mode (deterministic, no exploration)
                actions, internals = agent.act(
                    states=states, internals=internals, independent=True
                )
                states, terminal, reward = environment.execute(actions=actions)

                if not terminal:
                    total_time += time_weights[actions]

                if reward > 0:
                    correct += 1
            else: # Train mode (exploration and randomness)
                actions = agent.act(states=states)
                states, terminal, reward = environment.execute(actions=actions)
                agent.observe(terminal=terminal, reward=reward)

                score.reward.append(reward)
                score.reward_mean.append(np.mean(score.reward))

    if test:
        return correct / n_episodes, total_time
    else:
        return score.reward_mean[-1]

def runner(environment, agent, n_episodes, time_weights, n_episodes_test=1, combination=1):
    # Train agent
    train_result_vec = []
    test_result_vec = []
    time_result_vec = []
    for i in range(round(n_episodes / 100)): #Divide the number of episodes into batches of 100 episodes
        # Train Agent for 100 episode
        train_results = run(environment, agent, 100, time_weights) 
        train_result_vec.append(train_results)
        # Test Agent for this batch
        if i % 5 == 0:
            test_results, total_time = run(environment, agent, n_episodes_test, time_weights, test=True)
            # Append the results for this batch
            test_result_vec.append(test_results) 
            time_result_vec.append(total_time)
            
        print("batch", i, "Best result", train_result_vec[-1]) #Show the results for the current batch
    # Plot the evolution of the agent over the batches
    # plot_multiple(
    #     Series=[train_result_vec],
    #     labels = ["Reward"],
    #     xlabel = "Episodes",
    #     ylabel = "Reward",
    #     title = "Reward vs episodes",
    #     save_fig=False,
    #     path="env",
    #     folder=str(combination),
    #     time=False,
    # )

    plot_multiple(
        Series=[test_result_vec],
        labels = ["Acccuracy"],
        xlabel = "Episodes",
        ylabel = "Accuracy",
        title = "Accuracy vs episodes",
        save_fig=False,
        path="env",
        folder=str(combination),
        time=False,
    )

    plot_multiple(
        Series=[time_result_vec],
        labels = ["Time"],
        xlabel = "Episodes",
        ylabel = "Time",
        title = "Time vs episodes",
        save_fig=False,
        path="env",
        folder=str(combination),
        time=False,
    )
    #Terminate the agent and the environment
    agent.close()
    environment.close()


####################
# Plotting Functions
####################
def plot_multiple(
    Series,
    labels,
    xlabel,
    ylabel,
    title,
    save_fig=False,
    path=None,
    folder=None,
    time=True,
):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25, left=0.25)
    Series = [pd.Series(s) for s in Series]

    for i, s in enumerate(Series):
        ax.plot(s.index, s, label=labels[i])
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
    locs, xticks_labels = plt.xticks()
    if time:
        if len(Series[0]) < 10000:
            xticks_labels = [int(int(loc) / 10) for loc in locs]
        else:
            xticks_labels = [round(int(loc) / 36000, 2) for loc in locs]
            ax.set_xlabel(xlabel[:-3] + "(h)")
    plt.xticks(locs, xticks_labels)
    lines = ax.get_lines()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    if len(Series) > 1:
        ax.legend(
            lines,
            [line.get_label() for line in lines],
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        )
    ax.set_title(title)
    if save_fig:
        if path:
            if folder:
                plt.savefig(os.path.join(path, "Graphs", str(folder), title))
            else:
                plt.savefig(os.path.join(path, "Graphs", title))
        else:
            if folder:
                plt.savefig(os.path.join("Graphs", title))
                plt.close()
            else:
                plt.savefig(os.path.join("Graphs", str(folder), title))
                plt.close()
    else:
        plt.show()
    plt.close(fig="all")


if __name__ == '__main__':
    main()


