import tensorflow as tf
import numpy as np
from tensorforce import Agent, Environment
from ModelSelectionEnvironment import ModelSelectionEnvironment
import helpers.helper_funcs as helpers

def main():
    print("TEST TEST TEST TEST")
    num_models, output_size, model_outputs, y_test, avg_model_costs = data_loader()    
    environment = ModelSelectionEnvironment(num_models, output_size, model_outputs, y_test, avg_model_costs)

    agent = Agent.create(
        agent='ppo', environment=environment,
        # Automatically configured network
        network='auto',
        # Optimization
        batch_size=10, update_frequency=2, learning_rate=1e-3, subsampling_fraction=0.2,
        optimization_steps=5,
        # Reward estimation
        likelihood_ratio_clipping=0.2, discount=0.99, estimate_terminal=False,
        # Critic
        critic_network='auto',
        critic_optimizer=dict(optimizer='adam', multi_step=10, learning_rate=1e-3),
        # Preprocessing
        preprocessing=None,
        # Exploration
        exploration=0.0, variable_noise=0.0,
        # Regularization
        l2_regularization=0.0, entropy_regularization=0.0,
        # TensorFlow etc
        name='agent', device=None, parallel_interactions=1, seed=None, execution=None, saver=None,
        summarizer=None, recorder=None
    )

    #runner(environment, agent, n_episodes=300)


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
    l4_model = tf.keras.models.load_model('models/cifar/l4_model')
    l5_model = tf.keras.models.load_model('models/cifar/l5_model')
    l6_model = tf.keras.models.load_model('models/cifar/l6_model')
    l7_model = tf.keras.models.load_model('models/cifar/l7_model')
    l8_model = tf.keras.models.load_model('models/cifar/l8_model')
    l9_model = tf.keras.models.load_model('models/cifar/l9_model')
    l10_model = tf.keras.models.load_model('models/cifar/l10_model')
    models = [l1_model, l2_model, l3_model, l4_model, l5_model, l6_model, l7_model, l8_model, l9_model, l10_model]

    num_models = 10
    num_samples = x_test.shape[0]
    output_size = 10
    avg_model_costs = [0.2400, 0.2876, 0.3061, 0.3114, 0.3804, 0.4302, 0.3061, 0.3114, 0.3804, 0.4302]

    model_outputs = np.zeros((num_models, num_samples, output_size))

    for i in range(num_models):
        model = models[i]
        model_probs = model.predict(x_test)
        print(model_probs.shape)
        model_outputs[i] = model_probs
        print(model_outputs.shape)

    return num_models, output_size, model_outputs, y_test, avg_model_costs


##################
# Running Controls
##################
def run(environment, agent, n_episodes, test=False):
    # Train for n_episodes
    for _ in range(n_episodes):
        # Initialize episode
        states = environment.reset()
        terminal = False

        while not terminal:
            # Episode timestep
            actions = agent.act(states=states)
            states, terminal, reward = environment.execute(actions=actions)
            agent.observe(terminal=terminal, reward=reward)

def runner(environment, agent, n_episodes, n_episodes_test=1, combination=1):
    # Train agent
    result_vec = [] #initialize the result list
    for i in range(round(n_episodes / 100)): #Divide the number of episodes into batches of 100 episodes
        if result_vec:
            print("batch", i, "Best result", result_vec[-1]) #Show the results for the current batch
        # Train Agent for 100 episode
        run(environment, agent, 100, combination=combination) 
        # Test Agent for this batch
        test_results = run(environment, agent, n_episodes_test, combination=combination, test=True)
        # Append the results for this batch
        result_vec.append(test_results) 
    # Plot the evolution of the agent over the batches
    plot_multiple(
        Series=[result_vec],
        labels = ["Reward"],
        xlabel = "episodes",
        ylabel = "Reward",
        title = "Reward vs episodes",
        save_fig=True,
        path="env",
        folder=str(combination),
        time=False,
    )
    #Terminate the agent and the environment
    agent.close()
    environment.close()


if __name__ == '__main__':
    main()


