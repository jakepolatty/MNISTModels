from tensorforce import Agent, Environment
from ModelSelectionEnvironment import ModelSelectionEnvironment

def main():
    environment = ModelSelectionEnvironment()

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

    runner(environment, agent, n_episodes=300)


def run(environment, agent, n_episodes, test=False)
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






