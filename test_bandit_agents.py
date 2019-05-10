import numpy as np
import matplotlib.pyplot as plt

from relearn.bandit_world.agents import (EpsilonGreedy, 
                                         AnnealingEpsilonGreedy,
                                         OptimisticInitialValues)

from relearn.bandit_world.environments import (BernoulliBandit, 
                                               BernoulliCasino, 
                                               standard_bandit_problem,
                                               custom_bandit_problem,
                                               small_bandit_problem)

def epsilon_greedy_experiment(epsilon=0.1, budget=1000, random_state=None):
    '''
    simple example experiment of the MAB
    '''
    print('------\nAgent: Epsilon-Greedy')
    #to reproduce the result set a random seed
    np.random.seed(seed=random_state)

    bandit_arms = custom_bandit_problem(0.2, 0.5, 0.3, 0.75, 0.3)

    environment = BernoulliCasino(bandits=bandit_arms)

    agent = EpsilonGreedy(epsilon=0.1, budget=budget, environment=environment)
    agent.solve()
    
    print_reward(agent)
    visualise_agent_actions(agent)


def anneal_experiment(budget=1000, random_state=None):
    '''
    simple example experiment of the MAB
    using AnnealingEpsilonGreedy
    '''
    print('--------\nAgent:\tAnnealing Epsilon-Greedy')
    #to reproduce the result set a random seed
    np.random.seed(seed=random_state)

    bandit_arms = custom_bandit_problem(0.2, 0.5, 0.3, 0.75, 0.3)

    environment = BernoulliCasino(bandits=bandit_arms)

    agent = AnnealingEpsilonGreedy(budget=budget, environment=environment)
    agent.solve()
    
    print_reward(agent)
    visualise_agent_actions(agent)


def optimistic_experiment(budget=1000, random_state=None):
    '''
    simple example experiment of the MAB
    using AnnealingEpsilonGreedy
    '''
    print('-------\nAgent: Optimistic Initial Values')
    #to reproduce the result set a random seed
    np.random.seed(seed=random_state)

    bandit_arms = custom_bandit_problem(0.2, 0.5, 0.3, 0.75, 0.3)

    environment = BernoulliCasino(bandits=bandit_arms)

    agent = OptimisticInitialValues(budget=budget, environment=environment)
    agent.solve()

    print_reward(agent)
    visualise_agent_actions(agent)

def print_reward(agent):
    print('Total reward: {}'.format(agent.total_reward))
    print('\nFinal Model:\n------')
    for bandit_index in range(len(agent._means)):
        print('Bandit {0}:\t{1:.2f}'.format(bandit_index + 1, agent._means[bandit_index]))


def visualise_agent_actions(agent):
    '''
    Visualise the actions taken in a bar chart

    Keyword arguments:
    -----
    actions -- np.ndarray of the count that each 
               arm was pulled
    
    '''
    actions = agent.actions
    x = [i + 1 for i in range(actions.shape[0])]
    plt.bar(x, actions)
    plt.title('Histogram of Actions Taken by Algorithm')
    plt.xlabel('Arm')
    plt.ylabel('Number of times each arm was selected')
    plt.show()

if __name__ == '__main__':
    #experiment()
    anneal_experiment()
    optimistic_experiment()

