import numpy as np
import matplotlib.pyplot as plt

from relearn.bandit_world.agents import EpsilonGreedy, AnnealingEpsilonGreedy
from relearn.bandit_world.environments import (BernoulliBandit, 
                                               BernoulliCasino, 
                                               standard_bandit_problem,
                                               custom_bandit_problem,
                                               small_bandit_problem)

def experiment():
    '''
    simple example experiment of the MAB
    '''
    #to reproduce the result set a random seed
    np.random.seed(99)

    bandit_arms = standard_bandit_problem()

    environment = BernoulliCasino(bandits=bandit_arms)

    agent = EpsilonGreedy(epsilon=0.1, budget=1000, environment=environment)
    agent.solve()
    
    print_reward(agent)
    visualise_agent_actions(agent)


def anneal_experiment():
    '''
    simple example experiment of the MAB
    using AnnealingEpsilonGreedy
    '''
    #to reproduce the result set a random seed
    np.random.seed(99)

    bandit_arms = standard_bandit_problem()

    environment = BernoulliCasino(bandits=bandit_arms)

    agent = AnnealingEpsilonGreedy(budget=1000, environment=environment)
    agent.solve()
    
    print(agent._epsilon)
    print_reward(agent)
    visualise_agent_actions(agent)

def print_reward(agent):
    print('Total reward: {}'.format(agent.total_reward))
    print(agent._means)


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
    experiment()
    anneal_experiment()

