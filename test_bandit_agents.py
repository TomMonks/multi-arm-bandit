import numpy as np
import matplotlib.pyplot as plt

from relearn.bandit_world.agents import EpsilonGreedy
from relearn.bandit_world.environments import BernoulliBandit, BernoulliCasino

def experiment():
    '''
    main test function
    '''
    bandit_arms = create_bandits(0.3, 0.75, 0.2, 0.5, 0.6, 0.7, 0.4)

    environment = BernoulliCasino(bandits=bandit_arms)

    agent = EpsilonGreedy(epsilon=0.1, budget=2000, environment=environment)

    agent.solve()

    print_reward(agent)
    visualise_agent_actions(agent)

def create_bandits(*means):
    return [BernoulliBandit(mean) for mean in means]

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

