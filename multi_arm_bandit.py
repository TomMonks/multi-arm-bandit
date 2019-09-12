#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimisation module for multi-arm bandits applied to web analytics

Algorithms:
1. epsilon-greedy
2. upper confidence bound
3. thompson sampling
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    simulated_data = pd.read_csv('data/Ads_CTR_Optimisation.csv')
    #ucb_ads_selected, ucb_reward = ucb(simulated_data)
    ts_ads_selected, ts_reward  = thompson_sampling(simulated_data)
    
    #visualise_ads(ucb_ads_selected)
    visualise_ads(ts_ads_selected)
    
    print('UCB_Reward:\t{}'.format(ucb_reward))
    print('TS_Reward:\t{}'.format(ts_reward))
    
    return ts_ads_selected, ts_reward, ucb_ads_selected, ucb_reward
    

def ucb(simulated_data):
    '''
    Upper Confidence Bound Algorithm for reinforcement learning in the 
    online bernouli multi-arm bandit problem
    
    Keyword arguments:
    simulated_data -- the simulated_data to use in the learning
                      problem
    
    Returns:
    ------
    Tuple with 2 items
    0: np.ndarray (vector), ads selected at each round
    1: int, total reward
    '''
    
    total_rounds = simulated_data.shape[0]
    bandits_n = simulated_data.shape[1]

    ads_selected = np.zeros(total_rounds, np.int32)
    
    number_of_selections = np.zeros(bandits_n, np.int32)
    sums_of_rewards = np.zeros(bandits_n, np.int32)
    total_reward = 0

    for round_n in range(total_rounds):
       
        average_rewards = np.divide(sums_of_rewards, number_of_selections)
        deltas = np.sqrt(3/2 * (np.log(round_n + 1) / number_of_selections))
        upper_bounds = average_rewards + deltas
        
        max_upper_bound_index = np.argmax(upper_bounds)
        
        ads_selected[round_n] = max_upper_bound_index
        
        number_of_selections[max_upper_bound_index] += 1
        reward = simulated_data.values[round_n, max_upper_bound_index]
        sums_of_rewards[max_upper_bound_index] += reward
        total_reward += reward
        
    return ads_selected, total_reward
    


def thompson_sampling(simulated_data):
    '''
    Thompson Sampling Algorithm for reinforcement learning in the
    online bernouli multi-arm bandit problem
    
    Keyword arguments:
    simulated_data -- the simulated_data to use in the learning
                      problem
    
    Returns:
    ------
    Tuple with 2 items
    0: np.ndarray (vector), ads selected at each round
    1: int, total reward
    '''
    total_rounds = simulated_data.shape[0]
    bandits_n = simulated_data.shape[1]
    
    ads_selected = np.zeros(total_rounds, np.int32)
    
    number_of_rewards_1 = np.zeros(bandits_n, np.int32)
    number_of_rewards_0 = np.zeros(bandits_n, np.int32)
    total_reward = 0

    for round_n in range(total_rounds):
       
        random_betas = np.random.beta(number_of_rewards_1 + 1, 
                                      number_of_rewards_0 + 1)
        
        max_index = np.argmax(random_betas)
        
        ads_selected[round_n] = max_index
        reward = simulated_data.values[round_n, max_index]  
        
        if reward == 1:
            number_of_rewards_1[max_index] += 1
        else:
            number_of_rewards_0[max_index] += 1
        
        total_reward += reward
        
    return ads_selected, total_reward

 
def visualise_ads(ads_selected):
    '''
    Visualise the adverts selected as a histogram
    '''
    plt.hist(ads_selected)
    plt.title('Histogram of ads selections')
    plt.xlabel('Ads')
    plt.ylabel('Number of times each ad was selected')
    plt.show()


if __name__ == '__main__':
    ts_ads_selected, ts_reward, ucb_ads_selected, ucb_reward = main()
    