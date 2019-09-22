'''
module simulator

Provides:

1. Monte carlo classes and function to test of agents in MAB environments

'''

import numpy as np

class AgentSimulation(object):
    '''
    Encapsulates a Monte-Carlo simulation framework for agents in 
    multi-arm bandit environment

    Agents must implement the interface

    solve()
    reset()
    '''

    def __init__(self, environment, agent, replications=1000):
        self._env = environment
        self._agent = agent
        self._reps = replications

    def simulate(self):

        best_indexes = np.zeros(self._reps, np.int32)

        for rep in range(self._reps):
            self._agent.reset()
            self._agent.solve()
            best_indexes[rep] = self._agent.best_arm

        return best_indexes

class ExperimentResults(object):
    '''
    Results Container for an Agent Experiment
    '''
    def __init__(self, selections, correct_selections, p_correct_selections):
        self.selections = selections
        self.correct_selections = correct_selections
        self.p_correct_selections = p_correct_selections

class Experiment(object):
    '''
    Test the power of a given configuration of an agent
    at correct selection of a max of min 
    '''
    def __init__(self, env, agent, best_index=0, objective='max', replications=1000):
        self._env = env
        self._agent = agent
        self._sim = AgentSimulation(env, agent, replications = replications)
        self._best_index = best_index
        self._objective = objective
        self._reps = replications
    
    def execute(self):
        '''
        Execute the experiment
        '''
        selections = self._sim.simulate()
        
        correct_selections = (selections == self._best_index).sum()
        
        p_correct_selections = correct_selections / self._reps

        return ExperimentResults(selections, correct_selections, p_correct_selections)
    








