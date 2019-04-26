'''
Agents that are designed to solve an instance of multi-arm bandit problem

Classes:
--------
EpsilonGreedy --    Epsilon-Greedy tackles the explore-exploit dilemma by acting greedy
                    1 - epsilon of the time and explore epsilon of the time.  

'''
import numpy as np 


class EpsilonGreedy(object):
    '''
    Encapsulates an Epsilon-Greedy based agent for reinforcement learning
    in multu-arm bandit problems.  The agent interacts and learns from  
    an environment consisting of multiple bandit arms

    Epsilon-Greedy tackles the explore-exploit dilemma by acting greedy
    1 - epsilon of the time and explore epsilon of the time.  

    Note: Implements the observer pattern to recieve feedback from the 
    environment

    Public properties:
    -------
    total_reward -- float, the cumulative reward recieved from the environment
    actions -- np.ndarry (vector), a record of the actions (arms) that have been 
               taken by the agent.
            
    Public methods:
    --------
    solve() -- initiates the algorithm for the budget specified
    feedback() -- observer pattern notification method.  This is called by the
                  environment when the result of an action is ready to be reported.

    '''

    def __init__(self, epsilon, budget, environment):
        '''
        Epsilon-Greedy Algorithm constructor method

        Keyword arguments:
        ------
        epsilon -- float, proportion of time to explore the environment.
                   1 - epsilon = the proportion of time to exploit 
                   (make greedy decisions) about the environment

        budget -- int, number of iterations of the algorithm to run

        environment -- object implementing action interface...to add.
        '''
        self._validate_init(epsilon, budget, environment)
        environment.register_observer(self)
        self._env = environment
        self._epsilon = epsilon
        self._total_rounds = budget
        self._total_reward = 0
        self._current_round = 0
        self._actions = np.zeros(environment.number_of_arms, np.int32)
        self._means = np.zeros(environment.number_of_arms, np.float64)
        
    def _validate_init(self, epsilon, budget, environment):
        '''
        Validate the arguments passed to the constructor method

        Keyword arguments:
        ------
        epsilon -- float, proportion of time to explore the environment.
                   1 - epsilon = the proportion of time to exploit 
                   (make greedy decisions) about the environment

        budget -- int, number of iterations of the algorithm to run

        environment -- object implementing action interface...to add.
        '''
        self._validate_epsilon(epsilon)
        self._validate_budget(budget)
        
    def _validate_epsilon(self, epsilon):
        if type(epsilon) != float or epsilon < 0.0 or epsilon > 1.0:
            msg = 'epsilon argument must be a float value between 0 and 1'
            raise ValueError(msg)

    def _validate_budget(self, budget):
        if budget < 0:
            msg = 'budget argument must be a int > 0'
            raise ValueError(msg)

    def _get_total_reward(self):
        return self._total_reward

    def _get_action_history(self):
        return self._actions
    
    def solve(self):
        '''
        Run the epsilon greedy algorithm in the 
        environment to find the best arm 
        '''
        for i in range(self._total_rounds):
            sample = np.random.uniform()
            if sample > self._epsilon:
                self._exploit()  
            else:
                self._explore()
            self._current_round += 1

    def _exploit(self):
        '''
        Exploit the best arm found
        Interacts with environment and 
        performs the best know action
        '''
        best_index = self._best_arm()
        self._env.action(best_index)  
        

    def _best_arm(self):
        '''
        Return the index of the arm 
        with the highest expected value

        Returns:
        ------
        int, Index of the best arm
        '''
        return np.argmax(self._means)

    def _explore(self):
        '''
        Explore the environment.
        Take a random action and learn from it
        '''
        self._env.random_action()  


    def feedback(self, *args, **kwargs):
        '''
        Feedback from the environment
        Recieves a reward and updates understanding
        of an arm

        Keyword arguments:
        ------
        *args -- list of argument
                 0  sender object
                 1. arm index to update
                 2. reward

        *kwards -- dict of keyword arguments:
                   None expected!

        '''
        arm_index = args[1]
        reward = args[2]
        self._total_reward += reward
        self._actions[arm_index] +=1
        self._means[arm_index] = self.updated_reward_estimate(arm_index, reward)

    def updated_reward_estimate(self, arm_index, reward):
        '''
        Calculate the new running average of the arm

        Keyword arguments:
        ------
        arm_index -- int, index of the array to update
        reward -- float, reward recieved from the last action

        Returns:
        ------
        float, the new mean estimate for the selected arm
        '''
        n = self._actions[arm_index]
        current_value = self._means[arm_index]
        new_value = ((n - 1) / float(n)) * current_value + (1 / float(n)) * reward
        return new_value

    total_reward = property(_get_total_reward)
    actions = property(_get_action_history)



class AnnealingEpsilonGreedy(EpsilonGreedy):
    '''
    Encapsulates an Annealing Epsilon-Greedy based agent for reinforcement learning
    in multi-arm bandit problems.  The agent interacts and learns from  
    an environment consisting of multiple bandit arms

    Epsilon-Greedy tackles the explore-exploit dilemma by acting greedy
    1 - epsilon of the time and explore epsilon of the time. 

    A potential drawback of the classical epsilon-greedy algorithm is that
    epsilon is a constant and a hyper-parameter that cannot be known in advance.
    The annealing version of the algorithm starts with a large value of epsilon
    and gradually reduces its value over time.  This means that later experiments
    are more likely to exploit than predict.  

    Note: Implements the observer pattern to recieve feedback from the 
    environment

    Dev notes: Inherits from EpsilonGreedy - this could and probably should be 
    set up by composition.  The downside is that a user has to compose the object
    each time they wish to use it.  This keeps parameterisation lower.  

    Public properties:
    -------
    total_reward -- float, the cumulative reward recieved from the environment
    actions -- np.ndarry (vector), a record of the actions (arms) that have been 
               taken by the agent.
            
    Public methods:
    --------
    solve() -- initiates the algorithm for the budget specified
    feedback() -- observer pattern notification method.  This is called by the
                  environment when the result of an action is ready to be reported.

    '''

    def __init__(self, budget, environment):
        '''
        Constructor method for AnnealingEpsilonGreedy
        '''
        environment.register_observer(self)
        self._env = environment
        self._total_rounds = budget
        self._total_reward = 0
        self._current_round = 0
        self._actions = np.zeros(environment.number_of_arms, np.int32)
        self._means = np.zeros(environment.number_of_arms, np.float64)
        self._epsilon = np.inf
        self._anneal_epsilon()

    def _anneal_epsilon(self):
        '''
        Gradual cooling of epsilon so that exploration
        less likely in later rounds of learning
        '''
        t = np.sum(self._actions) + 1
        self._epsilon = 1 / np.log(t + 1e-6)

    def feedback(self, *args, **kwargs):
        '''
        Feedback from the environment
        Recieves a reward and updates understanding
        of an arm.  
        After each learning cycle it 

        Keyword arguments:
        ------
        *args -- list of argument
                 0  sender object
                 1. arm index to update
                 2. reward

        *kwards -- dict of keyword arguments:
                   None expected!

        '''
        arm_index = args[1]
        reward = args[2]
        self._total_reward += reward
        self._actions[arm_index] +=1
        self._means[arm_index] = self.updated_reward_estimate(arm_index, reward)
        self._anneal_epsilon()



