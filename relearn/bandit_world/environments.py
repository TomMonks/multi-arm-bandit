import numpy as np 

class BernoulliBandit(object):
    '''
    Classic one armed bandit gambling machine.

    A user plays the bandit by pulling its arm.

    The bandit returns a reward of 1 with probability p
    and 0 with probability 1 - p.
    '''

    def __init__(self, p_success):
        '''
        Constructor method for Bernoulli Bandit

        Keyword arguments:
        -----
        p_success -- probability p of drawing a 1 from the 
                     bernoulli distribution

        '''
        self._p_success = p_success
        self._number_of_plays = 0
        self._total_reward = 0
        self._observers = []

    def play(self):
        '''
        Pull the arm on the bernoulli bandit

        Returns:
        -----
        reward -- int with value = 0 when no reward
                  and 1 when the pull results in a win
        '''
        #reward = np.random.binomial(1, self._p_success)
        
        sample = np.random.uniform()

        if sample > self._p_success:
            reward = 0
        else:
            reward = 1

        self._total_reward += reward
        self._number_of_plays += 1

        return reward

    def reset(self):
        '''
        Reset the number of plays and 
        total rewards to zero.
        '''
        self._number_of_plays = 0
        self._total_reward = 0


    def win_proportion(self):
        '''
        The empirical success rate on the bandit

        Returns:
        -----
        float, proportion of plays that have resulted in a win
        '''
        if self._number_of_plays > 0:
            return self._total_reward / self._number_of_plays
        else:
            return 0.0



class BernoulliCasino(object):

    def __init__(self, bandits):
        '''
        Casino constructor method

        Keyword arguments:
        ------
        bandits -- list, of BernoulliBandits objects
        '''
        
        self._bandits = bandits
        self._current_index = 0
        self._observers = []
    
    def __getitem__(self, index):
        return self._bandits[index]

    def __get_number_of_arms(self):
        return len(self._bandits)
    
    def action(self, bandit_index):
        '''
        Play a specific bandit machine.#

        Notifies all observers of the outcome 

        Keyword arguments:
        -----
        bandit_index -- int, index of bandit to play 
        '''
        reward = self._bandits[bandit_index].play()
        self.notify_observers(bandit_index, reward)

    def random_action(self):
        '''
        Selects a bandit index at random and plays it.
        '''
        bandit_index = np.random.choice(len(self._bandits))
        self.action(bandit_index)

       
    def __iter__(self):
        return self

    def __next__(self):
        self._current_index += 1
        if self._current_index > len(self._bandits):
            raise StopIteration
        else:
            return self._bandits[self._current_index - 1]

    def register_observer(self, observer):
        self._observers.append(observer)
 
    def notify_observers(self, *args, **kwargs):
        for observer in self._observers:
            observer.feedback(self, *args, **kwargs) 

    number_of_arms = property(__get_number_of_arms)


def custom_bandit_problem(*means):
    '''
    Creates a list of BernouliBandit objects with
    user specified means

    Keyword arguments:
    ------
    *means - variable size list of means

    Returns:
    ------
    list, BernoulliBandits size = len(means)
    '''
    return [BernoulliBandit(mean) for mean in means]

def standard_bandit_problem():
    '''
    Creates a list of 10 BernouliBandit objects 
    means:
    0.1, 0.7, 0.1, 0.2, 0.3, 0.4, 0.45, 0.6, 0.9, 0.75

    Returns:
    ------
    list, BernoulliBandits size = 10
    '''

    return custom_bandit_problem(0.1, 0.7, 0.1, 0.2, 0.3, 0.4, 
                                 0.45, 0.6, 0.9, 0.75)
    

def small_bandit_problem():
    '''
    Creates a list of 3 BernouliBandit objects 
    means:
    0.3, 0.5, 0.1
    Returns:
    ------
    list, BernoulliBandits size = 10
    '''
    return custom_bandit_problem(0.3, 0.5, 0.1)






