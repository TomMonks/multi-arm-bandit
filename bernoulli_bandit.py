import numpy as np 

class BernoulliBandit(object):

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


    def play(self):
        '''
        Pull the arm on the bernoulli bandit

        Returns:
        -----
        reward -- int with value = 0 when no reward
                  and 1 when the pull results in a win
        '''
        reward = np.random.binomial(1, self._p_success)
        self._total_reward += reward
        self._number_of_plays += 1
        return reward


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
        
    
    def play(self, bandit_index):
        '''
        Play a specific bandit machine 

        Keyword arguments:
        -----
        bandit_index -- int, index of bandit to play 

        Returns
        -----
        Reward from playing bandit with index bandit_index
        '''
        return self._bandits[bandit_index].play()
    

    def play_random(self):
        '''
        Selects a bandit index at random and plays it

        Returns:
        -------
        Reward for playing a random bandit
        '''
        bandit_index = np.random.choice(len(self._bandits))
        return self.play(bandit_index)

    def __iter__(self):
        return self

    def __next__(self):
        self._current_index += 1
        if self._current_index > len(self._bandits):
            raise StopIteration
        else:
            return self._bandits[self._current_index - 1]



def main(budget):
    '''
    Keyword arguments:
    -----
    budget -- int, number of plays
    '''

    bandit_arms = [BernoulliBandit(p_success=0.3),
                   BernoulliBandit(p_success=0.5),
                   BernoulliBandit(p_success=0.7)]

    casino = BernoulliCasino(bandits=bandit_arms)

    for arm in casino:
        print(arm._number_of_plays)
    

if __name__ == '__main__':
    main(budget=100)



