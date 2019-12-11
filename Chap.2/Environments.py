import numpy as np
np.random.seed(41)

class Stat_Environment:
    def __init__(self, arms = 10, err = 1):
        self.arms = arms
        self.err = err
        self.q = np.random.normal(0,self.err,size = self.arms)
    
    def feedback(self, action):
        reward = self.q + np.random.normal(0,1.0,size = self.arms)
        return reward[action], reward.argmax()
    
class Non_Stat_Environment:
    def __init__(self, arms = 10, err = 1, change_rate=0.3):
        self.arms = arms
        self.err = err
        self.change_rate = change_rate
        self.q = np.random.normal(0,self.err,size = self.arms)
    
    def feedback(self, action):
        reward = self.q + np.random.normal(0,1.0,size = self.arms)
        self.q += np.random.normal(0, self.change_rate, size = self.arms)
        return reward[action], reward.argmax()
    
class Random_Walk_Environment:
    def __init__(self, arms = 10):
        self.arms = arms
        self.change_rate = change_rate
        self.q = np.zeros(10)
    
    def feedback(self, action):
        reward = self.q + np.random.normal(0,1.0,size = self.arms)
        self.q += np.random.randint(2,size=10)*2-1
        return reward[action], reward.argmax()