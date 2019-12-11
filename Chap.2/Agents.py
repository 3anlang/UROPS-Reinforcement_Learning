import numpy as np
np.random.seed(31)

class Agent:
    def __init__(self, environment):
        self.envir = environment
        self.time = 1
        self.visit = np.zeros(environment.arms)
        self.avg_reward = 0.0
        self.optimal_rate = 1.0
        
    def update(self, action, reward, optimal):
        self.avg_reward += 1.0/self.time*(reward - self.avg_reward)
        self.optimal_rate += 1.0/self.time*((action == optimal)-self.optimal_rate)
        self.time += 1


class SA_agent(Agent):
    def __init__(self, environment, Q = np.ones(environment.arms)):
        super.__init__(environment)
        self.Q = Q
        
    def act(self, environment = self.envir):
        action = self.Q.argmax()
        reward, optimal = environment.feedback(action)
        self.visit[action] += 1
        self.Q[action] += 1.0/self.visit[action]*(reward - self.Q[action])
        super().update(action, reward, optimal)
        return self.avg_reward, self.optimal_rate, action, optimal

class Exp_agent(Agent):
    def __init__(self, environment, Q = np.ones(environment.arms), alpha = 0.7):
        self.Q = Q
        self.alpha = alpha
        super().__init__(environment)
        
    def act(self, environment = self.envir):
        action = self.Q.argmax()
        reward, optimal = environment.feedback(action)
        self.Q[action] += self.alpha*(reward - self.Q[action])
        super().update(action, reward, optimal)
        return self.avg_reward, self.optimal_rate, action, optimal
    
class Epsilon_greedy_agent(Agent):
    def __init__(self, environment, Q = np.zeros(environment.arms), epsilon = 0.05):
        self.Q = Q
        self.epsilon = epsilon
        super().__init__(environment)
        
    def act(self, environment = self.envir):
        action = self.Q.argmax() if np.random.uniform(0,1) > self.epsilon else np.randint(len(self.Q))
        reward, optimal = environment.feedback(action)
        self.visit[action] += 1
        self.Q[action] += 1.0/self.visit[action]*(reward - self.Q[action])
        super().update(action, reward, optimal)
        return self.avg_reward, self.optimal_rate, action, optimal

class UCB_agent:
    def __init__(self, environment, Q = np.zeros(environment.arms), c = 2):
        self.Q = Q
        self.c = c
        super().__init__(environment)
        
    def act(self, environment = self.envir):
        zeros = np.where(self.visit ==0)
        if len(zeros) != 0:
            action = zeros[0]
        else:
            action = (self.Q + c*np.sqrt(np.log(self.time)/self.visit)).argmax()
        reward, optimal = environment.feedback(action)
        self.visit[action] += 1
        self.Q[action] += 1.0/self.visit[action]*(reward - self.Q[action])
        super().update(action, reward, optimal)
        return self.avg_reward, self.optimal_rate, action, optimal
    
class gradient_agent:
    def __init__(self, environment, alpha=2):
        self.H = np.zeros(environment.arms)
        self.pi = np.ones(environment.arms)/(1.0*environment.arms)
        self.alpha = alpha
        super().update(action, reward, optimal)
        
    def act(self, environment = self.envir):
        action = np.random.choice(np.arange(len(self.H)), p=self.pi)
        reward, optimal = environment.feedback(action)
        a = np.arange(len(self.H)) == action
        self.H += a*self.alpha*(reward - self.avg_reward)*(1-self.pi) \
        - (1-a)*self.alpha*(reward - self.avg_reward)*self.pi
        self.pi = np.exp(self.H)
        self.pi /= self.pi.sum()
        super().update(action, reward, optimal)
        return self.avg_reward, self.optimal_rate, action, optimal
    