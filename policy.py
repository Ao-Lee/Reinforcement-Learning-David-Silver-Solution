import numpy as np
from env import StateToCoord
min_epsilon = 0.05

def EpsilonGreedyPolicy(state, epsilon, Q):
    assert state['winner'] is None
    assert epsilon >= 0 and epsilon <= 1
    # we have posibility = epsilon to act randomly
    use_random = np.random.random() < epsilon
    if use_random:
        # act randomly
        return np.random.randint(low=0, high=2)
    else:
        # greedy action = argmax Q(s, a)
        x, y = StateToCoord(state)
        return np.argmax(Q[x, y ,:])
        
def MyPolicy(state, Q, history, n0=100):
    x, y = StateToCoord(state)
    visit_time = np.sum(history[x, y, :])
    epsilon = max(min_epsilon, n0 / (n0 + visit_time))
    return EpsilonGreedyPolicy(state, epsilon, Q)