import numpy as np
from tqdm import tqdm

from env import InitState, Step, StateToCoord, IsTerminalState
from env import range_dealer, range_player
from plot import Print2DFunction
from policy import MyPolicy
n0=20
debug = False
iteration = 500000

def GenerateEpisode(Q, history):
    # since discount factor is 1
    # and immediate return is always 0 unless the game terminates
    # each return value of intermediate step is equal to the reward of the final state
    # this function will generate [<s1, a1>, <s2, a2>, ..., <sn, an>], final_reward 
    episode = []
    final_reward = 0
    state = InitState()
    action = MyPolicy(state, Q, history, n0=n0)
    episode.append((state, action))
    while True:
        state, reward = Step(state, action)
        if IsTerminalState(state):
            final_reward = reward
            action = None
            episode.append((state, action))
            break
        else:
            action = MyPolicy(state, Q, history)
            episode.append((state, action))
            
    return episode, final_reward

# Monte-Carlo learning
def UpdateQ_MC(Q, simulation, history):
    episode, final_reward = simulation
    for s, a in episode:
        if IsTerminalState(s):
            continue
        
        x, y = StateToCoord(s)
        history[x, y, a] += 1
        # learning rate
        lr = 1 / history[x, y, a]
        Q[x, y, a] += lr * (final_reward - Q[x, y, a])
        if debug and y == 20 and x == 5 and a== 0:
            visit_time = np.sum(history[x, y, :])
            epsilon = n0 / (n0 + visit_time)
            print('Q: {:.2f}\t return: {}\t lr: {:.2f} \t epsilon: {:.2f}'.format(Q[x, y, a], final_reward, lr, epsilon))

def OptimizeQValue(Q):
    # count of visit times for each state <sum_dealer, sum_player, action>
    history = np.zeros(shape=[10, 21, 2])
    if debug:
        generator = range(iteration)
    else:
        generator = tqdm(range(iteration))
    for _ in generator:
        simulation = GenerateEpisode(Q, history)
        UpdateQ_MC(Q, simulation, history)
    return Q
        
def GetQvalue():
    Q = np.zeros(shape=[10, 21, 2])
    Q = OptimizeQValue(Q)
    return Q
   
def Section2_Monte_Carlo_Control():
    Q = GetQvalue()
    V = np.max(Q, axis=-1)
    Print2DFunction(V, range_dealer, range_player)
if __name__=='__main__':
    Section2_Monte_Carlo_Control()

    