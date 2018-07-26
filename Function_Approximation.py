import numpy as np
from tqdm import tqdm
from os.path import join

from env import IsTerminalState, InitState, Step, actions_idx
from env import range_dealer, range_player
from plot import PrintLambdaMSE, PrintLoss, Print2DFunction
from MC_Control import GetQvalue

epsilon = 0.05
lr = 0.01
debug = False

class Approximator(object):
    def __init__(self):
        self.w = np.zeros(shape=[3, 6, 2])
        
    def UpdateWeights(self, new_w):
        assert new_w.shape == self.w.shape
        self.w += new_w
        
    def EvaluateQValue(self, s, a):
        if IsTerminalState(s):
            assert a is None
            return 0
            
        feature = ToFeature(s, a)
        assert self.w.shape == feature.shape
        return np.sum(feature*self.w)
            
def EpsilonGreedyPolicy(state, epsilon, approximator):
    assert not IsTerminalState(state)
    assert epsilon >= 0 and epsilon <= 1
    # we have posibility = epsilon to act randomly
    use_random = np.random.random() < epsilon
    if use_random:
        # act randomly
        action = np.random.randint(low=0, high=2)
        return action
    else:
        # greedy action = argmax Q(s, a)
        q0 = approximator.EvaluateQValue(state, 0)
        q1 = approximator.EvaluateQValue(state, 1)
        return 0 if q0 > q1 else 1
 
def _GetDealerMask(n):
    m1 = n>=1 and n<=4
    m2 = n>=4 and n<=7
    m3 = n>=7 and n<=10
    mask = np.array([m1, m2, m3])
    return np.where(mask==True)[0]

def _GetPlayerMask(n):
    m1 = n>=1 and n<=6
    m2 = n>=4 and n<=9
    m3 = n>=7 and n<=12
    m4 = n>=10 and n<=15
    m5 = n>=13 and n<=18
    m6 = n>=16 and n<=21
    mask = np.array([m1, m2, m3, m4, m5, m6])
    return np.where(mask==True)[0]

def ToFeature(s, a):
    assert not IsTerminalState(s)
    assert a in actions_idx.keys()
    feature = np.zeros([3, 6, 2])
    mask_dealer = _GetDealerMask(s['sum_dealer'])
    mask_player = _GetPlayerMask(s['sum_player'])
    feature[mask_dealer[:, None], mask_player, a] = 1
    return feature

def _TestFeature():
    s = InitState()
    s['sum_dealer'] = 7
    s['sum_player'] = 17
    a = 0
    f = ToFeature(s, a)
    print(f[:,:,0])

# TD learning
def Update_TDLambda(approximator, eligibility, s1, a1, r2, s2, a2, lmbda):
    assert not IsTerminalState(s1)
    assert a1 is not None
    # no discount
    q1 = approximator.EvaluateQValue(s1, a1)
    q2 = approximator.EvaluateQValue(s2, a2) # q2 is 0 if s2 is a terminal state
    
    error = r2 + q2 - q1
    eligibility *= lmbda
    eligibility += ToFeature(s1, a1)
    w_delta = lr * error * eligibility
    approximator.UpdateWeights(w_delta)

def SARSA_TDLambda(lmbda, iteration, Qstar=None):
    approximator = Approximator()
    mse = []

    gen = range(iteration//10) if debug else tqdm(range(iteration))
    for _ in gen:
        s1 = InitState()
        a1 = EpsilonGreedyPolicy(s1, epsilon, approximator)
        eligibility = np.zeros(shape=[3, 6, 2])
        while True:
            s2, r2 = Step(s1, a1)
            if IsTerminalState(s2):
                Update_TDLambda(approximator, eligibility, s1, a1, r2, s2, None, lmbda)
                break
            a2 = EpsilonGreedyPolicy(s2, epsilon, approximator)
            Update_TDLambda(approximator, eligibility, s1, a1, r2, s2, a2, lmbda)
            s1 = s2
            a1 = a2
        if Qstar is not None:
            Q = Approximator2Q(approximator)
            err = np.mean((Q-Qstar)**2)
            mse.append(err)
    return approximator, mse
    
def Approximator2Q(app):
    Q = np.zeros(shape=[10, 21, 2])
    for num_dealer in range(10):
        for num_player in range(21):
            s = InitState()
            s['sum_dealer'] = num_dealer + 1
            s['sum_player'] = num_player + 1

            for a in range(2):
                Q[num_dealer, num_player, a] = app.EvaluateQValue(s, a)
    return Q
    
def Section4Question1(Qstar):
    list_lmbda = list(np.arange(0, 1, 0.1))
    list_mse = []
    for lmbda in list_lmbda:
        app, _ = SARSA_TDLambda(lmbda, iteration=30000)
        Q = Approximator2Q(app)
        mse = np.mean((Q-Qstar)**2)
        list_mse.append(mse)
        
    name_mse = 'MSE loss with Function Approximation'
    path_loss = join('results', name_mse + '.png')
    PrintLambdaMSE(list_lmbda, list_mse, title=name_mse, path=path_loss) 
    
def Section4Question2(Qstar):
    app0, mse_0 = SARSA_TDLambda(lmbda=0, iteration=3000, Qstar=Qstar)
    app1, mse_1 = SARSA_TDLambda(lmbda=1, iteration=3000, Qstar=Qstar)
    Q0 = Approximator2Q(app0)
    Q1 = Approximator2Q(app1)
    V0 = np.max(Q0, axis=-1)
    V1 = np.max(Q1, axis=-1)
    
    name_loss = 'training loss with Function Approximation'
    path_loss = join('results', name_loss + '.png')
    PrintLoss([mse_0, mse_1], tags=['lambda=0', 'lambda=1'], title=name_loss, path=path_loss)
    
    name_Q0 = 'Q value with Function Approximation (lambda=0)'
    path_Q0 = join('results', name_Q0 + '.png')
    name_Q1 = 'Q value with Function Approximation (lambda=1)'
    path_Q1 = join('results', name_Q1 + '.png')
    Print2DFunction(V0, range_dealer, range_player, title=name_Q0, path=path_Q0)
    Print2DFunction(V1, range_dealer, range_player, title=name_Q1, path=path_Q1)
    
if __name__=='__main__':
    # Qstar = GetQvalue()
    Section4Question1(Qstar)
    Section4Question2(Qstar)
