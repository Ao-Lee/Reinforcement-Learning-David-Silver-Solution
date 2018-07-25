import numpy as np
from tqdm import tqdm

from env import InitState, Step, StateToCoord, IsTerminalState
from env import range_dealer, range_player
from plot import PrintLambdaMSE, PrintLoss, Print2DFunction
from policy import MyPolicy
from MC_Control import GetQvalue
n0=20
debug = False

def GetQValue(Q, s, a):
    if IsTerminalState(s):
        assert a is None
        return 0

    x, y = StateToCoord(s)
    return Q[x, y, a]

# TD learning
def Update_TDLambda(Q, eligibility, history, s1, a1, r2, s2, a2, lmbda):
    assert not IsTerminalState(s1)
    assert a1 is not None
    x1, y1 = StateToCoord(s1)
    history[x1, y1, a1] += 1
    # learning rate
    lr = 1 / history[x1, y1, a1]
    # no discount
    q1 = GetQValue(Q, s1, a1)
    q2 = GetQValue(Q, s2, a2) # q2 is 0 if s2 is a terminal state
    eligibility[x1, y1, a1] += 1
    error = r2 + q2 - q1
    Q += lr * error * eligibility
    eligibility *= lmbda
    '''
    if debug and y1 == 20 and x1 == 5 and a1 == 0:
        visit_time = np.sum(history[x1, y1, :])
        epsilon = n0 / (n0 + visit_time)
        print('Q: {:.2f}\t return: {}\t lr: {:.2f} \t epsilon: {:.2f}'.format(Q[x1, y1, a1], r2, lr, epsilon))
    '''   
        
def SARSA_TDLambda(lmbda, Qstar=None):
    Q = np.zeros(shape=[10, 21, 2])
    history = np.zeros(shape=[10, 21, 2])
    mse = []

    gen = range(100000) if debug else tqdm(range(100000))
    for _ in gen:
        s1 = InitState()
        a1 = MyPolicy(s1, Q, history, n0=n0)
        eligibility = np.zeros(shape=[10, 21, 2])
        while True:
            s2, r2 = Step(s1, a1)
            if IsTerminalState(s2):
                Update_TDLambda(Q, eligibility, history, s1, a1, r2, s2, None, lmbda)
                break
            a2 = MyPolicy(s2, Q, history, n0=n0)
            Update_TDLambda(Q, eligibility, history, s1, a1, r2, s2, a2, lmbda)
            s1 = s2
            a1 = a2
        if Qstar is not None:
            err = np.mean((Q-Qstar)**2)
            mse.append(err)
    return Q, mse
   
def Section3Question1(Qstar):
    list_lmbda = list(np.arange(0, 1, 0.1))
    list_mse = []
    for lmbda in list_lmbda:
        Q, _ = SARSA_TDLambda(lmbda)
        mse = np.mean((Q-Qstar)**2)
        list_mse.append(mse)
    PrintLambdaMSE(list_lmbda, list_mse) 
    
def Section3Question2(Qstar):
    Q0, mse_0 = SARSA_TDLambda(lmbda=0, Qstar=Qstar)
    Q1, mse_1 = SARSA_TDLambda(lmbda=1, Qstar=Qstar)
    V0 = np.max(Q0, axis=-1)
    V1 = np.max(Q1, axis=-1)
    PrintLoss([mse_0, mse_1], tags=['lambda=0', 'lambda=1'])
    Print2DFunction(V0, range_dealer, range_player, title='V* for lambda=0')
    Print2DFunction(V1, range_dealer, range_player, title='V* for lambda=1')
    
if __name__=='__main__':
    Qstar = GetQvalue()
    Section3Question1(Qstar)
    Section3Question2(Qstar)

    