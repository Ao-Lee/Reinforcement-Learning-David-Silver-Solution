import numpy as np
actions = {'stick':0, 'hit':1}
actions_idx = {actions[k]:k for k in actions.keys()}
range_dealer = range(1, 11)
range_player = range(1, 22)

def DrawCard(must_be_black=False):
    value = np.random.randint(low=1, high=11)
    is_black = np.random.random() > 0.25
    if must_be_black: is_black = True
    score = value if is_black else -value
    return score
    
def InitState():
    result = {}
    result['sum_player'] = DrawCard(must_be_black=True)
    result['sum_dealer'] = DrawCard(must_be_black=True)
    result['winner'] = None
    return result

def _StepHit(s1):
    s2 = s1.copy()
    s2['sum_player'] += DrawCard()
    if s2['sum_player'] > 21 or s2['sum_player'] < 1:
        s2['winner'] = 'dealer'
        r2 = -1
    else:
        r2 = 0
    return s2, r2
        
def _StepStick(s1):
    s2 = s1.copy()
    while s2['sum_dealer'] < 17:
        s2['sum_dealer'] += DrawCard()
        if s2['sum_dealer'] > 21 or s2['sum_dealer'] < 1:
            r2 = 1
            s2['winner'] = 'player'
            return s2, r2
    
    assert s2['sum_dealer'] <= 21 and s2['sum_dealer'] >=1
    assert s2['sum_player'] <= 21 and s2['sum_player'] >=1
    if s2['sum_player'] > s2['sum_dealer']:
        r2 = 1
        s2['winner'] = 'player'
    elif s2['sum_player'] < s2['sum_dealer']:
        r2 = -1
        s2['winner'] = 'dealer'
    else:
        r2 = 0
        s2['winner'] = 'draw'
    return s2, r2
    
def Step(s1, a1):
    assert a1 in actions.values()
    if a1 == 1: # action hit
        s2, r2 = _StepHit(s1)
        return s2, r2
    if a1 == 0: # action stick
        s2, r2 = _StepStick(s1)
        return s2, r2
        
def IsTerminalState(s):
    return s['winner'] is not None

def StateToCoord(state):
    assert not IsTerminalState(state)
    x = state['sum_dealer'] - 1
    y = state['sum_player'] - 1
    return x, y
    