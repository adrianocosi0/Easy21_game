import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
from utilities.matrix_dimensions import useful_variables
from utilities.create_feature_vector import *
import numpy as np
from operator import add,sub

possible_events = useful_variables['possible_events']
p_s = useful_variables['p_s']
rew_matrix = useful_variables['rew_matrix']

def pick_a_card(tot,sample=possible_events,p_s=p_s):
    '''sample is a list of tuples, each tuple in it has a card and its colour, where black adds and red subtracts'''
    ind = np.random.choice(range(0,20),size=1,p=p_s)[0]
    card_picked,colour=sample[ind]
    return colour(tot,card_picked)

def transition(state,a,rew_matrix=rew_matrix,stop_when=16):
    '''state is a list where s[0] is card shown by dealer and s[1] is current sum for player'''
    s = [state[0],state[1]]
    r=0
    terminal=True
    if a == 'hit':
        s[1] = pick_a_card(s[1])
        if s[1]>20 or s[1]<0:
            return s,-1,terminal
        terminal=False
    else:
        while s[0]<stop_when:
            s[0]=pick_a_card(s[0])
            if np.sign(s[0])==-1:
                return s,1,terminal
        try:
            r=rew_matrix[s[0],s[1]]
        except IndexError:
            r=1
    return s, r, terminal

def pick_an_action(Q_inp,state,e):
    '''Pass the action_value matrix, the current state and the probability e of randomizing'''
    choice_new = np.random.choice(['max','random'],1,p=[1-e,e])[0]
    if choice_new == 'max':
        action_next = np.argmax(Q_inp[state[0],state[1]])
    else:
        action_next = np.random.randint(0,2)
    return action_next

def pick_an_action_approximate_function(param_vector, state, e):
    choice_new = np.random.choice(['max','random'],1,p=[1-e,e])[0]
    if choice_new == 'max':
        feat_vect_hit = create_feature_vector(state,0)
        Q_hit = param_vector @ feat_vect_hit
        feat_vect_stick = create_feature_vector(state,1)
        Q_stick = param_vector @ feat_vect_stick
        action_next = np.argmax([Q_hit,Q_stick])
    else:
        action_next = np.random.randint(0,2)
    return action_next
