import os
import dotenv
dotenv.load_dotenv('.env')
from utilities.matrix_dimensions import useful_variables
from utilities.moves import *
from utilities.plot_surface_decision_bound import plot_Q
import pickle as pkl

def temporal_difference_Monte_Q(initial_guess, loops, N0=100):
    ind_to_act = {0:'hit',1:'stick'}
    Q = initial_guess
    N_ep=np.zeros_like(initial_guess)
    state = [np.random.randint(0,10),np.random.randint(0,21)]
    states_actions=[]
    k=0

    while k<loops:
        '''In each of these loops we are playing an episode
           The loop starts with a state already selected (in previous episode or transition)'''

        e = N0/(N0+N_ep[state[0],state[1]].sum()+1)
        action = pick_an_action(Q,state,e)
        N_ep[state[0],state[1],action]+=1

        states_actions+=[(state,action)]

        state_next,r,terminal = transition(state,ind_to_act[action])

        if terminal:
            '''We are in a terminal state'''
            for st,act in states_actions:
                alpha=1/N_ep[st[0],st[1],act]
                Q[st[0],st[1],act] += alpha*(r-Q[st[0],st[1],act])

            state = [np.random.randint(0,10),np.random.randint(0,21)]
            states_actions=[]

            k+=1
            continue

        state=state_next

    return Q

if __name__=='__main__':
        FINAL_DATA_DIR = os.environ.get('FINAL_DATA_DIR')
        if os.path.exists(os.path.join(FINAL_DATA_DIR,'Q_true.pkl')):
            with open(os.path.join(FINAL_DATA_DIR,'Q_true.pkl'),'rb') as f:
                Q_true=pkl.load(f)
        else:
            Q_true=temporal_difference_Monte_Q(useful_variables['initial_guess_Q'],
                                               300000)
            with open(os.path.join(FINAL_DATA_DIR,'Q_true.pkl'),'wb') as w:
                pkl.dump(Q_true, w)
        fig = plot_Q(Q_true)
