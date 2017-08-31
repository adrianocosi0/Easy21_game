import sys
import os
import dotenv
dotenv.load_dotenv('.env')
from utilities.matrix_dimensions import useful_variables
from utilities.moves import *
from utilities.special_functions import inv0
from utilities.plot_surface_decision_bound import plot_Q, plot_MSE_lambda, plot_MSE_episodes
import pickle as pkl

def temporal_difference_SARSA_LAMBDA(initial_guess,
                                    lamb=0, loops=1000, N0=100, record=False):
    act_to_ind = {0:'hit',1:'stick'}
    Q_i = initial_guess
    N_ep=np.zeros_like(initial_guess)
    state = [np.random.randint(0,10),np.random.randint(0,21)]
    E = np.zeros_like(initial_guess)
    action = np.random.randint(0,2)
    k=0
    if record:
        import pickle as pkl
        MSE_s = []
        with open('final_data/Q_true.pkl','rb') as f:
            Q_true = pkl.load(f)

    while k<loops:
        '''In each of these loops we are playing an episode
           The loop starts with a state and action already selected (in previous episode or transition)'''
        N_ep[state[0],state[1],action]+=1
        e = N0/(N0+N_ep[state[0],state[1]].sum())

        alpha=inv0(N_ep)

        E[state[0],state[1],action]+=1

        state_next,r,terminal = transition(state,act_to_ind[action])

        if terminal:
            '''We are at a terminal state'''
            td_error=r-Q_i[state[0],state[1],action]
            Q_i += alpha*td_error*E

            state = [np.random.randint(0,10),np.random.randint(0,21)]
            E=np.zeros_like(initial_guess)
            action = np.random.randint(0,2)

            k+=1

            if record and k%50==0:
                MSE_s += [np.sum(np.square(Q_i-Q_true))/Q_true.size]
            continue

        action_next = pick_an_action(Q_i,state_next,e)

        td_error = r+Q_i[state_next[0],state_next[1],action_next]-Q_i[state[0],state[1],action]
        Q_i += E*alpha*td_error

        E=E*lamb
        state=state_next
        action=action_next

    if record:
        return Q_i, MSE_s

    return Q_i

if __name__=='__main__':
    FINAL_DATA_DIR = os.environ.get('FINAL_DATA_DIR')
    print(FINAL_DATA_DIR)
    if not os.path.exists(os.path.join(FINAL_DATA_DIR,'Q_true.pkl')):
        sys.exit('Run Monte Carlo script before to save true Q')
    with open(os.path.join(FINAL_DATA_DIR,'Q_true.pkl'),'rb') as f:
        Q_true=pkl.load(f)
    initial_guess_Q=useful_variables['initial_guess_Q']
    fig_1 = plot_MSE_lambda(temporal_difference_SARSA_LAMBDA,
                            np.arange(0,1.1,0.1), Q_true,
                            list(np.arange(1000,6000,2000)),
                            [50,75,100],
                            initial_guess=initial_guess_Q)

    fig_2 = plot_MSE_episodes(temporal_difference_SARSA_LAMBDA,
                              40000,list(np.linspace(0,1,4)),record=True,
                              initial_guess=initial_guess_Q)
