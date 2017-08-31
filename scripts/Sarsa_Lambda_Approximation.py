import sys
import os
import dotenv
dotenv.load_dotenv('.env')
from utilities.matrix_dimensions import useful_variables
from utilities.moves import *
from utilities.create_feature_vector import *
from utilities.special_functions import inv0
from utilities.plot_surface_decision_bound import plot_Q, plot_MSE_lambda, plot_MSE_episodes
from utilities.create_feature_vector import *
import pickle as pkl

def temporal_difference_SARSA_LAMBDA_approximation(initial_guess=np.zeros(36),
                                    lamb=0, loops=1000, N0=100, record=False):
    act_to_ind = {0:'hit',1:'stick'}
    param_vector = np.random.randn(36)/100
    state = [np.random.randint(0,10),np.random.randint(0,21)]
    E = np.zeros_like(initial_guess)
    action = np.random.randint(0,2)
    k=0
    feat_vect = create_feature_vector(state,action)  #36 entries vector of zeros and ones
    Q_x = param_vector @ feat_vect #scalar, approximation of action-value function at that state,action pair

    e=0.05 #exploration probability
    alpha=0.01 #step size
    estimated_Q = np.empty((10,21,2))

    if record:
        import pickle as pkl
        MSE_s = []
        with open('final_data/Q_true.pkl','rb') as f:
            Q_true = pkl.load(f)

    while k<loops:
        '''In each of these loops we are playing an episode
           The loop starts with a state and action already selected (in previous episode or transition)'''
        #print(state,action,feat_vect)
        E += feat_vect #update elegibility traces

        state_next,r,terminal = transition(state,act_to_ind[action])

        if terminal:
            '''We are at a terminal state'''
            td_error=r-Q_x
            param_vector += alpha*td_error*E
            #print(state, action, param_vector)

            state = [np.random.randint(0,10),np.random.randint(0,21)]
            E=np.zeros_like(initial_guess)
            action = np.random.randint(0,2)

            k+=1

            if record and k%50==0:
                for ind,x in np.ndenumerate(estimated_Q):
                    feat_vec = create_feature_vector(list(ind[:-1]),ind[-1])
                    estimated_Q[ind] = param_vector @ feat_vec
                MSE_s += [np.sum(np.square(estimated_Q-Q_true))/Q_true.size]
            continue

        action_next = pick_an_action_approximate_function(param_vector,state_next,e)
        feat_vect_next = create_feature_vector(state_next,action_next)

        Q_y = param_vector @ feat_vect_next
        td_error = r+Q_y-Q_x

        param_vector += E*alpha*td_error

        E=E*lamb
        state=state_next
        action=action_next
        Q_x = Q_y
        feat_vect = feat_vect_next

    for ind,x in np.ndenumerate(estimated_Q):
        feat_vec = create_feature_vector(list(ind[:-1]),ind[-1])
        estimated_Q[ind] = param_vector @ feat_vec

    if record:
        return estimated_Q, MSE_s

    return estimated_Q

if __name__=='__main__':
    FINAL_DATA_DIR = os.environ.get('FINAL_DATA_DIR')
    if not os.path.exists(os.path.join(FINAL_DATA_DIR,'Q_true.pkl')):
        sys.exit('Run Monte Carlo script before to save true Q')
    with open(os.path.join(FINAL_DATA_DIR,'Q_true.pkl'),'rb') as f:
        Q_true=pkl.load(f)
    # fig_1 = plot_MSE_lambda(temporal_difference_SARSA_LAMBDA_approximation,
    #                         np.arange(0,1.1,0.1), Q_true,
    #                         list(np.arange(1000,6000,2000)),
    #                         None)

    fig_2 = plot_MSE_episodes(temporal_difference_SARSA_LAMBDA_approximation,
                              10000,list(np.linspace(0,1,4)),record=True,
                              initial_guess=np.zeros(36))
