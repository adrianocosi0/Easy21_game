import numpy as np
from operator import add, sub

states_matrix_shape = (10,21)
rew_matrix_shape = (21,21)
trans_matrix = np.ones(states_matrix_shape[0])*0.1
x_arr,y_arr = np.meshgrid(np.arange(0,21),np.arange(0,21))[0],np.meshgrid(np.arange(0,21),np.arange(0,21))[1]
rew_matrix = (x_arr>y_arr).astype(int)
ind = np.tril_indices(21,-1)
rew_matrix[ind]=-1

possible_events=list(zip(list(range(1,11))+list(range(1,11)),[add]*10+[sub]*10))

p_s=np.empty(len(possible_events))
p_s[:10]=0.1*2/3
p_s[10:]=0.1*1/3

initial_guess_Q=np.zeros((10,21,2))


useful_variables = {'rew_matrix':rew_matrix,'possible_events':possible_events,
                    'p_s':p_s,'initial_guess_Q':initial_guess_Q
                    }
