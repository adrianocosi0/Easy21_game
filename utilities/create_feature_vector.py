import numpy as np
from utilities.special_functions import cartesian_product

ind_range=[range(0,10,3), range(4,13,3)]
cols_range=[range(0,16,3),range(6,22,3)]

def create_feature_vector(s,a,ind_range=ind_range, cols_range=cols_range):
    initial_mat = np.zeros((3,6,2))
    ind_matches=[]
    col_matches=[]
    for el,ser,cont in zip(s,[ind_range,cols_range],[ind_matches,col_matches]):
        for ind,z in enumerate(zip(*ser)):
            n,m=z
            if el in range(n,m):
                cont += [ind]
    all_matches = cartesian_product(np.array(ind_matches), np.array(col_matches))
    all_m_ind, all_m_col = all_matches[:,0], all_matches[:,-1]
    initial_mat[all_m_ind, all_m_col, a] = 1
    return initial_mat.ravel()
