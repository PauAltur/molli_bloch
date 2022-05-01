# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 17:58:29 2022

@author: paual
"""

import numpy as np
from molli_bloch.bloch_sequences.bloch_matrices.R_alfa import R_alfa
from molli_bloch.bloch_sequences.bloch_matrices.P_t import P_t
from molli_bloch.bloch_sequences.bloch_matrices.C_t import C_t


def A_mat(T1, T2, B1, TE, TR, alfa):
    '''
    Returns matrix A as defined in Hargreaves BA, Vasanawala SS, Pauly JM, 
    Nishimura DG. Characterization and reduction of the transient response in 
    steady-state MR imaging. Magn Reson Med. 2001 Jul;46(1):149-58. 
    doi: 10.1002/mrm.1170. PMID: 11443721. It is used to compute the net 
    magnetization vector M_k+1 from the net magnetitzation vector M_k.

    Parameters
    ----------
    T1 : float
        Value in milliseconds of the T1 parameter (spin-lattice relaxation).
    T2 : float
        Value in milliseconds of the T2 parameter (spin-spin relaxation).
    B1 : float
        The uncertainty parameter of the angle by which we invert the net 
        magnetization vector. The default is 1.
    TE : float
        Echo time of the acquisition sequence used to acquire the MOLLI 
        sequence.
    TR : float
        Repetiton time of the acquisition sequence used to acquire the MOLLI 
        sequence.
    alfa : float
        The angle by which the net magnetization vector will be inverted.

    Returns
    -------
    A : numpy.array
        The matrix A that is multiplied by M_k to obtain M_k+1.

    '''
    
    P1 = P_t(TE)
    P2 = P_t(TR - TE)
    
    C1 = C_t(T1, T2, TE)
    C2 = C_t(T1, T2, TR - TE)
    
    R_a = R_alfa(B1, alfa)
    
    A = np.matmul(np.matmul(np.matmul(np.matmul(P1, C1), R_a), P2), C2)
    
    return A
    