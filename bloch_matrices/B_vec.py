# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 18:19:20 2022

@author: paual
"""

import numpy as np
from molli_bloch.bloch_sequences.bloch_matrices.R_alfa import R_alfa
from molli_bloch.bloch_sequences.bloch_matrices.P_t import P_t
from molli_bloch.bloch_sequences.bloch_matrices.C_t import C_t
from molli_bloch.bloch_sequences.bloch_matrices.D_t import D_t

def B_vec(T1, T2, B1, TE, TR, alfa):
    '''
    Returns a vector B as defined in Hargreaves BA, Vasanawala SS, Pauly JM, 
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
        magnetization vector.
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
    B : numpy.array
        The vector B that is added to M_k to obtain M_k+1.
        
    '''
    P1 = P_t(TE)
    
    C1 = C_t(T1, T2, TE)
    C2 = C_t(T1, T2, TR - TE)
    
    R_a = R_alfa(B1, alfa)
    
    D1 = D_t(C1)
    D2 = D_t(C2)
    
    B = np.matmul(np.matmul(np.matmul(P1, C1), R_a), D1) + D2
    
    return B 