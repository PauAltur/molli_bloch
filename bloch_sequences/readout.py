# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 19:37:29 2022

@author: paual
"""

import numpy as np
from molli_bloch.bloch_matrices import A_mat, B_vec

def readout(M_pre, T1, T2, B1, TE, TR, alfa = 35):
    '''
    Returns the net magnetization vector M_post which is the version of 
    the net magnetization vector M_pre after applying a readout sequence to it
    with specific TE and TR values.

    Parameters
    ----------
    M_pre : numpy.array
        The net magnetization vector after a readout.
    T1 : float
        Value in milliseconds of the T1 parameter.
    T2 : float
        Value in milliseconds of the T2 parameter.
    B1 : float
        The uncertainty parameter of the angle by which we invert the net 
        magnetization vector.
    TE : float
        Echo time of the acquisition sequence used to acquire the MOLLI 
        sequence.
    TR : float
        Repetiton time of the acquisition sequence used to acquire the MOLLI 
        sequence.
    alfa : float, optional
        The angle by which the net magnetization vector will be inverted. The
        default is 35.

    Returns
    -------
    M_post : numpy.array
        The net magnetization vector after a readout.

    '''
    
    A = A_mat(T1, T2, B1, TE, TR, alfa)
    B = B_vec(T1, T2, B1, TE, TR, alfa)
    
    M_post = np.matmul(A, M_pre) + B
    
    return M_post