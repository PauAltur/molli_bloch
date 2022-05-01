# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 19:27:36 2022

@author: paual
"""

import numpy as np
from molli_bloch.bloch_matrices import C_t, D_t

def recovery(M_pre, T1, T2, delta_t):
    '''
    Returns the net magnetization vector M_post which is the recovered 
    version of the net magnetization vector M_pre after a time delta_t
    with specific T1 and T2 values.

    Parameters
    ----------
    M_pre : numpy.array
        The net magnetization vector before T1 and T2 recovery.
    T1 : float
        Value in milliseconds of the T1 parameter.
    T2 : float
        Value in milliseconds of the T2 parameter.
    delta_t : float
        The length of time in milliseconds during which the net magnetization 
        vector is let to recover. Usually it is the elapsed time between two 
        readouts.

    Returns
    -------
    M_post : numpy.array
        The net magnetization vector after T1 and T2 recovery.

    '''
    C = C_t(T1, T2, delta_t)
    D = D_t(C)
    
    M_post = np.matmul(C, M_pre) + D
    
    return M_post