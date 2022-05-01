# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 17:55:39 2022

@author: paual
"""

import numpy as np

def C_t(T1, T2, t):
    '''
    Returns a matrix C that represents T1 and T2 relaxation during a
    period of time t in addition to the vector D.

    Parameters
    ----------
    T1 : float
        Value in milliseconds of the T1 parameter.
    T2 : float
        Value in milliseconds of the T2 parameter.
    t : float
        Lenght of time in milliseconds during which T1 and T2 relaxation takes place.

    Returns
    -------
    C : numpy.array
        Matrix that represents T1 and T2 relaxation and is used in conjunction
        with vector D to simulate said relaxation during a length of time t.


    '''
    
    C = np.array([[np.exp(-t/T2), 0, 0], 
                  [0, np.exp(-t/T2), 0], 
                  [0, 0, np.exp(-t/T1)]])
    
    return C