# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 17:55:39 2022

@author: paual
"""

import numpy as np

def D_t(C, m_0 = 1):
    '''
    Returns a vector D that represents T1 and T2 relaxation during a
    period of time t in addition to the matrix C.

    Parameters
    ----------
    C : numpy.array
        Matrix that represents T1 and T2 relaxation during a
        period of time t.
    m_0 : float, optional
        The z component of the initial state of the net magnetization vector.
        The default is 1.

    Returns
    -------
    D : numpy.array
        Vector that represents T1 and T2 relaxation and is used in conjunction
        with matrix C to simulate said relaxation during a length of time t.

    '''
    
    D = np.matmul((np.eye(3) - C) , np.array([0, 0, m_0]))
    return D
