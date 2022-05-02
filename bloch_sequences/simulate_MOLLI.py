# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 20:13:07 2022

@author: paual
"""

import numpy as np
from molli_bloch.bloch_sequences import inversion
from molli_bloch.bloch_sequences import recovery
from molli_bloch.bloch_sequences import readout
from molli_bloch.bloch_times import T_i_to_T_chr


  
def simulate_MOLLI(T_i = np.array((150.0, 200.0, 1150.0, 1200.0, 2150.0, 2200.0, 3150.0, 4150.0)), 
                   T1 = 1180.0, T2 = 60.0, B1 = 1.0, # the variables
                   TE = 1.3, TR = 2.9, # the sequence parameters
                   M_0 = np.array((0,0,1))):
    '''
    Simulates a Molli sequence with a 5(3s)3 pulse sequence.

    Parameters
    ----------
    T_i : numpy.array, optional
        Array of inversion times of each acquisition of the MOLLI sequence ordered
        from high to low. The default is np.array((150.0, 200.0, 1150.0, 1200.0, 
                                                   2150.0, 2200.0, 3150.0, 4150.0)).
    T1 : float, optional
        Value in milliseconds of the T1 parameter. The default is 1180.0.
    T2 : float, optional
        Value in milliseconds of the T2 parameter.The default is 60.0.
    B1 : float, optional
        The uncertainty parameter of the angle by which we invert the net 
        magnetization vector. The default is 1.0.
    TE : float, optional
        DESCRIPTION. The default is 2.5.
    TR : float, optional
        Repetiton time of the acquisition sequence used to acquire the MOLLI 
        sequence. The default is 1.7.
    M_0 : numpy.array, optional
        The rest state of the net magnetization vector. The default is np.array((0,0,1)).

    Returns
    -------
    M_ord_abs : numpy.array
        Array containing the absolute value of its simulated net magnetization 
        vector in its inversion order (i.e sorted by ascending t_i), as it is 
        the shape from which T1 can be estimated.
    M_ord : numpy.array
        Array containing the simulated net magnetization vector in its inversion
        order (i.e sorted by ascending t_i), as it is the shape from which T1 can
        be estimated.
    M_raw : numpy.array
        Array containing the simulated net magnetization vector in its natural
        order (i.e the order in which it is acquired during a MOLLI sequence).
    

    '''
 
    T_rec = T_i_to_T_chr(T_i)
    
    M_raw = np.empty((3, T_rec.shape[0] + 3))
    
    M_raw[:, 0] = M_0
    M_raw[:, 1] = inversion(M_raw[:, 0], B1) # 1st inversion
    
    for i in range(2, 7): # series of 5 readouts
        t_rec = T_rec[i-2]
        M_rec = recovery(M_raw[:, i-1], T1, T2, t_rec)
        M_raw[:, i] = readout(M_rec, T1, T2, B1, TE, TR)
    
    t_rec = T_rec[5]
    M_raw[:, 7] = recovery(M_raw[:, 6], T1, T2, t_rec) # let M recover for 3 hb
    M_raw[:, 8] = inversion(M_raw[:, 7], B1) # 2nd inversion
    
    for i in range(9, 12):
        t_rec = T_rec[i-3]
        M_rec = recovery(M_raw[:, i-1], T1, T2, t_rec)
        M_raw[:, i] = readout(M_rec, T1, T2, B1, TE, TR)
    
    M_ord = M_raw[:, [2, 9, 3, 10, 4, 11, 5, 6]]
    M_ord_abs = np.abs(M_ord)
    
    return M_ord_abs, M_ord, M_raw