# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 12:27:23 2022

@author: paual
"""

import numpy as np
import pandas as pd
from molli_bloch.bloch_sequences import simulate_MOLLI
from datetime import datetime
from tqdm import tqdm

def generate_MOLLI_dataset(n_samples, T1_range, T2_range, B1_range, t_arrays, component = 'z', path_to_save = False, normalization = 'divide_5000', absolute = True):
    '''
    Returns a dataset of synthetic MOLLI sequences of a single net magnetization 
    vector along with the parameters that define the magnetization of that vector
    (T1 and T2) and the scanner's interaction with it (B1).

    Parameters
    ----------
    n_samples : int
        Number of single NMV MOLLI sequences to generate.
    T1_range : tuple, list
        Range of T1 values, in milliseconds, that will be used to generate the dataset.
    T2_range : tuple, list
        Range of T2 values, in milliseconds, that will be used to generate the dataset.
    B1_range : tuple, list
        Range of  B1 values that will be used to generate the dataset.    
    t_arrays : pandas.dataframe
        Dataframe of time arrays defining the inversion times at which each slice 
        was acquired.
    component : string -> optional
        Variable that determines which component of the net magnetization vector
        will be saved in the dataset, y or z. The default is 'z'.
    path_to_save : bool, string -> optional 
        Variable that determines whether the generated dataset is saved or not, and
        where should it be saved. The default is False.
    normalization : string -> optional
        Variable that determines the type of normalization that the parameters will 
        undergo. 'divide_5000' divides them by 5000 ms whereas 'min_max' performs
        the minmax normalization of each parameter independently. The default is 
        'divide_5000'.
    absolute : bool -> optional
        Variable that determines whether the generated MOLLI sequences will be in
        absolute value. The default is True.

    Returns
    -------
    mollis : numpy.array
        Array of shape (n_samples, 2, n_acquisitions). It contains the simulated 
        MOLLI sequences in the first row of each sample and its corresonding 
        inversion time array in the second row.
    params : numpy.array
        Array of shape (n_samples, 3). It contains the parameters T1, T2, and B1
        used to simulate each MOLLI sequence. They are ordered as T1, T2, B1 in
        each sample.

    '''
    
    T1_min, T1_max = T1_range
    T2_min, T2_max = T2_range
    B1_min, B1_max = B1_range
    
    T1_rand_array = (T1_max - T1_min) * np.random.random_sample((n_samples,)) + T1_min
    T2_rand_array = (T2_max - T2_min) * np.random.random_sample((n_samples,)) + T2_min
    B1_rand_array = (B1_max - B1_min) * np.random.random_sample((n_samples,)) + B1_min
    
    mollis = np.empty((n_samples, 2, t_arrays.shape[1]))
    params = np.empty((n_samples, 3))
    
    df_dataset = pd.DataFrame(columns = ['t0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 'T1', 'T2', 'B1'])
    
    if absolute:
        type_molli = 0
        
    if not absolute:
        type_molli = 1
        
    for i in tqdm(range(n_samples)):
        
        T_i = t_arrays.sample().values[0]
        
        T1_rand = T1_rand_array[i]  
        T2_rand = T2_rand_array[i]
        B1_rand = B1_rand_array[i] 
        
        if normalization == 'divide_5000':
            params[i, 0] = T1_rand / 5000 # normalization
            params[i, 1] = T2_rand / 5000
            params[i, 2] = B1_rand / 5
            
        elif normalization == 'min_max':
            params[i, 0] = (T1_rand - T1_min) / (T1_max - T1_min) # normalization
            params[i, 1] = (T2_rand - T2_min) / (T2_max - T2_min)
            params[i, 2] = (B1_rand - B1_min) / (B1_max - B1_min)
        
        molli = simulate_MOLLI(T_i, T1_rand, T2_rand, B1_rand)[type_molli]
        if component == 'y':
            mollis[i, 0, :] = molli[1, :]
        elif component == 'z':
            mollis[i, 0, :] = molli[2, :]
        
        mollis[i, 1, :] = T_i / 5000 # normalization
        
        if path_to_save:
        
            datetime_to_save = datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
            
            df_dataset.loc[len(df_dataset)] = mollis[i, 0, :].tolist() + [params[i, 0], params[i, 1], params[i, 2]]
            df_dataset.loc[len(df_dataset)] = mollis[i, 1, :].tolist() + [params[i, 0], params[i, 1], params[i, 2]]
    
    if path_to_save:
        df_dataset.to_csv('{}/MOLLI_dataset_{}_n_{}_T1_{}_{}_T2_{}_{}_B1_{}_{}_abs_{}_comp_{}.csv'.format(path_to_save, datetime_to_save, n_samples, T1_min, T1_max, T2_min, T2_max, B1_min, B1_max, absolute, component), index = False)
        
    return mollis, params