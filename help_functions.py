# -*- coding: utf-8 -*-
"""
Created on Dec 2021

@author: BIGMATH
"""

import matplotlib.pyplot as plt
import numpy as np

def get_p_e_vec(lock, p_e_max, p_e_lim, p_e_min, thr_m):
    '''
    Creates a vector contining p_e values for each day of simulation.
    Parameters:
        p_e_max - float; maximum value of p_e
        p_e_min - float; minimum value of p_e
        p_e_lim - float;
        thr_m   - int; number of months until p_e is expected to reach p_e_max
    '''
    lock = list(lock)
    thr  = thr_m*30
    l1_i = lock.index(1)
    p_e_vec = [p_e_max for i in range(l1_i)]
    lock_active = 1
    free_since  = 0
    for i,l in enumerate(lock):
        if i >= l1_i:
            if l == 1: # still in lockdown
                free_since = 0
                p_e_vec.append(p_e_min)
            else:
                if lock_active == 1:
                    lock_active = 0
                    free_since  = 1
                else:
                    free_since += 1
                aux = (1-free_since/thr)*p_e_min + free_since/thr*p_e_lim
                p_e_vec.append(min(aux, p_e_lim))
    return p_e_vec

def moving_average(arr,window_width=7):
    cumsum_vec = np.cumsum(np.insert(arr, 0, 0)) 
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    return ma_vec

def data_handling(data):    
    df = data.copy()   
    # school work flag
    df['school_closing'] = np.abs(df['school_closing'].values)
    df['workplace_closing'] = np.abs(df['workplace_closing'].values)
    df['internal_movement_restrictions'] = np.abs(df['internal_movement_restrictions'])
    df['stay_home_restrictions'] = np.abs(df['stay_home_restrictions'])
    df['international_movement_restrictions'] = np.abs(df['international_movement_restrictions'])
    
    df.loc[df['school_closing']==2, 'school_closing'] = 1
    df.loc[df['workplace_closing']==2, 'workplace_closing'] = 1
    sch_close = df['school_closing']
    wp_close = df['workplace_closing']    
    wp_international = df['internal_movement_restrictions']    
    flag_international = wp_international.values
    flag_international[flag_international>0]=1
    flag_wp_sch = wp_close.astype(str) + sch_close.astype(str)
    # hard lockdown flag
    stay_home = df['stay_home_restrictions']
    flag_hard_lock = stay_home>1
    flag_hard_lock = flag_hard_lock.values        
    vaccines = df.vaccines_new.values
    tests = df.tests_new.values
    testing_policy = df.testing_policy.values
    contact_tracing = df.contact_tracing.values
    total_pop = df['population'].iloc[0]       
    # ground truth
    df_gt = df[['confirmed_new','deaths_new','hosp']]
    df_gt = df_gt.mask(df_gt.lt(0)).ffill().fillna(0)
    true_new_cases = df_gt['confirmed_new'].values  
    true_new_deaths = df_gt['deaths_new'].values  
    true_hosp = df_gt['hosp'].values          
    return total_pop,flag_wp_sch, flag_hard_lock, vaccines, tests,testing_policy,contact_tracing,true_new_cases,true_new_deaths, true_hosp, flag_international
   
def plot_mean_std(Matrix,pop_factor,ground_truth,title):
    vector_mean = moving_average(Matrix.mean(axis=0)*pop_factor,7)
    vector_std = moving_average(Matrix.std(axis=0)*pop_factor,7)
    plt.figure(figsize=(10,5))
    plt.plot(vector_mean,c='b', label = 'Predicted')
    plt.plot(vector_mean+vector_std, c='b', alpha=.3)
    plt.plot(vector_mean-vector_std, c='b', alpha=.3)
    plt.plot(ground_truth,c='r', label = 'Observed')
    plt.title(title)
    plt.legend()       
    plt.show()
    
def plot_simulations(Matrix,pop_factor,ground_truth,title,repetition_number):
    plt.figure(figsize=(10,5))
    for i in range(repetition_number):
        vector = moving_average(Matrix[i]*pop_factor,7)
        if i == 0:
            plt.plot(vector,c='b', label = 'Predicted')
        else:
            plt.plot(vector,c='b')
    plt.plot(ground_truth,c='r', label = 'Observed')
    plt.title(title)
    plt.legend()       
    plt.show()
    
def plot_the_three(Matrix1,Matrix2,Matrix3,pop_factor,ground_truth,title):
    vector_mean1 = moving_average(Matrix1.mean(axis=0)*pop_factor,7)
    vector_mean2 = moving_average(Matrix2.mean(axis=0)*pop_factor,7)
    vector_mean3 = moving_average(Matrix3.mean(axis=0)*pop_factor,7)
    plt.figure(figsize=(10,5))
    plt.plot(vector_mean1,c='sienna', label = 'V_eff=0.9')
    plt.plot(vector_mean2,c='seagreen', label = 'V_eff=0.75')
    plt.plot(vector_mean3,c='cornflowerblue', label = 'V_eff=0')
    plt.title(title)
    plt.legend()       
    plt.show()
