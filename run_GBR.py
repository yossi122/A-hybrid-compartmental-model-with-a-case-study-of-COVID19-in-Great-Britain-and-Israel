# -*- coding: utf-8 -*-
"""
Created on Wed Dec 2021

@author: BIGMATH
"""

import numpy as np
import pandas as pd
from Model import run_model
from help_functions import get_p_e_vec, moving_average, data_handling
import os    
import matplotlib.pyplot as plt
path_parent = os.path.dirname(os.getcwd())
path_data = r'...\Data'
os.chdir(path_data)
Data = pd.read_csv('data_GBR.csv', index_col = 0)
total_pop,flag_wp_sch, flag_hard_lock, vaccines, tests,testing_policy,contact_tracing,true_new_cases,true_new_deaths, true_hosp, flag_international = data_handling(Data)
flag_wp_sch = flag_wp_sch.values
tracing = contact_tracing
policy = testing_policy

###############################################################################

M = 10
reg_sz = 100000
lower, upper = 90000, 110000
reg_sizes = [np.random.randint(lower, upper) for i in range(M)]
kids_perc = .18

true_deaths_ma    = moving_average(true_new_deaths)
true_new_cases_ma = moving_average(true_new_cases)
true_hosp_ma      = moving_average(true_hosp)

simulation_population = np.sum(reg_sizes)
pop_factor = total_pop/simulation_population
test_num = (tests*simulation_population/total_pop).astype(int)
num_vac  = (vaccines*simulation_population/(total_pop)).astype(int)

###############################################################################
repetition_number = 10
num_days = len(Data)

p_s    = 1/14
p_sy_t = 0.5
p_r    = 1/28
p_i    = 0.03
v_eff  = 0.94
p_international = 0.5
p_h_d  = 0.015
p_asy  = .5
p_e_min, p_e_lim, p_e_max = 0.11,0.3,0.3
thr_m  = 12
p_e_vec= get_p_e_vec(flag_hard_lock, p_e_max=p_e_max, p_e_lim = p_e_lim, p_e_min=p_e_min, thr_m = thr_m)
n_I_init, n_E_init = int(.00008*reg_sz)+1, int(.00016*reg_sz)+1
day_init = 30
p_sy_h = 0.15
p_h_r  = 0.11

dH_mtx, dE_mtx, dR_mtx, dD_mtx, dAsy_mtx, dSy_mtx, cH, dPos, cumulativeDet_list = run_model(M, reg_sizes, p_e_vec, p_i, p_r,
                                                                         p_h_d, p_h_r, p_asy, p_sy_h, p_international, v_eff, test_num, testing_policy, num_vac, p_s, contact_tracing,
                                                                         repetition_number, num_days, flag_wp_sch, flag_hard_lock, flag_international ,p_sy_t, day_init,n_I_init,n_E_init,kids_perc)

dDet = np.array([cumulativeDet_list[i+1] - cumulativeDet_list[i] for i in range(len(cumulativeDet_list)-1) ])
dDet_ma = moving_average(dDet)*pop_factor

cH_gbr_ma = np.array([moving_average(cH[i]*pop_factor) for i in range(len(cH))])

dD_mtx_gbr_ma = np.array([moving_average(dD_mtx[i]*pop_factor) for i in range(len(dD_mtx))])


fig, axs = plt.subplots(2, 1, figsize = (10,10))
axs[0].plot(cH_gbr_ma.T, c='b')
axs[0].plot(true_hosp_ma, c='r')
axs[0].set_title('GBR')
axs[0].set_ylim(0,2*np.max(true_hosp_ma))
axs[1].plot(dD_mtx_gbr_ma.T, c='b')
axs[1].plot(true_deaths_ma, c='r')
axs[1].set_ylim(0,2*np.max(true_deaths_ma))
plt.show()
