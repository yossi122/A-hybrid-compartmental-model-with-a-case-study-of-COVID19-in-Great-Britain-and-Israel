# -*- coding: utf-8 -*-
"""
Created on Dec 2021

@author: BIGMATH

"""
import networkx as nx
import numpy as np

def create_networks(size,kids_perc,k=4):
    '''
    Creates the underlying population graphs for different scenarios.
    Parameters:
        size - int; population size
        kids - int; percentage of population that children make
           k - int; average degree of a graph.
    '''
    population = size

    L_hard_lock = nx.watts_strogatz_graph(n = population, k = k, p = 0.0) # (Almost) regular lattice
    
    # Social distancing graphs, with different factors of (non-)compliance s:
    L_s = nx.watts_strogatz_graph(n = population, k = k, p = 0.007)
    L_s = nx.compose(L_hard_lock,L_s)
    L_m = nx.watts_strogatz_graph(n = population, k = k, p = 0.013)
    L_m = nx.compose(L_hard_lock,L_m)
    L_l = nx.watts_strogatz_graph(n = population, k = k, p = 0.027)
    L_l = nx.compose(L_hard_lock,L_l)

    R_s = nx.fast_gnp_random_graph(n = population, p = 1/(population-1))
    R_s = nx.compose(R_s,L_hard_lock)
    R_m = nx.fast_gnp_random_graph(n = population, p = 2/(population-1))
    R_m = nx.compose(R_m,L_hard_lock)
    R_l = nx.fast_gnp_random_graph(n = population, p = 3/(population-1))
    R_l = nx.compose(R_l,L_hard_lock)

    kids_all_schools = list(np.random.choice(L_hard_lock.nodes(), int(size*kids_perc)))

    return L_hard_lock, L_s, L_m, L_l, R_s ,R_m, R_l, kids_all_schools