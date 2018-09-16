# -*- coding: utf-8 -*-

import numpy as np
import sys, os
#from libdalton import energy, gradient
import matplotlib.pyplot as plt
    
#x = np.arange(1, 20, 0.1)

#partfun = lambda r: energy.get_e_elst(r,1,1,1,10)
#partfun2 = lambda r: energy.get_e_elst(r,1,1,1,0)
#y = list(map(partfun,x))
#y2 = list(map(partfun2,x))

#partfun = lambda r: gradient.get_g_mag_elst(r,1,1,1,100)
#partfun2 = lambda r: gradient.get_g_mag_elst(r,1,1,1,0)
#y = list(map(partfun,x))
#y2 = list(map(partfun2,x))
#
#plt.plot(x,y)
#plt.plot(x,y2)
#plt.show()

#print(energy.get_e_vdw(120,1,1,200),gradient.get_g_mag_vdw(120,1,1,200)) 

from multiprocessing import Pool
from itertools import combinations, starmap
from time import time, sleep


global max_comb
max_comb = 400

global sleep_time
sleep_time = 3e-5
n_comb = np.math.factorial(max_comb) // (2*np.math.factorial(max_comb-2)) 
wait = sleep_time * n_comb / (max_comb-1)

global real_sleep_time
real_sleep_time = 0.0

def starmap(fun):
    def fun_wrapper(args):
        return fun(*args)
    return fun_wrapper

@starmap
def par_fun(a,b):
    if b == a+1:
        sleep(wait)
        global real_sleep_time
        real_sleep_time += wait
    return a*b

def par_fun2(vec):
    if vec[1] == vec[0]+1:
        sleep(wait)
    return vec[0]*vec[1]

def main():
    global max_comb
    global sleep_time
    global real_sleep_time
    
    n_comb = np.math.factorial(max_comb) // (2*np.math.factorial(max_comb-2)) 
    out = np.zeros(n_comb)
    out_par = np.zeros(n_comb)
    
        
    time1 = time()
    for i, comb in enumerate(combinations(range(max_comb),2)):
        out[i] = par_fun(comb)
    time2 = time()
#    print(out)
    print('Processed in:', time2 - time1)
    
    with Pool(8) as p:
        time2 = time()
        for i, val in enumerate(p.imap(par_fun2,combinations(range(max_comb),2), chunksize=100)):
            out_par[i] = val
    time3 = time()
#    print(out_par)
    print('Processed with p.imap in:', time3 - time2)
    print('\nSleep time:', real_sleep_time / n_comb)
        
if __name__=='__main__':
    main()