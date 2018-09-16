# -*- coding: utf-8 -*-
"""
Created on Sun May 20 14:33:15 2018

@author: Mario
"""

import numpy as np
import sys
from time import sleep
import gc


"""
res=[]
i=0
while True:
    t=np.zeros((10^5,1),dtype=int)
    u=t[:1]
#    res.append(t)
    i=i+1
    print(i)


first_arr = np.ones(10**8,dtype=float)
print(first_arr.nbytes//1000000)

sleep(5)


first_arr = first_arr[5*10**7:10**8].copy()
print(first_arr.nbytes//1000000)

sleep(5)

first_arr = np.ones(10**8,dtype=float)
first_arr = np.delete(first_arr,range(0,5*10**7),0)
print(first_arr.nbytes//1000000)

sleep(5)

gc.collect()
sleep(5)
"""

"""
from sklearn.neighbors import KDTree
np.random.seed(1)
data = np.random.normal(0,1, (8,2))


data_tree = KDTree(data, leaf_size=2)


print(data)
print(data_tree.query(np.array([1,1]).reshape(1,-1),k=2))
print(data_tree.query_radius(np.array([1,1]).reshape(1,-1), r=1.8))

data[4,0] = 0.25
print(data)
print(data_tree.query(np.array([1,1]).reshape(1,-1),k=2))
print(data_tree.query_radius(np.array([1,1]).reshape(1,-1), r=1.8, return_distance=True))

data = np.append(data, np.array([[1,1]]), axis=0)
data_tree = KDTree(data, leaf_size=2)
print(data)
print(data_tree.query(np.array([1,1]).reshape(1,-1),k=2))
print(data_tree.query_radius(np.array([1,1]).reshape(1,-1), r=1.8, return_distance=True))
print(data_tree.query_radius(np.array([1,1]).reshape(1,-1), r=1.8)[0].tolist())
print(data_tree.query_radius(np.array([1,1]).reshape(1,-1), r=1.8, return_distance=True)[1][0].tolist())
"""

import itertools
for i, j in itertools.combinations(range(4), 2):
    print(i,j)
    
for i in range(4):
    print(i)
"""

""" 
from multiprocessing import Pool
from time import time

np.random.seed(1)

rows = 10
columns = 3
vals = np.random.normal(0,1,(rows,columns))

def fun(u,v,w):
    sleep(1)
    return 2*u + 3*v + 5*w

def run(values):
    out = []
    for row in range(rows):
        out.append(fun(*vals[row]))
        
    return out

def run_parallel(values):
    with Pool(4) as p:
        out = p.starmap(fun, vals)
    
    return out

if __name__ == '__main__':
    time1 = time()
    run(vals)
    time2 = time()
    print('done in', time2-time1)
    print(vals)
    run_parallel(vals)
    time3 = time()
    print('done in', time3-time2)
    print(vals)

#import itertools
#
#comb = np.array(list(itertools.combinations(range(10000), 2)))
#print(comb.nbytes / 1000000)