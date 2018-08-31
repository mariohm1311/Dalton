# -*- coding: utf-8 -*-

from libdalton import fileio
from libdalton import optimize

if __name__=='__main__':
    input_file_name = fileio.validate_input(__file__)
    
    optimization = optimize.Optimization(input_file_name)
    
    optimization.optimize()