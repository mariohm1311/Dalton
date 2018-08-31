# -*- coding: utf-8 -*-

from libdalton import fileio
from libdalton import simulate

if __name__=='__main__':
    input_file_name = fileio.validate_input(__file__)
    
    simulation = simulate.MolecularDynamics(input_file_name)
    
    simulation.run()