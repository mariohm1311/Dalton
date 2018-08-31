# -*- coding: utf-8 -*-

from libdalton import fileio
from libdalton import simulate

def main():
    simulation = simulate.MolecularDynamics('input/md/h2o_5.md')
    
    simulation.run()

if __name__=='__main__':
    main()