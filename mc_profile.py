# -*- coding: utf-8 -*-

from libdalton import fileio
from libdalton import simulate

def main():
    simulation = simulate.MonteCarlo('input/mc/ethane.mc')
    
    simulation.run()

if __name__=='__main__':
    main()