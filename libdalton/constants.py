
"""Mathematical and physical constants used in the library."""

import numpy as np

NUMDIM = int(3)

DEG2RAD = np.pi / 180

RAD2DEG = 180 / np.pi

BONDTHRESHOLD = 1.2

CEU2KCAL = 332.06375

KIN2KCAL = 0.00239005736

ACCCONV = 418.400000

KB = 0.001987204

KCALAMOL2PA = 69476.95

RGAS = 0.83144598

NUMDISP = 1.0E-6

LONGRANGECUTOFF = (1.0, 8.0)
#LONGRANGECUTOFF = (0.0, 0.0)

KINETICCALCMETHODS = ['nokinetic', 'direct', 'leapfrog']

GRADIENTCALCMETHODS = ['analytic', 'numerical']

PROPERTYDICTIONARY = {
    'e_total':    ['Total',      12, '#000000', 1],
    'e_kin':      ['Kinetic',    11, '#007D34', 2],
    'e_pot':      ['Potential',   1, '#C10020', 3],
    'e_nonbond':  ['Non-bonded',  7, '#0000FF', 4],
    'e_bonded':   ['Bonded',      2, '#FF6800', 5],
    'e_boundary': ['Boundary',   10, '#551A8B', 6],
    'e_vdw':      ['Vdw',         9, '#00BFFF', 7],
    'e_elst':     ['Elst',        8, '#EEC900', 8],
    'e_bond':     ['Bonds',       3, '#F08080', 9],
    'e_angle':    ['Angles',      4, '#90EE90', 10],
    'e_tors':     ['Torsions',    6, '#FF83FA', 11],
    'e_oop':      ['Outofplanes', 5, '#A9A9A9', 12],
    'temp':       ['Kinetic Temperature (K)',13, '#C10020', 13],
    'press':      ['Kinetic Pressure (Pa)',   14, '#0000FF', 14]}

PROPERTYKEYS = [
    'e_total', 'e_kin', 'e_pot', 'e_nonbond', 'e_bonded', 'e_boundary',
    'e_vdw', 'e_elst', 'e_bond', 'e_angle', 'e_tors', 'e_oop', 'temp', 'press']

OPTCRITERIAREFS = {
    'loose':     [1.0E-4,  1.0E-3, 2.0E-3, 1.0E-2, 2.0E-2],
    'default':   [1.0E-6,  1.0E-4, 2.0E-4, 1.0E-3, 2.0E-3],
    'tight':     [1.0E-8,  1.0E-5, 2.0E-5, 1.0E-4, 2.0E-4],
    'verytight': [1.0E-10, 1.0E-6, 2.0E-6, 1.0E-5, 2.0E-5]}

OPTSTEPADJUSTOR = np.sqrt(2)

NUMLINESEARCHSTEPS = 7

PERCENTIMAGEPLOT = 0.75

POINTSPERINCH = 72

TICCHARS = {0: '', 1: 'k', 2: 'M', 3: 'B', 4: 'T', 5: 'P'}
