"""Functions for computing molecular mechanics energy gradients.
"""

import itertools
import numpy as np

from libdalton import constants as const
from libdalton import geometry
from libdalton import gradient


def get_e_bond(r_ij, r_eq, k_b):
    """get_e_bond calculates the bonding energy as per the harmonic oscillator 
    model from the distances and bonding parameters passed in the arguments.
    
    Args: 
        r_ij (float): distance between a pair of atoms [Angstrom]. 
        r_eq (float): equilibrium bond length for a pair of atoms [Angstrom].
        k_b (float): bond spring constant for a pair of atoms [kcal/(mol*A^2)].
    
    Returns:
        e_bond (float): bonding energy of the bond [kcal/mol].
    """
    return k_b * (r_ij - r_eq)**2

def get_e_angle(a_ijk, a_eq, k_a):
    """get_e_angle calculates the bonding angle energy as per the harmonic oscillator 
    model from the angles and bonding parameters passed in the arguments.
    
    Args: 
        a_ijk (float): angle between atoms 1, 2 and 3 [degrees]. 
        a_eq (float): equilibrium bond angle for atoms 1, 2 and 3 [degrees].
        k_a (float): angle spring constant for atoms 1, 2 and 3 [kcal/(mol*rad^2)].
    
    Returns:
        e_angle (float): bonding energy of the angle [kcal/mol].
    """
    return k_a * (const.DEG2RAD * (a_ijk - a_eq))**2

def get_e_torsion(t_ijkl, v_n, gamma, n_fold, paths):
    return v_n * (1.0 + np.cos(const.DEG2RAD * (n_fold * t_ijkl - gamma))) / paths

def get_e_outofplane(o_ijkl, v_n):
    return v_n * (1.0 + np.cos(const.DEG2RAD * (2.0 * o_ijkl - 180.0)))

def get_e_vdw(r_ij, eps, ro, cutoff=0.0):
    if cutoff == 0.0:
        r6_ij = (ro / r_ij)**6
        return eps * (r6_ij**2 - 2.0 * r6_ij)
    else:
        cutoff_rad = cutoff * ro
        if r_ij >= cutoff_rad:
            return 0.0
        else:
            v_r = get_e_vdw(r_ij, eps, ro, cutoff=0.0)
            v_shift = get_e_vdw(cutoff_rad, eps, ro, cutoff=0.0)
            dvdr_shift = gradient.get_g_mag_vdw(cutoff_rad, eps, ro, cutoff=0.0)
            return v_r - v_shift - dvdr_shift * (r_ij - cutoff_rad)

def get_e_elst(r_ij, q_i, q_j, dielectric, cutoff=0.0):
    if q_i == 0.0 or q_j == 0.0:
        return 0.0
    
    if cutoff == 0.0:
        return const.CEU2KCAL * q_i * q_j / (dielectric * r_ij)
    else:
        if r_ij >= cutoff:
            return 0.0
        else:
            v_r = get_e_elst(r_ij, q_i, q_j, dielectric, cutoff=0.0)
            v_shift = get_e_elst(cutoff, q_i, q_j, dielectric, cutoff=0.0)
            dvdr_shift = gradient.get_g_mag_elst(cutoff, q_i, q_j, dielectric, cutoff=0.0)
            return v_r - v_shift - dvdr_shift * (r_ij - cutoff)

def get_e_boundary(coords, origin, k_bound, boundary, boundary_type, boundary_2=None):
    e_boundary = 0.0
      
    if boundary_type == 'cube':
        for dim in range(const.NUMDIM):
            r_io = abs(coords[dim] - origin[dim])
            
            if r_io > boundary:
                e_boundary += k_bound * (r_io - boundary)**2
    elif boundary_type == 'sphere':
        if not boundary_2:
            boundary_2 = boundary**2
        r2_io = geometry.get_r2_ij(origin, coords)
        
        if r2_io > boundary_2:
            r_io = np.sqrt(r2_io)
            e_boundary += k_bound * (r_io - boundary)**2
    
    return e_boundary

def get_e_kinetic(mass, vels):
    e_kinetic = 0.0
    
    for dim in range(const.NUMDIM):
        e_kinetic += mass * vels[dim]**2
    
    return 0.5 * const.KIN2KCAL * e_kinetic

def get_total_e_bonds(bonds):
    e_bonds = 0.0
    
    for bond in bonds:
        bond.get_energy()
        e_bonds += bond.energy
    
    return e_bonds

def get_total_e_angles(angles):
    e_angles = 0.0
    
    for angle in angles:
        angle.get_energy()
        e_angles += angle.energy
    
    return e_angles

def get_total_e_torsions(torsions):
    e_torsions = 0.0
    
    for torsion in torsions:
        torsion.get_energy()
        e_torsions += torsion.energy
    
    return e_torsions

def get_total_e_outofplanes(outofplanes):
    e_outofplanes = 0.0
    
    for outofplane in outofplanes:
        outofplane.get_energy()
        e_outofplanes += outofplane.energy
    
    return e_outofplanes

def get_total_e_nonbonded(atoms, nonints, dielectric, atom_tree, cutoff):
    e_vdw, e_elst = 0.0, 0.0
    vdw_cutoff = cutoff[0]
    elst_cutoff = cutoff[1]
    
    search_rad = max(6.8*vdw_cutoff, elst_cutoff)
    dont_truncate = cutoff == (0.0, 0.0)
    pivot = -1
    
    for i, j in itertools.combinations(range(len(atoms)), 2):
        if (i, j) in nonints:
            continue
        
        at_1, at_2 = atoms[i], atoms[j]
        eps = at_1.at_sreps * at_2.at_sreps
        ro = at_1.at_ro + at_2.at_ro
        
        if dont_truncate:
            r_ij = geometry.get_r_ij(at_1.coords, at_2.coords)
            e_elst += get_e_elst(r_ij, at_1.at_charge, at_2.at_charge, dielectric)
            e_vdw += get_e_vdw(r_ij, eps, ro)
        else:
            if not i == pivot:
                pivot = i
                nn_pivot = atom_tree.query_radius(at_1.coords.reshape(1,-1), r=search_rad,
                                                  return_distance=True)
                nn_idx = nn_pivot[0][0].tolist()
                nn_dist = nn_pivot[1][0]
            
            if j in nn_idx:
                r_ij = nn_dist[nn_idx.index(j)]
                if r_ij <= vdw_cutoff * ro:
                    e_elst += get_e_elst(r_ij, at_1.at_charge, at_2.at_charge, dielectric,
                                         cutoff=elst_cutoff)
                
                if r_ij <= elst_cutoff:
                    e_vdw += get_e_vdw(r_ij, eps, ro, cutoff=vdw_cutoff)
        
    return e_vdw, e_elst

#def get_total_e_nonbonded(atoms, nonints, dielectric):
#    e_vdw, e_elst = 0.0, 0.0
#    
#    for i, j in itertools.combinations(range(len(atoms)), 2):
#        if (i, j) in nonints:
#            continue
#        
#        at_1, at_2 = atoms[i], atoms[j]
#        r_ij = geometry.get_r_ij(at_1.coords, at_2.coords)
#        eps = at_1.at_sreps * at_2.at_sreps
#        ro = at_1.at_ro + at_2.at_ro
#        
#        e_elst += get_e_elst(r_ij, at_1.at_charge, at_2.at_charge, dielectric)
#        e_vdw += get_e_vdw(r_ij, eps, ro)
#    
#    return e_vdw, e_elst

def get_total_e_boundary(atoms, origin, k_bound, boundary, boundary_type):
    e_boundary = 0.0
    boundary_2 = boundary**2
    
    for atom in atoms:
        e_boundary += get_e_boundary(atom.coords, origin, k_bound, boundary,
                                     boundary_type, boundary_2=boundary_2)
    
    return e_boundary

def get_total_e_kinetic(atoms, calc_method='leapfrog'):
    e_kinetic = 0.0
    
    if calc_method == 'nokinetic':
        return 0.0
    
    if calc_method == 'direct':
        for atom in atoms:
            e_kinetic += get_e_kinetic(atom.at_mass, atom.vels)
    
    if calc_method == 'leapfrog':
        for atom in atoms:
            vels = 0.5 * (atom.vels + atom.pvels)
            e_kinetic += get_e_kinetic(atom.at_mass, vels)
    
    return e_kinetic

def get_temperature(e_kinetic, n_atoms):
    return (2.0 / const.NUMDIM) * e_kinetic / (const.KB * n_atoms)        