"""Functions for computing molecular mechanics energy gradients.
"""

import itertools
import numpy as np

from libdalton import constants as const
from libdalton import geometry

def get_g_mag_bond(r_ij, r_eq, k_b):
    """get_g_mag_bond calculates the magnitude of the bond energy gradient as per 
    the harmonic oscillator model from the distances and bonding parameters passed 
    in the arguments.
    
    Args: 
        r_ij (float): distance between a pair of atoms [Angstrom]. 
        r_eq (float): equilibrium bond length for a pair of atoms [Angstrom].
        k_b (float): bond spring constant for a pair of atoms [kcal/(mol*A^2)].
    
    Returns:
        g_bond (float): signed magnitude of the energy gradient of the bond [kcal/(mol*A)].
    """
    return 2 * k_b * (r_ij - r_eq)
    
def get_g_mag_angle(a_ijk, a_eq, k_a):
    """get_g_mag_angle calculates the magnitude of the angle energy gradient as 
    per the harmonic oscillator model from the angles and bonding parameters passed 
    in the arguments.
    
    Args: 
        a_ijk (float): angle between atoms 1, 2 and 3 [degrees]. 
        a_eq (float): equilibrium bond angle for atoms 1, 2 and 3 [degrees].
        k_a (float): angle spring constant for atoms 1, 2 and 3 [kcal/(mol*rad^2)].
    
    Returns:
        g_angle (float): bonding energy of the angle [kcal/(mol*rad)].
    """
    return 2 * k_a * const.DEG2RAD * (a_ijk - a_eq)

def get_g_mag_torsion(t_ijkl, v_n, gamma, n_fold, paths):
    return -v_n * n_fold * np.sin(const.DEG2RAD * (n_fold * t_ijkl - gamma)) / paths

def get_g_mag_outofplane(o_ijkl, v_n):
    return -2.0 * v_n * np.sin(const.DEG2RAD * (2.0 * o_ijkl - 180.0))

def get_g_mag_vdw(r_ij, eps, ro, cutoff=0.0):
    if cutoff == 0.0:
        rrel_ij = ro / r_ij
        return -12.0 * (eps / ro) * (rrel_ij**13 - rrel_ij**7)
    else:
        cutoff_rad = cutoff * ro
        if r_ij >= cutoff_rad:
            return 0.0
        else:
            dvdr_r = get_g_mag_vdw(r_ij, eps, ro, cutoff=0.0)
            dvdr_shift = get_g_mag_vdw(cutoff_rad, eps, ro, cutoff=0.0)
            return dvdr_r - dvdr_shift

def get_g_mag_elst(r_ij, q_i, q_j, dielectric, cutoff=0.0):
    if q_i == 0.0 or q_j == 0.0:
        return 0.0
    
    if cutoff == 0.0:
        return -const.CEU2KCAL * q_i * q_j / (dielectric * r_ij**2)
    else:
        if r_ij >= cutoff:
            return 0.0
        else:
            dvdr_r = get_g_mag_elst(r_ij, q_i, q_j, dielectric, cutoff=0.0)
            dvdr_shift = get_g_mag_elst(cutoff, q_i, q_j, dielectric, cutoff=0.0)
            return dvdr_r - dvdr_shift

#def get_g_mag_elst(r_ij, q_i, q_j, dielectric):
#    return -const.CEU2KCAL * q_i * q_j / (dielectric * r_ij**2)

def get_g_dir_bond(coords_1, coords_2, r_12=None):
    gdir_1 = geometry.get_u_ij(coords_2, coords_1, r_12)
    gdir_2 = -1.0 * gdir_1
    return gdir_1, gdir_2

def get_g_dir_angle(coords_1, coords_2, coords_3, r_21=None, r_23=None):
    if r_21 is None:
        r_21 = geometry.get_r_ij(coords_2, coords_1)
    if r_23 is None:
        r_23 = geometry.get_r_ij(coords_2, coords_3)
    
    u_21 = geometry.get_u_ij(coords_2, coords_1, r_21)
    u_23 = geometry.get_u_ij(coords_2, coords_3, r_23)
    ucp_2123 = np.cross(u_21, u_23)
    ucp_2123 /= np.linalg.norm(ucp_2123)
    
    gdir_1 = np.cross(u_21, ucp_2123)
    gdir_1 /= np.linalg.norm(gdir_1) * r_21
    gdir_3 = np.cross(ucp_2123, u_23)
    gdir_3 /= np.linalg.norm(gdir_3) * r_23
    gdir_2 = -1.0 * (gdir_1 + gdir_3)
    
    return gdir_1, gdir_2, gdir_3

def get_g_dir_torsion(coords_1, coords_2, coords_3, coords_4, r_12=None,
                      r_23=None, r_34=None):
    if r_12 is None:
        r_12 = geometry.get_r_ij(coords_1, coords_2)
    if r_23 is None:
        r_23 = geometry.get_r_ij(coords_2, coords_3)
    if r_34 is None:
        r_34 = geometry.get_r_ij(coords_3, coords_4)
    
    u_21 = geometry.get_u_ij(coords_2, coords_1, r_12)
    u_34 = geometry.get_u_ij(coords_3, coords_4, r_34)
    u_23 = geometry.get_u_ij(coords_2, coords_3, r_23)
    u_32 = -1.0 * u_23
    
    a_123 = geometry.get_a_ijk(coords_1, coords_2, coords_3, r_12, r_23)
    a_432 = geometry.get_a_ijk(coords_4, coords_3, coords_2, r_34, r_23)
    
    s_123 = np.sin(const.DEG2RAD * a_123)
    s_432 = np.sin(const.DEG2RAD * a_432)
    c_123 = np.cos(const.DEG2RAD * a_123)
    c_432 = np.cos(const.DEG2RAD * a_432)
    
    gdir_1 = np.cross(u_21, u_23)
    gdir_1 /= np.linalg.norm(gdir_1) * r_12 * s_123
    gdir_4 = np.cross(u_34, u_32)
    gdir_4 /= np.linalg.norm(gdir_4) * r_34 * s_432
    gdir_2 = (r_12 * c_123 / r_23 - 1.0) * gdir_1 - (r_34 * c_432 / r_23) * gdir_4
    gdir_3 = (r_34 * c_432 / r_23 - 1.0) * gdir_4 - (r_12 * c_123 / r_23) * gdir_1

    return gdir_1, gdir_2, gdir_3, gdir_4

def get_g_dir_outofplane(coords_1, coords_2, coords_3, coords_4, o_ijkl,
                         r_31=None, r_32=None, r_34=None):
    if r_31 is None:
        r_31 = geometry.get_r_ij(coords_3, coords_1)
    if r_32 is None:
        r_32 = geometry.get_r_ij(coords_3, coords_2)
    if r_34 is None:
        r_34 = geometry.get_r_ij(coords_3, coords_4)

    u_31 = geometry.get_u_ij(coords_3, coords_1, r_31)
    u_32 = geometry.get_u_ij(coords_3, coords_2, r_32)
    u_34 = geometry.get_u_ij(coords_3, coords_4, r_34)
    
    cp_3234 = np.cross(u_32, u_34)
    cp_3431 = np.cross(u_34, u_31)
    cp_3132 = np.cross(u_31, u_32)
    
    a_132 = geometry.get_a_ijk(coords_1, coords_3, coords_2, r_31, r_34)
    s_132 = np.sin(const.DEG2RAD * a_132)
    c_132 = np.cos(const.DEG2RAD * a_132)
    c_o_ijkl = np.cos(const.DEG2RAD * o_ijkl)
    t_o_ijkl = np.tan(const.DEG2RAD * o_ijkl)
    
    gdir_1 = ((1.0 / r_31) * (cp_3234 / (c_o_ijkl * s_132) 
        - (t_o_ijkl / s_132**2) * (u_31 - c_132 * u_32)))
    gdir_2 = ((1.0 / r_32) * (cp_3431 / (c_o_ijkl * s_132) 
        - (t_o_ijkl / s_132**2) * (u_32 - c_132 * u_31)))
    gdir_4 = ((1.0 / r_34) * (cp_3132 / (c_o_ijkl * s_132) - (t_o_ijkl * u_34)))
    gdir_3 = -1.0 * (gdir_1 + gdir_2 + gdir_4)
    
    return gdir_1, gdir_2, gdir_3, gdir_4

def get_g_boundary(coords, origin, k_bound, boundary, boundary_type, boundary_2=None):
    g_boundary = np.zeros(const.NUMDIM)
    
    if boundary_type == 'cube':
        for dim in range(const.NUMDIM):
            r_io = coords[dim] - origin[dim]
            
            if abs(r_io) > boundary:
                sign = 1.0 if r_io <= 0.0 else -1.0
                g_boundary[dim] = -2.0 * sign * k_bound * (abs(coords[dim]) - boundary)
    elif boundary_type == 'sphere':
        if not boundary_2:
            boundary_2 = boundary**2
            
        r2_io = geometry.get_r2_ij(origin, coords)
        
        if r2_io > boundary_2:
            r_io = np.sqrt(r2_io)
            u_io = geometry.get_u_ij(origin, coords, r_io)
            g_boundary = 2.0 * k_bound * (r_io - boundary) * u_io
    
    return g_boundary

def get_all_g_bond(g_bonds, bonds, atoms):
    g_bonds.fill(0.0)
    
    for bond in bonds:
        bond.get_gradient_mag()
        
        coords_1 = atoms[bond.at_1].coords
        coords_2 = atoms[bond.at_2].coords
        gdir_1, gdir_2 = get_g_dir_bond(coords_1, coords_2, bond.r_ij)
        
        g_bonds[bond.at_1] += bond.grad_mag * gdir_1
        g_bonds[bond.at_2] += bond.grad_mag * gdir_2

def get_all_g_angle(g_angles, angles, atoms, bond_graph):
    g_angles.fill(0.0)
    
    for angle in angles:
        angle.get_gradient_mag()
        
        coords_1 = atoms[angle.at_1].coords
        coords_2 = atoms[angle.at_2].coords
        coords_3 = atoms[angle.at_3].coords
        r_21 = bond_graph[angle.at_2][angle.at_1]
        r_23 = bond_graph[angle.at_2][angle.at_3]
        gdir_1, gdir_2, gdir_3 = get_g_dir_angle(coords_1, coords_2, coords_3,
                                                 r_21, r_23)
        
        g_angles[angle.at_1] += angle.grad_mag * gdir_1
        g_angles[angle.at_2] += angle.grad_mag * gdir_2
        g_angles[angle.at_3] += angle.grad_mag * gdir_3
        
def get_all_g_torsions(g_torsions, torsions, atoms, bond_graph):
    g_torsions.fill(0.0)
    
    for torsion in torsions:
        torsion.get_gradient_mag()
        
        coords_1 = atoms[torsion.at_1].coords
        coords_2 = atoms[torsion.at_2].coords
        coords_3 = atoms[torsion.at_3].coords
        coords_4 = atoms[torsion.at_4].coords
        r_12 = bond_graph[torsion.at_1][torsion.at_2]
        r_23 = bond_graph[torsion.at_2][torsion.at_3]
        r_34 = bond_graph[torsion.at_3][torsion.at_4]
        gdir_1, gdir_2, gdir_3, gdir_4 = get_g_dir_torsion(coords_1, coords_2, 
                                                           coords_3, coords_4, 
                                                           r_12, r_23, r_34)
        
        g_torsions[torsion.at_1] += torsion.grad_mag * gdir_1
        g_torsions[torsion.at_2] += torsion.grad_mag * gdir_2
        g_torsions[torsion.at_3] += torsion.grad_mag * gdir_3
        g_torsions[torsion.at_4] += torsion.grad_mag * gdir_4
        
def get_all_g_outofplanes(g_outofplanes, outofplanes, atoms, bond_graph):
    g_outofplanes.fill(0.0)
    
    for outofplane in outofplanes:
        outofplane.get_gradient_mag()
        
        coords_1 = atoms[outofplane.at_1].coords
        coords_2 = atoms[outofplane.at_2].coords
        coords_3 = atoms[outofplane.at_3].coords
        coords_4 = atoms[outofplane.at_4].coords
        r_31 = bond_graph[outofplane.at_3][outofplane.at_1]
        r_32 = bond_graph[outofplane.at_3][outofplane.at_2]
        r_34 = bond_graph[outofplane.at_3][outofplane.at_4]
        gdir_1, gdir_2, gdir_3, gdir_4 = get_g_dir_outofplane(coords_1, coords_2, 
                                                              coords_3, coords_4,
                                                              outofplane.o_ijkl,
                                                              r_31, r_32, r_34)
        
        g_outofplanes[outofplane.at_1] += outofplane.grad_mag * gdir_1
        g_outofplanes[outofplane.at_2] += outofplane.grad_mag * gdir_2
        g_outofplanes[outofplane.at_3] += outofplane.grad_mag * gdir_3
        g_outofplanes[outofplane.at_4] += outofplane.grad_mag * gdir_4

def get_all_g_nonbonded(g_vdw, g_elst, atoms, nonints, dielectric, atom_tree, cutoff):
    g_vdw.fill(0.0)
    g_elst.fill(0.0)
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
            gdir_1, gdir_2 = get_g_dir_bond(at_1.coords, at_2.coords, r_ij)        
            grad_elst_mag = get_g_mag_elst(r_ij, at_1.at_charge, at_2.at_charge, dielectric)
            g_elst[i] += grad_elst_mag * gdir_1
            g_elst[j] += grad_elst_mag * gdir_2
            grad_vdw_mag = get_g_mag_vdw(r_ij, eps, ro)
            g_vdw[i] += grad_vdw_mag * gdir_1
            g_vdw[j] += grad_vdw_mag * gdir_2
        else:
            if not i == pivot:
                pivot = i
                nn_pivot = atom_tree.query_radius(at_1.coords.reshape(1,-1), r=search_rad,
                                                  return_distance=True)
                nn_idx = nn_pivot[0][0].tolist()
                nn_dist = nn_pivot[1][0]
            
            if j in nn_idx:
                r_ij = nn_dist[nn_idx.index(j)]
                gdir_1, gdir_2 = get_g_dir_bond(at_1.coords, at_2.coords, r_ij)  
                
                if r_ij <= elst_cutoff:
                    grad_elst_mag = get_g_mag_elst(r_ij, at_1.at_charge, at_2.at_charge, dielectric,
                                                   cutoff=elst_cutoff)
                    g_elst[i] += grad_elst_mag * gdir_1
                    g_elst[j] += grad_elst_mag * gdir_2
                
                if r_ij <= vdw_cutoff * ro:
                    grad_vdw_mag = get_g_mag_vdw(r_ij, eps, ro, cutoff=vdw_cutoff)
                    g_vdw[i] += grad_vdw_mag * gdir_1
                    g_vdw[j] += grad_vdw_mag * gdir_2
        
#def get_all_g_nonbonded(g_vdw, g_elst, atoms, nonints, dielectric):
#    g_vdw.fill(0.0)
#    g_elst.fill(0.0)
#    
#    for i, j in itertools.combinations(range(len(atoms)), 2):
#        if (i, j) in nonints:
#            continue
#        
#        at_1, at_2 = atoms[i], atoms[j]
#        r_ij = geometry.get_r_ij(at_1.coords, at_2.coords)
#        gdir_1, gdir_2 = get_g_dir_bond(at_1.coords, at_2.coords, r_ij)
#        
#        eps = at_1.at_sreps * at_2.at_sreps
#        ro = at_1.at_ro + at_2.at_ro
#        grad_elst_mag = get_g_mag_elst(r_ij, at_1.at_charge, at_2.at_charge, dielectric)
#        grad_vdw_mag = get_g_mag_vdw(r_ij, eps, ro)
#        g_vdw[i] += grad_vdw_mag * gdir_1
#        g_vdw[j] += grad_vdw_mag * gdir_2
#        g_elst[i] += grad_elst_mag * gdir_1
#        g_elst[j] += grad_elst_mag * gdir_2

def get_all_g_boundary(g_boundary, atoms, origin, k_bound, boundary, boundary_type):
    g_boundary.fill(0.0)
    boundary_2 = boundary**2
    
    for i, atom in enumerate(atoms):
        g_boundary[i] += get_g_boundary(atom.coords, origin, k_bound, boundary,
                                        boundary_type, boundary_2=boundary_2)

def get_virial(g_total, atoms):
    virial = 0.0
    
    for i, atom in enumerate(atoms):
        for dim in range(const.NUMDIM):
            virial += atom.coords[dim] * g_total[i][dim]
    
    return virial

def get_pressure(n_atoms, temperature, virial, volume):
    return const.KCALAMOL2PA * (
            n_atoms * const.KB * temperature + virial / const.NUMDIM) / volume

def get_all_g_numerical(mol):
    mol.g_bonds.fill(0.0)
    mol.g_angles.fill(0.0)
    mol.g_torsions.fill(0.0)
    mol.g_outofplanes.fill(0.0)
    mol.g_vdw.fill(0.0)
    mol.g_elst.fill(0.0)
    mol.g_boundary.fill(0.0)
    
    for i, atom in enumerate(mol.atoms):
        for dim in range(const.NUMDIM):
            c_dim = atom.coords[dim]
            
            cp_dim = c_dim + 0.5 * const.NUMDISP
            atom.set_coord(dim, cp_dim)
            mol.update_internals()
            mol.get_energy()
            ep_bond, ep_ang, ep_tor, ep_oop, ep_vdw, ep_elst, ep_bound = (
                    mol.e_bonds, mol.e_angles, mol.e_torsions, mol.e_outofplanes,
                    mol.e_vdw, mol.e_elst, mol.e_boundary)
            
            cm_dim = c_dim - 0.5 * const.NUMDISP
            atom.set_coord(dim, cm_dim)
            mol.update_internals()
            mol.get_energy()
            em_bond, em_ang, em_tor, em_oop, em_vdw, em_elst, em_bound = (
                    mol.e_bonds, mol.e_angles, mol.e_torsions, mol.e_outofplanes,
                    mol.e_vdw, mol.e_elst, mol.e_boundary)
            
            atom.set_coord(dim, c_dim)
            mol.g_bonds[i][dim] = (ep_bond - em_bond) / const.NUMDISP
            mol.g_angles[i][dim] = (ep_ang - em_ang) / const.NUMDISP
            mol.g_torsions[i][dim] = (ep_tor - em_tor) / const.NUMDISP
            mol.g_outofplanes[i][dim] = (ep_oop - em_oop) / const.NUMDISP
            mol.g_vdw[i][dim] = (ep_vdw - em_vdw) / const.NUMDISP
            mol.g_elst[i][dim] = (ep_elst - em_elst) / const.NUMDISP
            mol.g_boundary[i][dim] = (ep_bound - em_bound) / const.NUMDISP
    
    mol.update_internals()
    mol.get_energy()