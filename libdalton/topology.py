""" TODO Doc

"""

import itertools
import numpy as np

from libdalton import constants as const
from libdalton import geometry
from libdalton import molecule
from libdalton import param

def get_bond_graph(atoms):
    bond_graph = {i:{} for i in range(len(atoms))}
    
    for i, j in itertools.combinations(range(len(atoms)), 2):
        at_1, at_2 = atoms[i], atoms[j]
        threshold = const.BONDTHRESHOLD * (at_1.at_rad + at_2.at_rad)
        r2_12 = geometry.get_r2_ij(at_1.coords, at_2.coords)
        
        if r2_12 < threshold**2:
            r_12 = np.sqrt(r2_12)
            bond_graph[i][j] = bond_graph[j][i] = r_12
    
    return bond_graph
    
def get_bond_graph_from_bonds(bonds, n_atoms):
    bond_graph = {i:{} for i in range(n_atoms)}
    
    for bond in bonds:
        bond_graph[bond.at_1][bond.at_2] = bond.r_ij
        bond_graph[bond.at_2][bond.at_1] = bond.r_ij
        
    return bond_graph

def get_bonds(atoms, bond_graph):
    bonds = []
    
    for i, at_1 in enumerate(atoms):
        for j in bond_graph[i]:
            if i > j:
                continue
            
            at_2 = atoms[j]
            r_ij = bond_graph[i][j]
            k_b, r_eq = param.get_bond_param(at_1.at_type, at_2.at_type)
            
            if k_b > 0.0:
                bonds.append(molecule.Bond(i, j, r_ij, r_eq, k_b))
    
    bonds.sort(key = lambda b: (b.at_1, b.at_2))
#    Equivalent
#    bonds = sorted(bonds, key = lambda b: b.at_2)
#    bonds = sorted(bonds, key = lambda b: b.at_1)
    return bonds
    
def get_angles(atoms, bond_graph):
    angles = []
    
    for j, at_2 in enumerate(atoms):
        for i, k in itertools.combinations(bond_graph[j], 2):
            if i > k:
                continue
            
            at_1, at_3 = atoms[i], atoms[k]
            a_ijk = geometry.get_a_ijk(at_1.coords, at_2.coords, at_3.coords,
                                       r_ij=bond_graph[i][j], r_jk=bond_graph[j][k])
            k_a, a_eq = param.get_angle_param(at_1.at_type, at_2.at_type, at_3.at_type)
            
            if k_a > 0.0:
                angles.append(molecule.Angle(i, j, k, a_ijk, a_eq, k_a))
    
    angles.sort(key = lambda a: (a.at_1, a.at_2, a.at_3))
    return angles

def get_torsions(atoms, bond_graph):
    torsions = []
    
    for j, at_2 in enumerate(atoms):
        for i, k in itertools.permutations(bond_graph[j], 2):
            if j > k:
                continue
            
            at_1, at_3 = atoms[i], atoms[k]
            
            for l in bond_graph[k]:
                if l == i or l == j:
                    continue

                at_4 = atoms[l]
                t_ijkl = geometry.get_t_ijkl(at_1.coords, at_2.coords, at_3.coords,
                                             at_4.coords, r_ij=bond_graph[i][j],
                                             r_jk=bond_graph[j][k], r_kl=bond_graph[k][l])
                params = param.get_torsion_param(at_1.at_type, at_2.at_type,
                                                 at_3.at_type, at_4.at_type)
                
                for v_n, gamma, n_fold, paths in params:
                    if v_n:
                        torsions.append(molecule.Torsion(i, j, k, l, t_ijkl,
                                                         v_n, gamma, n_fold, paths))
                    
    torsions.sort(key = lambda t: (t.at_1, t.at_2, t.at_3, t.at_4))
    return torsions

def get_outofplanes(atoms, bond_graph):
    outofplanes = []
    
    for k in range(len(atoms)):
        for i, j, l in itertools.combinations(bond_graph[k], 3):            
            arrangements = [(min(i, j), max(i, j), k, l),
                            (min(j, l), max(j, l), k, i),
                            (min(i, l), max(i, l), k, j)]
            
            for arrangement in arrangements:
                o_ijkl = geometry.get_o_ijkl(*[atoms[idx].coords for idx in arrangement])
                v_n = param.get_outofplane_param(*[atoms[idx].at_type for idx in arrangement])
                
                if v_n:
                    outofplanes.append(molecule.OutOfPlane(*arrangement, o_ijkl, v_n))
    
    outofplanes.sort(key = lambda o: (o.at_1, o.at_2, o.at_3, o.at_4))
    return outofplanes

def get_noninteracting(bonds, angles, torsions):
    nonints = set()
    
    for bond in bonds:
        nonints.add((bond.at_1, bond.at_2))
        nonints.add((bond.at_2, bond.at_1))
    for angle in angles:
        nonints.add((angle.at_1, angle.at_3))
        nonints.add((angle.at_3, angle.at_1))
    for torsion in torsions:
        nonints.add((torsion.at_1, torsion.at_4))
        nonints.add((torsion.at_4, torsion.at_1))
    
    return nonints

def set_bonds(bonds, atoms, bond_graph):
    for bond in bonds:
        coords_1 = atoms[bond.at_1].coords
        coords_2 = atoms[bond.at_2].coords
        bond.set_rij(geometry.get_r_ij(coords_1, coords_2))
        bond_graph[bond.at_1][bond.at_2] = bond_graph[bond.at_2][bond.at_1] = bond.r_ij
    
def set_angles(angles, atoms, bond_graph):
    for angle in angles:
        r_12 = bond_graph[angle.at_1][angle.at_2]
        r_23 = bond_graph[angle.at_2][angle.at_3]
        coords_1 = atoms[angle.at_1].coords
        coords_2 = atoms[angle.at_2].coords
        coords_3 = atoms[angle.at_3].coords
        angle.set_aijk(geometry.get_a_ijk(coords_1, coords_2, coords_3, r_12, r_23))

def set_torsions(torsions, atoms, bond_graph):
    for torsion in torsions:
        r_12 = bond_graph[torsion.at_1][torsion.at_2]
        r_23 = bond_graph[torsion.at_2][torsion.at_3]
        r_34 = bond_graph[torsion.at_3][torsion.at_4]
        coords_1 = atoms[torsion.at_1].coords
        coords_2 = atoms[torsion.at_2].coords
        coords_3 = atoms[torsion.at_3].coords
        coords_4 = atoms[torsion.at_4].coords
        torsion.set_tijkl(geometry.get_t_ijkl(coords_1, coords_2, coords_3, coords_4, 
                                              r_12, r_23, r_34))
        
def set_outofplanes(outofplanes, atoms, bond_graph):
    for outofplane in outofplanes:
        r_31 = bond_graph[outofplane.at_3][outofplane.at_1]
        r_32 = bond_graph[outofplane.at_3][outofplane.at_2]
        r_34 = bond_graph[outofplane.at_3][outofplane.at_4]
        coords_1 = atoms[outofplane.at_1].coords
        coords_2 = atoms[outofplane.at_2].coords
        coords_3 = atoms[outofplane.at_3].coords
        coords_4 = atoms[outofplane.at_4].coords
        outofplane.set_oijkl(geometry.get_t_ijkl(coords_1, coords_2, coords_3, coords_4, 
                                                 r_31, r_32, r_34))