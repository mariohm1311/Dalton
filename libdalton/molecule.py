
"""Classes and methods for molecular primitives used in the library."""


import numpy as np
from sklearn.neighbors import KDTree
import os

from libdalton import constants as const
from libdalton import param
from libdalton import gradient
from libdalton import energy
from libdalton import fileio
from libdalton import topology
from libdalton import geometry

class Atom:
    """Atom class for atomic geometry and parameter data.
  
    Initialize attributes to corresponding specified argument values, look up in
    parameter tables, or set to zero.
      
    Args:
        at_type (str): AMBER94 atom type.
        coords (float*): 3 cartesian coordinates [Angstrom].
        at_charge (float): atomic partial charge [e].
        at_ro (float): AMBER94's LJ 6-12 ro parameter [Angstrom].
        at_eps (float): AMBER94's LJ 6-12 eps parameter [kcal/mol].
        at_mass (float): relative atomic mass [amu or g/mol].
    
    Attributes:
        at_type (str): AMBER94 atom type.
        element (str): atomic element string.
        at_charge (float): atomic partial charge [e].
        at_ro (float): AMBER94's LJ 6-12 ro parameter [Angstrom].
        at_eps (float): AMBER94's LJ 6-12 eps parameter [kcal/mol].
        at_sreps (float): square root of AMBER94's LJ 6-12 eps parameter [kcal/mol**0.5].
        at_mass (float): relative atomic mass [amu or g/mol].
        at_rad (float): ionic/covalent radius [Angstrom].
        
        coords (float*): 3 cartesian coordinates [Angstrom].
        vels (float*): current velocity vector [Angstrom/ps].
        pvels (float*): past velocity vector [Angstrom/ps].
        accs (float*): current acceleration vector [Angstrom/(ps^2)].
        paccs (float*): past acceleration vector [Angstrom/(ps^2)].
    """
    
    def __init__(self, at_type, coords, at_charge, at_ro=None, at_eps=None):
        self.at_type = at_type
        self.element = fileio.get_element(at_type)
        self.set_charge(at_charge)
        
        if at_ro == None or at_eps == None:
            at_ro, at_eps = param.get_vdw_param(self.at_type)
        self.set_ro(at_ro)
        self.set_eps(at_eps)
        
        self.set_mass(param.get_at_mass(self.element))
        self.set_radius(param.get_at_rad(self.element))
        
        self.set_coords(coords)
        self.set_vels(np.zeros(const.NUMDIM))
        self.set_pvels(np.zeros(const.NUMDIM))
        self.set_accs(np.zeros(const.NUMDIM))
        self.set_paccs(np.zeros(const.NUMDIM))
        
        #initialize params
    
    def set_type(self, at_type):
        """set_type sets the AMBER94 atom type for a given atom object.
        
        Args:
            at_type (str): AMBER94 atom type.
        """
        self.at_type = at_type
    
    def set_charge(self, at_charge):
        """set_charge sets the partial charge for a given atom object.
        
        Args:
            at_charge (float): atomic partial charge [e].
        """
        self.at_charge = at_charge
    
    def set_radius(self, at_rad):
        """set_radius sets the ionic/covalent radius for a given atom object.
        
        Args:
            at_rad (float): ionic/covalent radius [Angstrom].
        """
        self.at_rad = at_rad
    
    def set_ro(self, at_ro):
        """set_ro sets AMBER94's Lennard-Jones 6-12 potential ro parameter for 
        a given atom object.
        
        Args:
            at_ro (float): AMBER94's LJ 6-12 ro parameter [Angstrom].
        """
        self.at_ro = at_ro
    
    def set_eps(self, at_eps):
        """set_ro sets AMBER94's Lennard-Jones 6-12 potential eps parameter for 
        a given atom object.
        
        Args:
            at_eps (float): AMBER94's LJ 6-12 eps parameter [kcal/mol].
        """
        self.at_eps = at_eps
        self.at_sreps = np.sqrt(at_eps)
    
    def set_mass(self, at_mass):
        """set_radius sets the atomic mass for a given atom object.
        
        Args:
            at_mass (float): relative atomic mass [amu or g/mol].
        """
        self.at_mass = at_mass
    
    def set_coords(self, coords):
        """set_coords sets the cartesian coordinates for a given atom object.
        
        Args:
            coords (float*): atomic coordinates [Angstrom].
        """
        self.coords = coords
        
    def set_coord(self, index, coord):
        """set_coord sets a single cartesian coordinate for a given atom object.
        
        Args:
            index (int): index for the coordinate [].
            coord (float): atomic coordinate [Angstrom].
        """
        self.coords[index] = coord
        
    def set_vels(self, vels):
        """set_vels sets the current velocity vector for a given atom object.
        
        Args:
            vels (float*): current velocity vector [Angstrom/ps].
        """
        self.vels = vels
        
    def set_pvels(self, pvels):
        """set_pvels sets the previous velocity vector for a given atom object.
        
        Args:
            pvels (float*): past velocity vector [Angstrom/ps].
        """
        self.pvels = pvels
        
    def set_accs(self, accs):
        """set_accs sets the current acceleration vector for a given atom object.
        
        Args:
            accs (float*): current acceleration vector [Angstrom/(ps^2)].
        """
        self.accs = accs
        
    def set_paccs(self, paccs):
        """set_paccs sets the previous acceleration vector for a given atom object.
        
        Args:
            paccs (float*): past acceleration vector [Angstrom/(ps^2)].
        """
        self.paccs = paccs
    
    
class Bond:
    """Bond class for atomic geometry and parameter data.
  
    Initialize attributes to corresponding specified argument values, look up in
    parameter tables, or set to zero.
      
    Args:
        at_1 (int): index of atom 1.
        at_2 (int): index of atom 2.
        r_ij (float): distance between atom 1 and 2 [Angstrom]. 
        r_eq (float): equilibrium bond length for atom 1 and 2 [Angstrom].
        k_b (float): bond spring constant for atom 1 and 2 [kcal/(mol*A^2)].
    
    Attributes:
        energy (float): energy of the bond [kcal/mol].
        grad_mag (float): signed magnitude of the energy gradient of the bond [kcal/(mol*A)].
    """
    
    def __init__(self, at_1, at_2, r_ij, r_eq, k_b):
        self.set_at1(at_1)
        self.set_at2(at_2)
        self.set_rij(r_ij)
        self.set_req(r_eq)
        self.set_kb(k_b)
               
    def set_at1(self, at_1):
        """set_at1 sets the index of atom 1 for later use.
        
        Args:
            at_1 (int): index of atom 1.
        """
        self.at_1 = at_1
    
    def set_at2(self, at_2):
        """set_at2 sets the index of atom 2 for later use.
        
        Args:
            at_2 (int): index of atom 2.
        """
        self.at_2 = at_2
        
    def set_rij(self, r_ij):
        """set_rij sets the distance between atom 1 and 2.
        
        Args:
            r_ij (float): distance between atom 1 and 2 [Angstrom]. 
        """
        self.r_ij = r_ij
        
    def set_req(self, r_eq):
        """set_req sets the equilibrium bond length for atom 1 and 2.
        
        Args:
            r_eq (float): equilibrium bond length for atom 1 and 2 [Angstrom].
        """
        self.r_eq = r_eq
        
    def set_kb(self, k_b):
        """set_kb sets the bond spring constant from the harmonic oscillator 
        model  of the bond between atom 1 and 2.
        
        Args:
            k_b (float): bond spring constant for atom 1 and 2 [kcal/(mol*A^2)].
        """
        self.k_b = k_b
        
    def get_energy(self):
        """get_energy calculates the bond energy as per the harmonic oscillator
        model from the distances and bonding parameters given for atom 1 and 2.
        """        
        self.energy = energy.get_e_bond(self.r_ij, self.r_eq, self.k_b)
    
    def get_gradient_mag(self):
        """get_gradient_mag calculates the magnitude of the energy gradient as per
        the harmonic oscillator model from the distances and bonding parameters
        given for atom 1 and 2.
        """
        self.grad_mag = gradient.get_g_mag_bond(self.r_ij, self.r_eq, self.k_b)


class Angle:
    """Angle class for atomic geometry and parameter data.
  
    Initialize attributes to corresponding specified argument values, look up in
    parameter tables, or set to zero.
      
    Args:
        at_1 (int): index of atom 1.
        at_2 (int): index of atom 2.
        at_3 (int): index of atom 3.
        a_ijk (float): angle between atoms 1, 2 and 3 [degrees]. 
        a_eq (float): equilibrium bond angle for atoms 1, 2 and 3 [degrees].
        k_a (float): angle spring constant for atoms 1, 2 and 3 [kcal/(mol*rad^2)].
    
    Attributes:
        energy (float): energy of the bond angle [kcal/mol].
        grad_mag (float): signed magnitude of the energy gradient of the angle [kcal/(mol*rad)].
    """
    
    def __init__(self, at_1, at_2, at_3, a_ijk, a_eq, k_a):
        self.set_at1(at_1)
        self.set_at2(at_2)
        self.set_at3(at_3)
        self.set_aijk(a_ijk)
        self.set_aeq(a_eq)
        self.set_ka(k_a)
        
    def set_at1(self, at_1):
        """set_at1 sets the index of atom 1 for later use.
        
        Args:
            at_1 (int): index of atom 1.
        """
        self.at_1 = at_1
    
    def set_at2(self, at_2):
        """set_at2 sets the index of atom 2 for later use.
        
        Args:
            at_2 (int): index of atom 2.
        """
        self.at_2 = at_2

    def set_at3(self, at_3):
        """set_at3 sets the index of atom 3 for later use.
        
        Args:
            at_3 (int): index of atom 3.
        """
        self.at_3 = at_3

    def set_aijk(self, a_ijk):
        """set_aijk sets the angle between atoms 1, 2 and 3.
        
        Args:
            a_ijk (float): angle between atoms 1, 2 and 3 [degrees]. 
        """
        self.a_ijk = a_ijk
        
    def set_aeq(self, a_eq):
        """set_aeq sets the equilibrium bond angle for atoms 1, 2 and 3.
        
        Args:
            a_eq (float): equilibrium angle for atoms 1, 2 and 3 [degrees]. 
        """
        self.a_eq = a_eq
        
    def set_ka(self, k_a):
        """set_ka sets the harmonic oscillator angle constant for atoms 1, 2 and 3.
        
        Args:
            k_a (float): angle constant for atoms 1, 2 and 3 [kcal/(mol*rad^2)]. 
        """
        self.k_a = k_a
    
    def get_energy(self):
        """get_energy calculates the angle energy as per the harmonic oscillator
        model from the angles and bonding parameters given for atoms 1, 2 and 3.
        """        
        self.energy = energy.get_e_angle(self.a_ijk, self.a_eq, self.k_a)
    
    def get_gradient_mag(self):
        """get_gradient_mag calculates the magnitude of the energy gradient as per
        the harmonic oscillator model from the angles and bonding parameters
        given for atoms 1, 2 and 3.
        """
        self.grad_mag = gradient.get_g_mag_angle(self.a_ijk, self.a_eq, self.k_a)    
        

class Torsion:
    def __init__(self, at_1, at_2, at_3, at_4, t_ijkl, v_n, gamma, n_fold, paths):
        self.set_at1(at_1)
        self.set_at2(at_2)
        self.set_at3(at_3)
        self.set_at4(at_4)
        self.set_tijkl(t_ijkl)
        self.set_vn(v_n)
        self.set_gamma(gamma)
        self.set_nfold(n_fold)
        self.set_paths(paths)
        
        self.get_energy()
        self.get_gradient_mag()
    
    def set_at1(self, at_1):
        """set_at1 sets the index of atom 1 for later use.
        
        Args:
            at_1 (int): index of atom 1.
        """
        self.at_1 = at_1
    
    def set_at2(self, at_2):
        """set_at2 sets the index of atom 2 for later use.
        
        Args:
            at_2 (int): index of atom 2.
        """
        self.at_2 = at_2

    def set_at3(self, at_3):
        """set_at3 sets the index of atom 3 for later use.
        
        Args:
            at_3 (int): index of atom 3.
        """
        self.at_3 = at_3
    
    def set_at4(self, at_4):
        """set_at4 sets the index of atom 4 for later use.
        
        Args:
            at_4 (int): index of atom 4.
        """
        self.at_4 = at_4
        
    def set_tijkl(self, t_ijkl):
        self.t_ijkl = t_ijkl
     
    def set_vn(self, v_n):
        self.v_n = v_n
        
    def set_gamma(self, gamma):
        self.gamma = gamma
    
    def set_nfold(self, n_fold):
        self.n_fold = n_fold
    
    def set_paths(self, paths):
        self.paths = paths
    
    def get_energy(self):
        self.energy = energy.get_e_torsion(self.t_ijkl, self.v_n, self.gamma,
                                           self.n_fold, self.paths)
    
    def get_gradient_mag(self):
        self.grad_mag = gradient.get_g_mag_torsion(self.t_ijkl, self.v_n, self.gamma,
                                                   self.n_fold, self.paths)


class OutOfPlane:
    def __init__(self, at_1, at_2, at_3, at_4, o_ijkl, v_n):
        self.set_at1(at_1)
        self.set_at2(at_2)
        self.set_at3(at_3)
        self.set_at4(at_4)
        self.set_oijkl(o_ijkl)
        self.set_vn(v_n)
    
    def set_at1(self, at_1):
        """set_at1 sets the index of atom 1 for later use.
        
        Args:
            at_1 (int): index of atom 1.
        """
        self.at_1 = at_1
    
    def set_at2(self, at_2):
        """set_at2 sets the index of atom 2 for later use.
        
        Args:
            at_2 (int): index of atom 2.
        """
        self.at_2 = at_2

    def set_at3(self, at_3):
        """set_at3 sets the index of atom 3 for later use.
        
        Args:
            at_3 (int): index of atom 3.
        """
        self.at_3 = at_3
    
    def set_at4(self, at_4):
        """set_at4 sets the index of atom 4 for later use.
        
        Args:
            at_4 (int): index of atom 4.
        """
        self.at_4 = at_4
        
    def set_oijkl(self, o_ijkl):
        self.o_ijkl = o_ijkl
     
    def set_vn(self, v_n):
        self.v_n = v_n
    
    def get_energy(self):
        self.energy = energy.get_e_outofplane(self.o_ijkl, self.v_n)
    
    def get_gradient_mag(self):
        self.grad_mag = gradient.get_g_mag_outofplane(self.o_ijkl, self.v_n)
                                               

class Molecule:
    def __init__(self, infile_name):
        self.infile = os.path.realpath(infile_name)
        self.indir = os.path.dirname(self.infile)
        self.filetype = self.infile.split('.')[-1]
        self.name = os.path.splitext(os.path.basename(self.infile))[0]
        
        self.atoms = []
        self.bonds = []
        self.angles = []
        self.torsions = []
        self.outofplanes = []
        
        self.coords_array = []
        self.atom_tree = []
        self.nonints = set()
        self.bond_graph = dict()
        
        self.n_atoms = 0
        self.n_bonds = 0
        self.n_angles = 0
        self.n_torsions = 0
        self.n_outofplanes = 0
        
        self.dielectric = 1.0
        self.mass = 0.0
        self.long_range_cutoff = (0.0, 0.0)
        self.k_bound = 250.0
        self.boundary = 1.0E10
        self.boundary_type = 'sphere'
        self.origin = np.zeros(const.NUMDIM)
        self.volume = float('inf')
        self.kinetic_calc_method = 'leapfrog'
        self.gradient_calc_method = 'analytic'
        self.temperature = 0.0
        self.pressure = 0.0
        self.virial = 0.0
        
        self.e_bonds = 0.0
        self.e_angles = 0.0
        self.e_torsions = 0.0
        self.e_outofplanes = 0.0
        self.e_vdw = 0.0
        self.e_elst = 0.0
        self.e_boundary = 0.0
        
        self.e_bonded = 0.0
        self.e_nonbonded = 0.0
        self.e_kinetic = 0.0
        self.e_potential = 0.0
        self.e_total = 0.0
        
        if self.filetype == 'xyzq':
            self.read_xyzq()
        elif self.filetype == 'prm':
            self.read_prm()
        
        self.g_bonds = np.zeros((self.n_atoms, const.NUMDIM))
        self.g_angles = np.zeros((self.n_atoms, const.NUMDIM))
        self.g_torsions = np.zeros((self.n_atoms, const.NUMDIM))
        self.g_outofplanes = np.zeros((self.n_atoms, const.NUMDIM))
        self.g_vdw = np.zeros((self.n_atoms, const.NUMDIM))
        self.g_elst = np.zeros((self.n_atoms, const.NUMDIM))
        self.g_boundary = np.zeros((self.n_atoms, const.NUMDIM))
        
        self.g_bonded = np.zeros((self.n_atoms, const.NUMDIM))
        self.g_nonbonded = np.zeros((self.n_atoms, const.NUMDIM))
        self.g_total = np.zeros((self.n_atoms, const.NUMDIM))
        
    def read_xyzq(self):
        infile_array = fileio.get_file_str_array(self.infile)
        self.atoms = fileio.get_atoms_from_xyzq(infile_array)
        self.n_atoms = len(self.atoms)
        self.get_topology()
        
        self.coords_array = np.zeros((self.n_atoms, const.NUMDIM))
        
        for i, atom in enumerate(self.atoms):
            self.mass += atom.at_mass
            self.coords_array[i] = atom.coords
        
        self.atom_tree = KDTree(self.coords_array, leaf_size=15)
    
    def read_prm(self):
        infile_array = fileio.get_file_str_array(self.infile)
        
        self.atoms = fileio.get_atoms_from_prm(infile_array)
        self.bonds = fileio.get_bonds_from_prm(infile_array, self.atoms)
        self.angles = fileio.get_angles_from_prm(infile_array, self.atoms)
        self.torsions = fileio.get_torsions_from_prm(infile_array, self.atoms)
        self.outofplanes = fileio.get_outofplanes_from_prm(infile_array, self.atoms)
        
        self.n_atoms = len(self.atoms)
        self.n_bonds = len(self.bonds) 
        self.n_angles = len(self.angles)
        self.n_torsions = len(self.torsions)
        self.n_outofplanes = len(self.outofplanes)
        
        self.coords_array = np.zeros((self.n_atoms, const.NUMDIM))
        
        for i, atom in enumerate(self.atoms):
            self.mass += atom.at_mass
            self.coords_array[i] = atom.coords
        
        self.atom_tree = KDTree(self.coords_array, leaf_size=15)
        self.bond_graph = topology.get_bond_graph_from_bonds(self.bonds, self.n_atoms)
        self.nonints = topology.get_noninteracting(self.bonds, self.angles, self.torsions)
    
    def set_kinetic_calc_method(self, kinetic_calc_method):
        if kinetic_calc_method in const.KINETICCALCMETHODS:
            self.kinetic_calc_method = kinetic_calc_method
        else:
            raise ValueError("Unexpected kinetic energy type: %s\n"
                             "Use 'nokinetic', 'direct' or 'leapfrog'"
                             % self.kinetic_calc_method)
    
    def set_gradient_calc_method(self, gradient_calc_method):
        self.gradient_calc_method = gradient_calc_method
    
    def set_long_range_cutoff(self, cutoff_tuple):
        self.long_range_cutoff = cutoff_tuple
        
    def get_topology(self):
        self.bond_graph = topology.get_bond_graph(self.atoms)
        self.bonds = topology.get_bonds(self.atoms, self.bond_graph)
        self.angles = topology.get_angles(self.atoms, self.bond_graph)
        self.torsions = topology.get_torsions(self.atoms, self.bond_graph)
        self.outofplanes = topology.get_outofplanes(self.atoms, self.bond_graph)
        self.nonints = topology.get_noninteracting(self.bonds, self.angles, self.torsions)
        
        self.n_bonds = len(self.bonds)
        self.n_angles = len(self.angles)
        self.n_torsions = len(self.torsions)
        self.n_outofplanes = len(self.outofplanes)
        
    def get_energy(self):
        self.e_bonds = energy.get_total_e_bonds(self.bonds)
        self.e_angles = energy.get_total_e_angles(self.angles)
        self.e_torsions = energy.get_total_e_torsions(self.torsions)
        self.e_outofplanes = energy.get_total_e_outofplanes(self.outofplanes)
        self.e_vdw, self.e_elst = energy.get_total_e_nonbonded(self.atoms, self.nonints,
                                                               self.dielectric, self.atom_tree,
                                                               self.long_range_cutoff)
        
        self.e_boundary = energy.get_total_e_boundary(self.atoms, self.origin,
                                                      self.k_bound, self.boundary,
                                                      self.boundary_type)
        
        self.e_kinetic = energy.get_total_e_kinetic(self.atoms, self.kinetic_calc_method)
        
        self.e_bonded = self.e_bonds + self.e_angles + self.e_torsions + self.e_outofplanes
        
        self.e_nonbonded = self.e_vdw + self.e_elst
        
        self.e_potential = self.e_bonded + self.e_nonbonded + self.e_boundary
        
        self.e_total = self.e_potential + self.e_kinetic
        
    def get_gradient(self):
        if self.gradient_calc_method == 'analytic':
            self.get_analytical_gradient()
        elif self.gradient_calc_method == 'numerical':
            self.get_numerical_gradient()
        else:
            raise ValueError("Unexpected gradient type: %s\nUse 'analytic' or 'numerical'"
                             % self.gradient_calc_method)
    
        self.g_bonded.fill(0.0)
        self.g_bonded += self.g_bonds + self.g_angles + self.g_torsions + self.g_outofplanes
        
        self.g_nonbonded.fill(0.0)
        self.g_nonbonded += self.g_vdw + self.g_elst
        
        self.g_total.fill(0.0)
        self.g_total += self.g_bonded + self.g_nonbonded + self.g_boundary
    
    def get_analytical_gradient(self):
        gradient.get_all_g_bond(self.g_bonds, self.bonds, self.atoms)
        gradient.get_all_g_angle(self.g_angles, self.angles, self.atoms, self.bond_graph)
        gradient.get_all_g_torsions(self.g_torsions, self.torsions, self.atoms, self.bond_graph)
        gradient.get_all_g_outofplanes(self.g_outofplanes, self.outofplanes, self.atoms, self.bond_graph)
        gradient.get_all_g_nonbonded(self.g_vdw, self.g_elst, self.atoms, self.nonints,
                                     self.dielectric, self.atom_tree, self.long_range_cutoff)
        gradient.get_all_g_boundary(self.g_boundary, self.atoms, self.origin, self.k_bound,
                                    self.boundary, self.boundary_type)
    
    def get_numerical_gradient(self):
        gradient.get_all_g_numerical(self)
    
    def get_temperature(self):
        self.temperature = energy.get_temperature(self.e_kinetic, self.n_atoms)
        
    def get_pressure(self):
        self.virial = gradient.get_virial(self.g_total, self.atoms)
        self.pressure = gradient.get_pressure(self.n_atoms, self.temperature,
                                              self.virial, self.volume)
        
    def get_volume(self):
        self.volume = geometry.get_volume(self.boundary, self.boundary_type)
    
    def update_internals(self):
        if not self.long_range_cutoff == (0.0, 0.0):
            self.coords_array.fill(0.0) 
            for i, atom in enumerate(self.atoms):
                self.coords_array[i] = atom.coords
#            self.atom_tree = KDTree(self.coords_array, leaf_size=15)
        
        topology.set_bonds(self.bonds, self.atoms, self.bond_graph)
        topology.set_angles(self.angles, self.atoms, self.bond_graph)
        topology.set_torsions(self.torsions, self.atoms, self.bond_graph)
        topology.set_outofplanes(self.outofplanes, self.atoms, self.bond_graph)
    
    def print_data(self):
        self.print_energy()
        self.print_geometry()
        self.print_bonds()
        self.print_angles()
        self.print_torsions()
        self.print_outofplanes()
    
    def print_energy(self):
        print(fileio.get_print_energy_string(self))
    
    def print_geometry(self):
        print(fileio.get_print_geometry_string(self.atoms))
    
    def print_bonds(self):
        print(fileio.get_print_bonds_string(self.bonds, self.atoms))
    
    def print_angles(self):
        print(fileio.get_print_angles_string(self.angles, self.atoms))
    
    def print_torsions(self):
        print(fileio.get_print_torsions_string(self.torsions, self.atoms))
    
    def print_outofplanes(self):
        print(fileio.get_print_outofplanes_string(self.outofplanes, self.atoms))
    
    def print_gradient(self):
        comment = self.gradient_calc_method.capitalize() + ' total gradient:'
        print(fileio.get_print_gradient_string(self.g_total, self.atoms, comment))
        

if __name__=='__main__':
    pass