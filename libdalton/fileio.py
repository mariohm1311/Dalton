""" TODO Doc

"""

import numpy as np
import os, sys

from libdalton import constants as const
from libdalton import molecule
from libdalton import geometry


# Formatting constants for print string creation functions.

_GEOM_PRINT_HEADER = ' Molecular Geometry and Non-bonded Parameters '
_GEOM_PRINT_BANNER_LENGTH = 65
_GEOM_PRINT_FIELDS = ['type', 'x', 'y', 'z', 'q', 'ro/2', 'eps']
_GEOM_PRINT_SPACES = [6, 5, 9, 9, 7, 6, 4]
_GEOM_PRINT_ABSENT = '\n No Atoms Detected'

_BOND_PRINT_HEADER = ' Bond Length Data '
_BOND_PRINT_BANNER_LENGTH = 57
_BOND_PRINT_FIELDS = ['k_b', 'r_eq', 'r_ij', 'types', 'energy', 'atoms']
_BOND_PRINT_SPACES = [10, 5, 5, 3, 4, 1]
_BOND_PRINT_ABSENT = '\n No Bonds Detected'

_ANGLE_PRINT_HEADER = ' Bond Angle Data '
_ANGLE_PRINT_BANNER_LENGTH = 58
_ANGLE_PRINT_FIELDS = ['k_a', 'a_eq', 'a_ijk', 'types', 'energy', 'atoms']
_ANGLE_PRINT_SPACES = [9, 3, 4, 4, 5, 2]
_ANGLE_PRINT_ABSENT = '\n No Bond Angles Detected'

_TORSION_PRINT_HEADER = ' Torsion Angle Data '
_TORSION_PRINT_BANNER_LENGTH = 67
_TORSION_PRINT_FIELDS = ['vn/2', 'gamma', 't_ijkl n p', 'types', 'energy',
                         'atoms']
_TORSION_PRINT_SPACES = [9, 2, 3, 5, 6, 3]
_TORSION_PRINT_ABSENT = '\n No Torsion Angles Detected'

_OUTOFPLANE_PRINT_HEADER = ' Out-of-plane Angle Data '
_OUTOFPLANE_PRINT_BANNER_LENGTH = 55
_OUTOFPLANE_PRINT_FIELDS = ['vn/2', 'o_ijkl', 'types', 'energy', 'atoms']
_OUTOFPLANE_PRINT_SPACES = [9, 2, 5, 6, 3]
_OUTOFPLANE_PRINT_ABSENT = '\n No Out-of-plane Angles Detected'

_ENERGY_PRINT_HEADER = ' Energy Values '
_ENERGY_PRINT_BANNER_LENGTH = 33
_ENERGY_PRINT_FIELDS = ['component', '[kcal/mol]']
_ENERGY_PRINT_SPACES = [3, 9]

# Formatting column headers and class attributes for energy components
_ENERGY_PRINT_LABELS = [
    'Total', 'Kinetic', 'Potential', 'Non-bonded', 'Bonded', 'Boundary',
    'van der Waals', 'Electrostatic', 'Bonds', 'Angles', 'Torsions',
    'Out-of-planes']
_ENERGY_PRINT_ATTRIBUTES = [
    'e_total', 'e_kinetic', 'e_potential', 'e_nonbonded', 'e_bonded',
    'e_boundary', 'e_vdw', 'e_elst', 'e_bonds', 'e_angles', 'e_torsions',
    'e_outofplanes']

_AVERAGE_PRINT_HEADER = ' Energy Component Properties [kcal/mol] '
_AVERAGE_PRINT_BANNER_LENGTH = 68
_AVERAGE_PRINT_FIELDS = ['component', 'avg', 'std', 'min', 'max']
_AVERAGE_PRINT_SPACES = [3, 11, 9, 9, 9]

# Syntax use messages for programs within the molecular mechanics repository
_PROGRAM_MESSAGES = {
  'mm.py': 'xyzq or prm file for molecular mechanics\n',
  'md.py': 'simluation file for molecular dynamics\n',
  'mc.py': 'simulation file for Metropolis Monte Carlo\n',
  'opt.py': 'optimization file for energy minimization\n',
  'ana.py': 'plot file for data analysis\n'}



def get_element(at_type):
    if len(at_type) == 1 or not at_type[1].islower():
        return at_type[0].upper()
    else:
        return at_type[0:2].capitalize()

def get_file_str_array(infile_name):
    if not os.path.exists(infile_name):
        raise ValueError(
                'Attempted to read from file which does not exist: {}'.format(infile_name))
    infile = open(infile_name, 'r')
    infile_data = infile.readlines()
    infile.close()
    
    infile_array = [line.split() for line in infile_data]   
    return infile_array

def get_atoms_from_xyzq(infile_array):
    atoms = []
    n_atoms = int(infile_array[0][0])
    
    for line in infile_array[2:n_atoms+2]:
        at_type = line[0]
        coords = np.array(list(map(float, line[1:1+const.NUMDIM])))
        
        try:
            at_charge = float(line[1+const.NUMDIM])
        except:
            raise IOError('Atomic partial charge value not present.')
        
        atoms.append(molecule.Atom(at_type, coords, at_charge))
        
    return atoms

def get_atom_from_prm(record):
    at_type = record[2]
    coords = np.array(tuple(map(float, record[3:3+const.NUMDIM])))
    at_charge, at_ro, at_eps = tuple(map(float, record[3+const.NUMDIM:6+const.NUMDIM]))
    return molecule.Atom(at_type, coords, at_charge, at_ro, at_eps)

def get_bond_from_prm(record, atoms):
    at_1, at_2 = (i-1 for i in map(int, record[1:3]))
    k_b, r_eq = tuple(map(float, record[3:5]))
    coords_1, coords_2 = (atoms[i].coords for i in (at_1, at_2))
    r_ij = geometry.get_r_ij(coords_1, coords_2)
    return molecule.Bond(at_1, at_2, r_ij, r_eq, k_b)

def get_angle_from_prm(record, atoms):
    at_1, at_2, at_3 = (i-1 for i in map(int, record[1:4]))
    k_a, a_eq = tuple(map(float, record[4:6]))
    coords_1, coords_2, coords_3 = (atoms[i].coords for i in (at_1, at_2, at_3))
    a_ijk = geometry.get_a_ijk(coords_1, coords_2, coords_3)
    return molecule.Angle(at_1, at_2, at_3, a_ijk, a_eq, k_a)

def get_torsion_from_prm(record, atoms):
    at_1, at_2, at_3, at_4 = (i-1 for i in map(int, record[1:5]))
    v_n, gamma = tuple(map(float, record[5:7]))
    n_fold, paths = tuple(map(int, record[7:9]))
    coords_1, coords_2, coords_3, coords_4 = (atoms[i].coords for i in (at_1, at_2, at_3, at_4))
    t_ijkl = geometry.get_t_ijkl(coords_1, coords_2, coords_3, coords_4)
    return molecule.Torsion(at_1, at_2, at_3, at_4, t_ijkl, v_n, gamma, n_fold, paths)

def get_outofplane_from_prm(record, atoms):
    at_1, at_2, at_3, at_4 = (i-1 for i in map(int, record[1:5]))
    v_n = float(record[5])
    coords_1, coords_2, coords_3, coords_4 = (atoms[i].coords for i in (at_1, at_2, at_3, at_4))
    o_ijkl = geometry.get_o_ijkl(coords_1, coords_2, coords_3, coords_4)
    return molecule.OutOfPlane(at_1, at_2, at_3, at_4, o_ijkl, v_n)

def get_atoms_from_prm(records):
    atoms = []
    
    for record in records:
        if not _is_correct_record_type(record, 'ATOM'):
            continue
        atoms.append(get_atom_from_prm(record))
    
    return atoms

def get_bonds_from_prm(records, atoms):
    bonds = []
    
    for record in records:
        if not _is_correct_record_type(record, 'BOND'):
            continue
        bonds.append(get_bond_from_prm(record, atoms))
    
    return bonds

def get_angles_from_prm(records, atoms):
    angles = []
    
    for record in records:
        if not _is_correct_record_type(record, 'ANGLE'):
            continue
        angles.append(get_angle_from_prm(record, atoms))
    
    return angles

def get_torsions_from_prm(records, atoms):
    torsions = []
    
    for record in records:
        if not _is_correct_record_type(record, 'TORSION'):
            continue
        torsions.append(get_torsion_from_prm(record, atoms))
    
    return torsions

def get_outofplanes_from_prm(records, atoms):
    outofplanes = []
    
    for record in records:
        if not _is_correct_record_type(record, 'OUTOFPLANE'):
            continue
        outofplanes.append(get_outofplane_from_prm(record, atoms))
    
    return outofplanes

def _is_correct_record_type(record, record_type):
    if not record:
        return False
    return record[0].upper() == record_type

def _get_banner_string(text, length, lead_lines, trail_lines):
    left_pad = (length - len(text)) // 2 - 1
    right_pad = length - len(text) - left_pad - 2
    
    out_str = lead_lines*'\n' + left_pad*'-' + text + right_pad*'-' + trail_lines*'\n'
    return out_str

def _get_padded_string(fields, spacings):
    out_str = []
    
    for field, spacing in zip(fields, spacings):
        out_str.append(spacing*' ' + field)
    
    out_str.append('\n')
    return ''.join(out_str)

def _get_header_string(header, banner_length, fields, spacings):
    out_str = []
    out_str.append(_get_banner_string(header, banner_length, 1, 1))
    out_str.append(_get_padded_string(fields, spacings))
    out_str.append(_get_banner_string('', banner_length, 0, 0))
    return ''.join(out_str)
    
def get_print_coords_xyz_string(atoms, comment='', total_chars=12, decimal_chars=6):
    out_str = ['%i\n%s\n' % (len(atoms), comment)]
    
    for atom in atoms:
        out_str.append('%-2s' % atom.element)
        
        for dim in range(const.NUMDIM):
            out_str.append(' %*.*f' % (total_chars, decimal_chars, atom.coords[dim]))
        
        out_str.append('\n')
    
    return ''.join(out_str)

def get_print_geometry_string(atoms):
    if not atoms:
        return _GEOM_PRINT_ABSENT
    
    out_str = [_get_header_string(_GEOM_PRINT_HEADER, _GEOM_PRINT_BANNER_LENGTH,
                                  _GEOM_PRINT_FIELDS, _GEOM_PRINT_SPACES), '\n']
    
    for i, atom in enumerate(atoms):
        out_str.append('%4i | %-2s' % (i+1, atom.at_type))
        
        for dim in range(const.NUMDIM):
            out_str.append('%10.4f' % atom.coords[dim])
        
        out_str.append(' %7.4f %7.4f %7.4f\n' % (atom.at_charge, atom.at_ro, atom.at_eps))
    
    return ''.join(out_str)

def get_print_bonds_string(bonds, atoms):
    if not bonds:
        return _BOND_PRINT_ABSENT
    
    out_str = [_get_header_string(_BOND_PRINT_HEADER, _BOND_PRINT_BANNER_LENGTH,
                                  _BOND_PRINT_FIELDS, _BOND_PRINT_SPACES)]

    for i, bond in enumerate(bonds):
        type_1, type_2 = atoms[bond.at_1].at_type, atoms[bond.at_2].at_type
        out_str.append('%4i | %7.2f %8.4f %8.4f (%2s-%2s) %8.4f (%i-%i)' % (
                i+1, bond.k_b, bond.r_eq, bond.r_ij, type_1, type_2, bond.energy, 
                bond.at_1+1, bond.at_2+1))
    
    return '\n'.join(out_str)

def get_print_angles_string(angles, atoms):
    if not angles:
        return _ANGLE_PRINT_ABSENT
    
    out_str = [_get_header_string(_ANGLE_PRINT_HEADER, _ANGLE_PRINT_BANNER_LENGTH,
                                  _ANGLE_PRINT_FIELDS, _ANGLE_PRINT_SPACES)]

    for i, angle in enumerate(angles):
        type_1, type_2, type_3 = (atoms[angle.at_1].at_type,
                                  atoms[angle.at_2].at_type,
                                  atoms[angle.at_3].at_type)
        out_str.append('%4i | %6.2f %7.3f %7.3f (%2s-%2s-%2s) %7.4f (%i-%i-%i)' %(
                i+1, angle.k_a, angle.a_eq, angle.a_ijk, type_1, type_2, type_3,
                angle.energy, angle.at_1+1, angle.at_2+1, angle.at_3+1))
    
    return '\n'.join(out_str)

def get_print_torsions_string(torsions, atoms):
    if not torsions:
        return _TORSION_PRINT_ABSENT
    
    out_str = [_get_header_string(_TORSION_PRINT_HEADER, _TORSION_PRINT_BANNER_LENGTH,
                                  _TORSION_PRINT_FIELDS, _TORSION_PRINT_SPACES)]

    for i, torsion in enumerate(torsions):
        type_1, type_2, type_3, type_4 = (atoms[torsion.at_1].at_type,
                                          atoms[torsion.at_2].at_type,
                                          atoms[torsion.at_3].at_type,
                                          atoms[torsion.at_3].at_type)
        out_str.append('%4i | %6.2f %6.1f %8.3f %i %i (%2s-%2s-%2s-%2s) %7.4f (%i-%i-%i-%i)'
                       % (i+1, torsion.v_n, torsion.gamma, torsion.t_ijkl, torsion.n_fold,
                          torsion.paths, type_1, type_2, type_3, type_4, torsion.energy,
                          torsion.at_1+1, torsion.at_2+1, torsion.at_3+1, torsion.at_4+1))
        
    return '\n'.join(out_str)

def get_print_outofplanes_string(outofplanes, atoms):
    if not outofplanes:
        return _OUTOFPLANE_PRINT_ABSENT
    
    out_str = [_get_header_string(_OUTOFPLANE_PRINT_HEADER, _OUTOFPLANE_PRINT_BANNER_LENGTH,
                                  _OUTOFPLANE_PRINT_FIELDS, _OUTOFPLANE_PRINT_SPACES)]

    for i, outofplane in enumerate(outofplanes):
        type_1, type_2, type_3, type_4 = (atoms[outofplane.at_1].at_type,
                                          atoms[outofplane.at_2].at_type,
                                          atoms[outofplane.at_3].at_type,
                                          atoms[outofplane.at_3].at_type)
        out_str.append('%4i | %6.2f %7.3f (%2s-%2s-%2s-%2s) %7.4f (%i-%i-%i-%i)' % (
                i+1, outofplane.v_n, outofplane.o_ijkl, type_1, type_2, type_3,
                type_4, outofplane.energy, outofplane.at_1+1, outofplane.at_2+1,
                outofplane.at_3+1, outofplane.at_4+1))
        
    return '\n'.join(out_str)

def get_print_energy_string(mol):
    out_str = [_get_header_string(_ENERGY_PRINT_HEADER, _ENERGY_PRINT_BANNER_LENGTH,
                                  _ENERGY_PRINT_FIELDS, _ENERGY_PRINT_SPACES)]
    
    for label, attribute in zip(_ENERGY_PRINT_LABELS, _ENERGY_PRINT_ATTRIBUTES):
        out_str.append('   %-13s | %10.4f' % (label, getattr(mol, attribute)))
    
    return '\n'.join(out_str)

def get_print_gradient_string(gradient, atoms, comment='', total_chars=12,
                              decimal_chars=6):
    out_str = ['\n %s\n' % comment]
    
    for i, grad in enumerate(gradient):
        out_str.append('%-2s' % atoms[i].at_type)
        
        for dim in range(const.NUMDIM):
            out_str.append(' %*.*f' % (total_chars, decimal_chars, grad[dim]))
        
        out_str.append('\n')
        
    return ''.join(out_str)

def get_print_averages(ana):
    out_str = [_get_header_string(_AVERAGE_PRINT_HEADER, _AVERAGE_PRINT_BANNER_LENGTH,
                                  _AVERAGE_PRINT_FIELDS, _AVERAGE_PRINT_SPACES)]
    
    pdict = const.PROPERTYDICTIONARY
    keys = sorted(list(pdict.keys()), key = lambda x: pdict[x][3])
    keys = [key for key in keys if (key in ana.prop and key in const.PROPERTYKEYS[:-2])]
    labels = [pdict[key][0] for key in keys]
    
    for key, label in zip(keys, labels):
        out_str.append('   %-13s | %11.4e %11.4e %11.4e %11.4e' % (
                label, ana.eavg[key], ana.estd[key], ana.emin[key], ana.emax[key]))
    
    return '\n'.join(out_str)

def validate_input(program_path):
    if (len(sys.argv) < 2):
        program_name = program_path.split('/')[-1]
        
        if program_name in _PROGRAM_MESSAGES:
            print('\nUsage: python %s INPUT_FILE\n' % program_name)
            print('INPUT_FILE: %s' % _PROGRAM_MESSAGES[program_name])
            sys.exit()
        else:
            raise ValueError('Program name not recognized: %s' % program_name)
        
    input_file = sys.argv[1]
    
    if os.path.isfile(input_file):
        return input_file
    else:
        raise FileNotFoundError('Specified input is not a file: %s' % input_file)
        
def get_optimization_data(opt):
    infile_array = get_file_str_array(opt.infile)
    cwd = os.getcwd()
    os.chdir(opt.indir)
    
    for i, line in enumerate(infile_array):
        if len(line) < 2:
            continue
        
        kwarg = line[0].lower()
        kwarg_val = line[1]
        
        if kwarg == 'molecule':
            opt.mol = molecule.Molecule(os.path.realpath(kwarg_val))
        elif kwarg == 'opttype':
            opt.opt_type = kwarg_val.lower()
        elif kwarg == 'optcriteria':
            opt.opt_str = kwarg_val.lower()
        elif kwarg == 'kinetic_calc_method':
            opt.kinetic_calc_method = kwarg_val.lower()
            opt.mol.set_kinetic_calc_method(opt.kinetic_calc_method)
        elif kwarg == 'gradient_calc_method':
            opt.gradient_calc_method = kwarg_val.lower()
            opt.mol.set_gradient_calc_method(opt.gradient_calc_method)
        elif kwarg == 'e_converge':
            opt.conv_delta_e = float(kwarg_val)
        elif kwarg == 'grms_converge':
            opt.conv_grad_rms = float(kwarg_val)
        elif kwarg == 'gmax_converge':
            opt.conv_grad_max = float(kwarg_val)
        elif kwarg == 'drms_converge':
            opt.conv_disp_rms = float(kwarg_val)
        elif kwarg == 'dmax_converge':
            opt.conv_disp_max = float(kwarg_val)
        elif kwarg == 'nmaxiter':
            opt.n_maxiter = float(kwarg_val)
        elif kwarg == 'geomout':
            opt.out_geom = os.path.realpath(kwarg_val)
        elif kwarg == 'energyout':
            opt.out_energy = os.path.realpath(kwarg_val)
            
    os.chdir(cwd)
        
def get_simulation_data(sim):
    infile_array = get_file_str_array(sim.infile)
    cwd = os.getcwd()
    os.chdir(sim.indir)
    
    for i, line in enumerate(infile_array):
        if len(line) < 2:
            continue
        
        kwarg = line[0].lower()
        kwarg_val = line[1]
        kwarg_arr = line[1:]
        
        if kwarg == 'molecule':
            sim.mol = molecule.Molecule(os.path.realpath(kwarg_val))
            sim.mol.set_long_range_cutoff(const.LONGRANGECUTOFF)
        elif kwarg == 'temperature': 
            sim.temperature = float(kwarg_val)
        elif kwarg == 'pressure':
            sim.pressure = float(kwarg_val)
        elif kwarg == 'vanderwaalscutoff':
            cutoff_tuple = (float(kwarg_val), sim.mol.long_range_cutoff[1])
            sim.mol.set_long_range_cutoff(cutoff_tuple)
        elif kwarg == 'electrostaticcutoff':
            cutoff_tuple = (sim.mol.long_range_cutoff[0], float(kwarg_val))
            sim.mol.set_long_range_cutoff(cutoff_tuple)
        elif kwarg == 'boundaryspring':
            sim.mol.k_bound = float(kwarg_val)
        elif kwarg == 'boundary':
            sim.mol.boundary = float(kwarg_val)
            sim.mol.get_volume()
        elif kwarg == 'boundarytype':
            sim.mol.boundary_type = kwarg_val.lower()
            sim.mol.get_volume()
        elif kwarg == 'kinetic_calc_method':
            sim.kinetic_calc_method = kwarg_val.lower()
            sim.mol.set_kinetic_calc_method(sim.kinetic_calc_method)
        elif kwarg == 'gradient_calc_method':
            sim.gradient_calc_method = kwarg_val.lower()
            sim.mol.set_gradient_calc_method(sim.gradient_calc_method)
        elif kwarg == 'origin':
            sim.mol.origin = list(map(float, kwarg_arr[:const.NUMDIM]))
        elif kwarg == 'totaltime':
            sim.total_time = float(kwarg_val)
        elif kwarg == 'totalconf':
            sim.total_conf = int(kwarg_val)
        elif kwarg == 'timestep':
            sim.timestep = float(kwarg_val)
        elif kwarg == 'geomtime':
            sim.geom_time = float(kwarg_val)
        elif kwarg == 'geomconf':
            sim.geom_conf = int(kwarg_val)
        elif kwarg == 'geomout':
            sim.out_geom = os.path.realpath(kwarg_val)
        elif kwarg == 'energytime':
            sim.energy_time = float(kwarg_val)
        elif kwarg == 'energyconf':
            sim.energy_conf = int(kwarg_val)
        elif kwarg == 'energyout':
            sim.out_energy = os.path.realpath(kwarg_val) 
        elif kwarg == 'statustime':
            sim.status_time = float(kwarg_val)
        elif kwarg == 'eqtime':
            sim.eq_time = float(kwarg_val)
        elif kwarg == 'eqrate':
            sim.eq_rate = float(kwarg_val)
        elif kwarg == 'randomseed':
            sim.random_seed = int(kwarg_val) % 2**32
    
    os.chdir(cwd)
    
def get_analysis_data(ana):
    infile_array = get_file_str_array(ana.infile)
    cwd = os.getcwd()
    os.chdir(ana.indir)
    
    for i, line in enumerate(infile_array):
        if len(line) < 2:
            continue
        
        kwarg = line[0].lower()
        kwarg_val = line[1]
        
        if kwarg == 'input':
            ana.sim_file = os.path.realpath(kwarg_val)
            ana.sim_dir = os.path.dirname(ana.sim_file)
        elif kwarg == 'simtype':
            ana.sim_type = kwarg_val.lower()
        elif kwarg == 'plotout':
            ana.out_plot = os.path.realpath(kwarg_val)
        elif kwarg == 'percentstart':
            ana.percent_start = float(kwarg_val)
        elif kwarg == 'percentstop':
            ana.percent_stop = float(kwarg_val)
        elif kwarg == 'plotdistrib':
            if kwarg_val.lower() == 'y':
                ana.plot_distrib = True
            else:
                ana.plot_distrib = False
    
    os.chdir(cwd)

def get_properties(prop_file):
    prop_array = get_file_str_array(prop_file)
    prop_keys = const.PROPERTYKEYS
    key1 = prop_keys[2]
    key_line = 0
    
    for i, prop in enumerate(prop_array):
        if key1 in prop:
            key_line = i
            break
    
    n_keys = len(prop_array[key_line]) - 1
    n_confs = 0
    excluded_lines = []
    
    for i, prop in enumerate(prop_array):
        if '#' in prop[0] or not len(prop) == n_keys:
            excluded_lines.append(i)
        else:
            n_confs += 1
    
    props = {}
    
    for j in range(n_keys):
        key = prop_array[key_line][j+1]
        props[key] = np.zeros(n_confs)
        confnum = 0
        
        for i, prop in enumerate(prop_array):
            if not i in excluded_lines:
                props[key][confnum] = float(prop[j])
                confnum += 1
    
    return props

def get_trajectory(traj_file):
    traj_array = get_file_str_array(traj_file)
    
    n_lines = len(traj_array)
    n_atoms = int(traj_array[0][0])
    n_confs = int(np.floor(n_lines / (n_atoms + 2)))
    
    traj = np.zeros((n_confs, n_atoms, const.NUMDIM))
    
    for p in range(n_confs):
        geom_start =p * (n_atoms+2)
       
        for i in range(n_atoms):
            atom_start = geom_start + i + 2
            
            for j in range(const.NUMDIM):
                traj[p][i][j] = float(traj_array[atom_start][j+1])
    
    return traj