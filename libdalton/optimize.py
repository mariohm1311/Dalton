# -*- coding: utf-8 -*-

import numpy as np
import os

from libdalton import constants as const
from libdalton import fileio

class Trajectory:
    def __init__(self, mol):
        self.n_atoms = mol.n_atoms
        self.n_steps = 0
        self.energy = []
        self.coords = []
        self.gradients = []
        
        self.append_step(mol)
    
    def append_step(self, mol):
        self.n_steps += 1
        self.coords.append(np.zeros((self.n_atoms, const.NUMDIM)))
        self.gradients.append(np.zeros((self.n_atoms, const.NUMDIM)))
        self.energy.append(mol.e_total)
        
        for i, atom in enumerate(mol.atoms):
            for dim in range(const.NUMDIM):
                self.coords[-1][i][dim] = atom.coords[dim]
                self.gradients[-1][i][dim] = mol.g_total[i][dim]


class Optimization:
    def __init__(self, infile_name):
        self.infile = os.path.realpath(infile_name)
        self.indir = os.path.dirname(self.infile)
        self.name = '.'.join(self.infile.split('/')[-1].split('.')[:-1])
        self.opt_type = 'sd'
        self.opt_str = 'default'
        self.kinetic_calc_method = 'nokinetic'
        self.gradient_calc_method = 'analytic'
        self.mol = []
        
        self.out_geom = 'geom.xyz'
        self.out_energy = 'energy.dat'
        
        self.delta_e = float('inf')
        self.grad_rms = float('inf')
        self.grad_max = float('inf')
        self.disp_rms = float('inf')
        self.disp_max = float('inf')
        
        self.conv_delta_e = const.OPTCRITERIAREFS['default'][0]
        self.conv_grad_rms = const.OPTCRITERIAREFS['default'][1]
        self.conv_grad_max = const.OPTCRITERIAREFS['default'][2]
        self.conv_disp_rms = const.OPTCRITERIAREFS['default'][3]
        self.conv_disp_max = const.OPTCRITERIAREFS['default'][4]
        
        self.must_converge = [True for i in range(5)]
        self.are_converged = [False for i in range(5)]
        self.is_converged = False
        self.n_maxiter = 200
        self.n_iter = 0
        self.n_subiter = 0
        
        self.disp_mag = 1.0E-4
        self.disp_deriv = 0.0
        
        self.read_data()
        self.set_opt_criteria()
        self._update_energy()
        self._update_gradient()
        self.traj = Trajectory(self.mol)
    
    def read_data(self):
        fileio.get_optimization_data(self)
    
    def set_opt_criteria(self):
        if self.opt_str in const.OPTCRITERIAREFS:
            opt_vals = const.OPTCRITERIAREFS[self.opt_str]
            self.conv_delta_e = opt_vals[0]
            self.conv_grad_rms = opt_vals[1]
            self.conv_grad_max = opt_vals[2]
            self.conv_disp_rms = opt_vals[3]
            self.conv_disp_max = opt_vals[4]
        else:
            raise ValueError("Optimization criteria string not recognized: %s\n"
                             "Use 'loose', 'default', 'tight', or 'verytight'" % 
                             (self.opt_str))
    
    def _update_opt_criteria(self):
        grad = self.traj.gradients[-1]
        disp = self.traj.coords[-1] - self.traj.coords[-2]
        self.delta_e = self.traj.energy[-1] - self.traj.energy[-2]
        self.grad_max = np.amax(grad)
        self.disp_max = np.amax(disp)
        self.grad_rms = np.sqrt(np.mean(grad**2))
        self.disp_rms = np.sqrt(np.mean(disp**2))
    
    def optimize(self):
        print('\n##################### INITIAL STATUS #####################')
        self.mol.print_energy()
        print('\n')
        
        self._open_output_files()
        
        while self.n_iter < self.n_maxiter and not self.is_converged:
            self.n_iter += 1
            self._choose_step_direction(self.opt_type)
            self._line_search(-1.0 * self.step_dir)
            self._update_energy()
            self._update_gradient()
            self.traj.append_step(self.mol)
            self._update_opt_criteria()
            self._check_convergence()
            self._print_status()
            
            if (self.n_iter+1) % 10 == 0:
                n_conv = sum([1 if crit == True else 0 for crit in self.are_converged])
                print('Iter: %i. Optimization energy = %.4e. Converged %i / 5.' % (
                        self.n_iter+1, self.mol.e_total, n_conv))
        
        self._close_output_files()
        
        print('\n#################### OPTIMIZED STATUS ####################')
        self.mol.print_energy()
    
    def _choose_step_direction(self, opt_type):
        if opt_type == 'sd':
            self._get_sd_step_dir()
        elif opt_type == 'cg':
            self._get_cg_step_dir()
        else:
            raise ValueError("Unexpected optimization type: %s\nUse 'sd' or 'cg'." %
                             opt_type)
    
    def _get_sd_step_dir(self):
        self.step_dir = self.mol.g_total
        
    def _get_cg_step_dir(self):
        if self.n_iter <= 1:
            self.hvec = self.mol.g_total
            gamma = 0.0
        else:
            v1 = self.traj.gradients[-1] - self.traj.gradients[-2]
            v1 = v1.reshape((1, const.NUMDIM * self.mol.n_atoms))
            v2 = self.traj.gradients[-1].reshape((const.NUMDIM * self.mol.n_atoms, 1))
            gamma = np.linalg.norm(np.dot(v1, v2))
            gamma *= 1.0 / np.linalg.norm(self.traj.gradients[-1])**2
            self.hvec = self.mol.g_total + gamma * self.hvec
        
        self.step_dir = self.hvec
    
    def _check_convergence(self):
        self.is_converged = True
        self.are_converged[0] = (abs(self.delta_e) < self.conv_delta_e)
        self.are_converged[1] = (self.grad_rms < self.conv_grad_rms)
        self.are_converged[2] = (self.grad_max < self.conv_grad_max)
        self.are_converged[3] = (self.disp_rms < self.conv_disp_rms)
        self.are_converged[4] = (self.disp_max < self.conv_disp_max)
        
        for i in range(5):
            if self.must_converge[i] and not self.are_converged[i]:
                self.is_converged = False
                break
    
    def _line_search(self, disp_vector):
        self.get_disp_deriv(self.disp_mag, disp_vector)
        disp_mag = self.disp_mag
        disp_sign = 1.0 if self.disp_deriv <= 0.0 else -1.0
        disp_mag *= disp_sign
        disp_sign_same = True
        ref_energy = self.mol.e_total
        
        self.n_subiter = 0
        while disp_sign_same:
            self.n_subiter += 1
            self._displace_coords(+1.0 * disp_mag, disp_vector)
            self.get_disp_deriv(disp_mag, disp_vector)
            self._displace_coords(-1.0 * disp_mag, disp_vector)
            
            if self.mol.e_total > ref_energy:
                disp_mag *= 0.5
                break
            
            old_disp_sign = disp_sign
            disp_sign = 1.0 if self.disp_deriv <= 0.0 else -1.0
            disp_sign_same = bool(disp_sign == old_disp_sign)
            disp_mag *= 2.0
        
        self.get_disp_deriv(disp_mag, disp_vector)
        self.adjust_disp_mag(self.n_subiter)
        
        numer = 1.0
        denom = 2.0
        
        for i in range(const.NUMLINESEARCHSTEPS):
            self.n_subiter += 1
            test_disp = disp_mag * numer / denom
            
            self._displace_coords(+1.0 * test_disp, disp_vector)
            self.get_disp_deriv(disp_mag * 2**i, disp_vector)
            self._displace_coords(-1.0 * test_disp, disp_vector)
            
            direc = 1.0 if self.disp_deriv < 0.0 else -1.0
            numer = 2.0 * numer + direc
            denom = 2.0 * denom
        
        disp_mag *= numer / denom
        self._displace_coords(+1.0 * disp_mag, disp_vector)
    
    def get_disp_deriv(self, disp_mag, disp_vector):
        num_disp = 0.01 * disp_mag
        
        # Using central finite difference method to calculate derivative
        self._displace_coords(+0.5 * num_disp, disp_vector)
        self._update_energy()
        ep_total = self.mol.e_total
        
        self._displace_coords(-1.0 * num_disp, disp_vector)
        self._update_energy()
        em_total = self.mol.e_total
        
        self._displace_coords(+0.5 * num_disp, disp_vector)
        self._update_energy()
        self.disp_deriv = (ep_total - em_total) / num_disp
    
    def adjust_disp_mag(self, n_subiter):
        if n_subiter == 1:
            self.disp_mag *= 1.0 / const.OPTSTEPADJUSTOR
        else:
            self.disp_mag *= const.OPTSTEPADJUSTOR
    
    def _update_energy(self):
        self.mol.get_energy()
    
    def _update_gradient(self):
        self.mol.get_gradient()
    
    def _update_coords(self, new_coords):
        for i, atom in enumerate(self.mol.atoms):
            atom.set_coords(new_coords)
        
        self.mol.update_internals()
    
    def _displace_coords(self, disp_mag, disp_vector):
        for i, atom in enumerate(self.mol.atoms):
            atom.set_coords(atom.coords + disp_mag * disp_vector[i])
        
        self.mol.update_internals()
        
    def print_energy_header(self):
        self.efile.write('#                      ')
        self.efile.write('%10.3e %9.3e %9.3e %9.3e %9.3e\n' % (
                self.conv_delta_e, self.conv_grad_max, self.conv_grad_rms, 
                self.conv_disp_max, self.conv_disp_rms))
        self.efile.write('# iter          energy    delta_e   grad_max')
        self.efile.write('   grad_rms   disp_max   disp_rms\n')
    
    def print_energy(self, n_iter):
        e = self.efile
        t = self.traj
        grad = t.gradients[n_iter]
        disp = t.coords[n_iter] - t.coords[max(0, n_iter-1)]
        delta_e = t.energy[n_iter] - t.energy[max(0, n_iter-1)]
        gmax = np.amax(grad)
        dmax = np.amax(disp)
        grms = np.sqrt(np.mean(grad**2))
        drms = np.sqrt(np.mean(disp**2))
        conv_str = [' ' for i in range(5)]
        
        if n_iter > 0:
            conv_str[0] = '*' if abs(delta_e) < self.conv_delta_e else ' '
            conv_str[1] = '*' if abs(grms) < self.conv_grad_rms else ' '
            conv_str[2] = '*' if abs(gmax) < self.conv_grad_max else ' '
            conv_str[3] = '*' if abs(drms) < self.conv_disp_rms else ' '
            conv_str[4] = '*' if abs(dmax) < self.conv_disp_max else ' '
        
        out_str = '%3i' % (n_iter)
        out_str += ' %18.12f' % t.energy[n_iter]
        out_str += ' %10.3e%s' % (delta_e, conv_str[0])
        out_str += ' %9.3e%s' % (grms, conv_str[1])
        out_str += ' %9.3e%s' % (gmax, conv_str[2])
        out_str += ' %9.3e%s' % (drms, conv_str[3])
        out_str += ' %9.3e%s' % (dmax, conv_str[4])
        
        e.write('%s\n' % out_str)
    
    def _print_status(self):
        comment = 'iter %i' % self.n_iter
        self.gfile.write(fileio.get_print_coords_xyz_string(self.mol.atoms,
                                                            comment, 14, 8))
        self.print_energy(self.n_iter)
        self.gfile.flush()
        self.efile.flush()
    
    def _open_output_files(self):
        if not os.path.exists(os.path.dirname(self.out_energy)):
            os.makedirs(os.path.dirname(self.out_energy))
        if not os.path.exists(os.path.dirname(self.out_geom)):
            os.makedirs(os.path.dirname(self.out_geom))

        self.efile = open(self.out_energy, 'w+')
        self.gfile = open(self.out_geom, 'w+')
        self.print_energy_header()
    
    def _close_output_files(self):
        self.efile.close()
        self.gfile.close()