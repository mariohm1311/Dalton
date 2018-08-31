

import numpy as np
import os
import sys
import time

from libdalton import constants as const
from libdalton import fileio

class Simulation:
    def __init__(self, infile_name):
        self.infile = os.path.realpath(infile_name)
        self.indir = os.path.dirname(self.infile)
        
        self.mol = []
        self.temperature = 298.15
        self.pressure = 1.0
        self.kinetic_calc_method = 'leapfrog'
        self.gradient_calc_method = 'analytic'
        
        self.out_geom = 'geom.xyz'
        self.out_energy = 'energy.dat'
        
        self.status_time = 60.0
        self.random_seed = np.random.randint(2**31)
        
        self.eprintdig = 3
        self.eprintchar = 10
        self.gprintdig = 3
        self.gprintchar = 7
        
        self.read_data()
    
    def read_data(self):
        fileio.get_simulation_data(self)
        self.mol.set_kinetic_calc_method(self.kinetic_calc_method)
        self.mol.set_gradient_calc_method(self.gradient_calc_method)
        self.temperature += 1.0E-20
    
    def _open_output_files(self):
        if not os.path.exists(os.path.dirname(self.out_energy)):
            os.makedirs(os.path.dirname(self.out_energy))
        if not os.path.exists(os.path.dirname(self.out_geom)):
            os.makedirs(os.path.dirname(self.out_geom))

        self.efile = open(self.out_energy, 'w+')
        self.gfile = open(self.out_geom, 'w+')
        self._print_energy_header()
        self.start_time = time.time()
        
        if self.sim_type == 'md':
            self.gtime = 10E-10
            self.etime = 10E-10
        elif self.sim_type == 'mc':
            self.gconf = 0
            self.econf = 0
            self.dconf = 0
        
    def _close_output_files(self):
        self._print_status()
        self.efile.close()
        self.gfile.close()
    
    def _flush_buffers(self):
        self.efile.flush()
        self.gfile.flush()
        sys.stdout.flush()
    
    def _print_geometry(self):
        if self.sim_type == 'md':
            comment = '%.4f ps' % self.time
        elif self.sim_type == 'mc':
            comment = 'conf %i' % self.conf
        
        self.gfile.write(fileio.get_print_coords_xyz_string(self.mol.atoms, comment,
                                                            self.gprintchar,
                                                            self.gprintdig))
    
    def _print_val(self, val, total_char, dec_char, ptype='f', n_spaces=1):
        if ptype == 'f':
            self.efile.write('%*s%*.*f' % (n_spaces, '', total_char, dec_char, val))
        elif ptype == 'e':
            self.efile.write('%*s%*.*e' % (n_spaces, '', total_char, dec_char, val))
    
    def _print_energy_terms(self, total_char, dec_char, ptype):
        mol = self.mol
        energy_terms = [
                mol.e_kinetic, mol.e_potential, mol.e_nonbonded, mol.e_bonded,
                mol.e_boundary, mol.e_vdw, mol.e_elst, mol.e_bonds, mol.e_angles,
                mol.e_torsions, mol.e_outofplanes, mol.temperature, mol.pressure]
        
        if self.sim_type == 'mc':
            energy_terms = energy_terms[2:11]
        
        for term in energy_terms:
            self._print_val(term, total_char, dec_char, ptype)
    
    def _print_energy(self):
        if self.sim_type == 'md':
            self._print_val(self.time, self.tprintchar, self.tprintdig, 'f', 0)
        elif self.sim_type == 'mc':
            self._print_val(self.conf, self.cprintchar, 0, 'f', 0)
        
        self._print_val(self.mol.e_total, self.eprintchar+2, self.eprintdig+2, 'e')
        self._print_energy_terms(self.eprintchar, self.eprintdig, 'e')
        self.efile.write('\n')
    
    def _print_status(self):
        if self.sim_type == 'md':
            print('%.*f/%.*f ps' % (self.tprintdig, self.time, self.tprintdig,
                                    self.total_time), end='')
        elif self.sim_type == 'mc':
            print('%i/%i confs' % (self.conf, self.total_conf), end='')
        
        print(' as of %s' % time.strftime('%H:%M:%S'))
        self._flush_buffers()
    

class MolecularDynamics(Simulation):
    def __init__(self, infile_name):
        self.sim_type = 'md'
        
        self.total_time = 1.0
        self.timestep = 1.0E-3
        self.time = 1.0E-10
        self.eq_time = 0
        self.eq_rate = 0.5
        self.energy_time = 1.0E-2
        self.geom_time = 1.0E-2
        
        super().__init__(infile_name)
        
        if self.kinetic_calc_method == 'nokinetic':
            raise ValueError("Can't use 'nokinetic' mode for MD simulations.")
        
        self.tprintdig = 4
        self.tprintchar = 7
    
    def run(self):
        self._open_output_files()
        self._initialize_vels()
        self.mol.get_energy()
        self.mol.get_gradient()
        self._update_accs()
        self._check_print(0.0, print_all=True)
        self._update_vels(0.5 * self.timestep)
        
        while self.time < self.total_time:
            self._update_coords(self.timestep)
            self.mol.get_gradient()
            self._update_accs()
            self._update_vels(self.timestep)
            self.mol.get_energy()
            
            self.mol.get_temperature()
            self.mol.get_pressure()
            
            if self.time < self.eq_time:
                self._equilibrate_temp()
            
            self._check_print(self.timestep)
            self.time += self.timestep
        
        self._check_print(self.timestep)
        self._close_output_files()
        
    def _equilibrate_temp(self):
        tscale = self.timestep / max(self.timestep, self.eq_rate)
        tweight = 10.0 * self.timestep
        self.etemp = (self.etemp + tweight * self.mol.temperature) / (1.0 + tweight)
        vscale = 1.0 + tscale * (np.sqrt(self.temperature / self.etemp) - 1.0)
        
        for atom in self.mol.atoms:
            atom.set_vels(atom.vels * vscale)
    
    def _initialize_vels(self):
        if self.temperature:
            self.etemp = self.temperature
            np.random.seed(self.random_seed)
            sigma_base = np.sqrt(2.0 * const.RGAS * self.temperature / const.NUMDIM)
            
            for atom in self.mol.atoms:
                sigma = sigma_base / np.sqrt(atom.at_mass)
                atom.set_vels(np.random.normal(0.0, sigma, const.NUMDIM))
            
            self.mol.get_energy()
            self.mol.get_temperature()
            
            vscale = np.sqrt(self.temperature / self.mol.temperature)
            
            for atom in self.mol.atoms:
                atom.set_vels(atom.vels * vscale)
    
    def _update_coords(self, timestep):
        for atom in self.mol.atoms:
            atom.set_coords(atom.coords + atom.vels * timestep)
        
        self.mol.update_internals()
    
    def _update_vels(self, timestep):
        for i, atom in enumerate(self.mol.atoms):
            atom.set_pvels(atom.vels)
            atom.set_vels(atom.vels + atom.accs * timestep)
    
    def _update_accs(self):
        for i, atom in enumerate(self.mol.atoms):
            atom.set_paccs(atom.accs)
            atom.set_accs(-const.ACCCONV * self.mol.g_total[i] / atom.at_mass)
    
    def _check_print(self, timestep, print_all=False):
        if print_all or self.etime >= self.energy_time:
            self._print_energy()
            self.etime = 1.0E-10
        if print_all or self.gtime >= self.geom_time:
            self._print_geometry()
            self.gtime = 1.0E-10
        if print_all or (time.time() - self.stime) > self.status_time:
            self._print_status()
            self.stime = time.time()
        
        self.etime += timestep
        self.gtime += timestep
    
    def _print_energy_header(self):
        e = self.efile
        e.write('#\n# INPUTFILE %s' % (self.infile))
        e.write('\n#\n# -- INPUT DATA --\n#')
        e.write('\n# MOLFILE %s' % (self.mol.infile))
        e.write('\n# ENERGYOUT %s' % (self.out_energy))
        e.write('\n# GEOMOUT %s' % (self.out_geom))
        e.write('\n# RANDOMSEED %i' % (self.random_seed))
        e.write('\n# TEMPERATURE %.6f K' % (self.temperature))
        e.write('\n# BOUNDARY %.6f A' % (self.mol.boundary))
        e.write('\n# BOUNDARYSPRING %.6f kcal/(mol*A^2)' % (self.mol.k_bound))
        e.write('\n# BOUNDARYTYPE %s' % (self.mol.boundary_type))
        e.write('\n# STATUSTIME %.6f s' % (self.status_time))
        e.write('\n# ENERGYTIME %.6f ps' % (self.energy_time))
        e.write('\n# GEOMTIME %.6f ps' % (self.geom_time))
        e.write('\n# TOTALTIME %.6f ps' % (self.total_time))
        e.write('\n# TIMESTEP %.6f ps' % (self.timestep))
        e.write('\n# EQTIME %.6f ps' % (self.eq_time))
        e.write('\n# EQRATE %.6f ps' % (self.eq_rate))
        e.write('\n#\n# -- ENERGY DATA --\n#')
        e.write('\n# energy terms [kcal/mol]\n#  time      e_total      ')
        e.write('e_kin      e_pot  e_nonbond   e_bonded e_boundary      ')
        e.write('e_vdw     e_elst     e_bond    e_angle     e_tors      ')
        e.write('e_oop       temp      press\n')
        
class MonteCarlo(Simulation):
    def __init__(self, infile_name):
        self.sim_type = 'mc'
        
        self.total_conf = 1000
        self.conf = 0
        self.n_accept = 0
        self.n_reject = 0
        
        self.disp_mag = 0.1
        self.disp_inc = np.log(2.0)
        
        self.disp_conf = 100
        self.energy_conf = 100
        self.geom_conf = 100
        
        super().__init__(infile_name)
        self.rand_disp = np.zeros((self.mol.n_atoms, const.NUMDIM))
        self.cprintchar = 7
        
        self.mol.set_kinetic_calc_method('nokinetic')
    
    def run(self):
        self._open_output_files()
        self._zero_vels()
        np.random.seed(self.random_seed)
        
        self.mol.get_energy()
        self._check_print(0, print_all=True)
        prev_energy = self.mol.e_total
        
        while self.conf < self.total_conf:
            self._get_rand_disp()
            self._displace_coords(self.rand_disp)
            
            self.mol.get_energy()
            delta_e = self.mol.e_total - prev_energy
            bf = min(1.0, np.exp(-delta_e / (const.KB * self.temperature)))
            
            if bf >= np.random.random():
                self._check_print(1)
                self.conf += 1
                self.n_accept += 1
                prev_energy = self.mol.e_total
            else:
                self._displace_coords(-self.rand_disp)
                self.n_reject += 1
            
            self._check_disp()
            
        self._check_print(0)
        self._close_output_files()
        
    def _zero_vels(self):
        for atom in self.mol.atoms:
            atom.set_vels(np.zeros(const.NUMDIM))
            
    def _get_rand_disp(self):
        self.rand_disp.fill(0.0)
        self.rand_disp = np.random.normal(0.0, self.disp_mag, size=self.rand_disp.shape)
    
    def _displace_coords(self, disp_vector):
        for i, atom in enumerate(self.mol.atoms):
            atom.set_coords(atom.coords + disp_vector[i])
        
        self.mol.update_internals()
    
    def _change_disp(self):
        p_accept = float(self.n_accept) / float(self.n_reject + self.n_accept)
        self.n_accept, self.n_reject = 0, 0
        self.disp_mag *= np.exp(2.0 * self.disp_inc * (p_accept - 0.5))
        
    def _check_print(self, confstep, print_all=False):
        if print_all or self.econf >= self.energy_conf:
            self._print_energy()
            self.econf = 0
        if print_all or self.gconf >= self.geom_conf:
            self._print_geometry()
            self.gconf = 0
        if print_all or (time.time() - self.stime) > self.status_time:
            self._print_status()
            self.stime = time.time()
        
        self.econf += confstep
        self.gconf += confstep
    
    def _check_disp(self):
        if self.dconf >= self.disp_conf:
            self._change_disp()
            self.dconf = 0
        
        self.dconf += 1
        
    def _print_energy_header(self):
        e = self.efile
        e.write('#\n# INPUTFILE %s' % (self.infile))
        e.write('\n#\n# -- INPUT DATA --\n#')
        e.write('\n# MOLFILE %s' % (self.mol.infile))
        e.write('\n# ENERGYOUT %s' % (self.out_energy))
        e.write('\n# GEOMOUT %s' % (self.out_geom))
        e.write('\n# RANDOMSEED %i' % (self.random_seed))
        e.write('\n# TEMPERATURE %.6f K' % (self.temperature))
        e.write('\n# BOUNDARY %.6f A' % (self.mol.boundary))
        e.write('\n# BOUNDARYSPRING %.6f kcal/(mol*A^2)' % (self.mol.k_bound))
        e.write('\n# BOUNDARYTYPE %s' % (self.mol.boundary_type))
        e.write('\n# STATUSTIME %.6f s' % (self.status_time))
        e.write('\n# ENERGYCONF %i' % (self.energy_conf))
        e.write('\n# GEOMCONF %i' % (self.geom_conf))
        e.write('\n# TOTALCONF %i' % (self.total_conf))
        e.write('\n#\n# -- ENERGY DATA --\n#')
        e.write('\n# energy terms [kcal/mol] vs. configuration\n')
        e.write('#  conf        e_pot  e_nonbond   ')
        e.write('e_bonded e_boundary      e_vdw     e_elst     e_bond    ')
        e.write('e_angle     e_tors      e_oop\n')
    