import numpy as np

from libdalton.integrators.base_integrators import Base_Thermostatted_Integrator
from libdalton import constants as const


class Leapfrog_w_Berendsen_Thermostat(Base_Thermostatted_Integrator):
    def __init__(self, mol, target_temp, eq_rate=0.0):
        self.mol = mol
        self._timestep = None
        self.temperature = target_temp
        
        self.etemp = target_temp
        self.eq_rate = eq_rate
    
    def initialize(self):
        self.mol.get_energy()
        self.mol.get_gradient()
        self._update_accs()
        self._update_vels(0.5 * self._timestep)
        
    def step(self):
        self._update_coords(self._timestep)
        self.mol.get_gradient()
        self._update_accs()
        self._update_vels(self._timestep)
        
        self.mol.get_energy()
        self.mol.get_temperature()
        self.mol.get_pressure()
    
    def run_thermostat(self):
        tscale = self._timestep / max(self._timestep, self.eq_rate)
        tweight = 10.0 * self._timestep
        self.etemp = (self.etemp + tweight * self.mol.temperature) / (1.0 + tweight)
        vscale = 1.0 + tscale * (np.sqrt(self.temperature / self.etemp) - 1.0)
        
        for atom in self.mol.atoms:
            atom.set_vels(atom.vels * vscale)        
    
    def _update_accs(self):
        for i, atom in enumerate(self.mol.atoms):
            atom.set_paccs(atom.accs)
            atom.set_accs(-const.ACCCONV * self.mol.g_total[i] / atom.at_mass)
    
    def _update_vels(self, timestep):
        for i, atom in enumerate(self.mol.atoms):
            atom.set_pvels(atom.vels)
            atom.set_vels(atom.vels + atom.accs * timestep)
    
    def _update_coords(self, timestep):
        for atom in self.mol.atoms:
            atom.set_coords(atom.coords + atom.vels * timestep)
        
        self.mol.update_internals()
        
    @property
    def timestep(self):
        return self._timestep
    
    @timestep.setter
    def timestep(self, timestep):
        self._timestep = timestep