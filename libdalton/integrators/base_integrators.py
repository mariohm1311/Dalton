from abc import ABC, abstractmethod

class Base_Integrator(ABC):
    thermostatted = False
    
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def initialize(self, timestep):
        pass
    
    @abstractmethod
    def step(self, timestep):
        pass
       
    @property
    @abstractmethod
    def timestep(self):
        pass
    
    @timestep.setter
    @abstractmethod
    def timestep(self):
        pass
    

class Base_Thermostatted_Integrator(ABC):
    thermostatted = True
    
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def initialize(self, timestep):
        pass
    
    @abstractmethod
    def step(self, timestep):
        pass
    
    @abstractmethod
    def run_thermostat(self):
        pass
    
    @property
    @abstractmethod
    def timestep(self):
        pass
    
    @timestep.setter
    @abstractmethod
    def timestep(self):
        pass
    