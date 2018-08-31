# -*- coding: utf-8 -*-

import numpy as np
from libdalton import energy, gradient
import matplotlib.pyplot as plt
    
x = np.arange(0.85, 4, 0.01)

partfun = lambda r: energy.get_e_vdw(r,1,1,1.5)
partfun2 = lambda r: energy.get_e_vdw(r,1,1,0)
y = list(map(partfun,x))
y2 = list(map(partfun2,x))

#partfun = lambda r: gradient.get_g_mag_vdw(r,1,1,1.5)
#partfun2 = lambda r: gradient.get_g_mag_vdw(r,1,1,0)
#y = list(map(partfun,x))
#y2 = list(map(partfun2,x))

plt.plot(x,y)
plt.plot(x,y2)
plt.show()

print(energy.get_e_vdw(120,1,1,200),gradient.get_g_mag_vdw(120,1,1,200))