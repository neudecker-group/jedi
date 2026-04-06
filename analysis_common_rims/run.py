# Script to run jedi analysis of butane structures from butane_angle.xyz

from strainjedi.jedi import Jedi
from ase.io import read, write
from ase.vibrations import VibrationsData

vib = VibrationsData.read('hessian.json')


traj = read('butane_angles.xyz@0:')
for i, struc in enumerate(traj):
    if i==0:
    	opt = struc
    else:
        dist = struc
        print()
        print()
        print()
        print(dist.get_angle(0, 1, 2).round(1))
        Jedi(opt, dist, vib).run(printout=True, indices = [0, 1, 2, 3])
