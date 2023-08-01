from ase.vibrations.vibrations import VibrationsData
import numpy as np
from ase.units import Bohr,Hartree
import os
from ase.calculators.orca import ORCA
from ase.calculators.calculator import FileIOCalculator, Parameters, ReadError
from typing import Dict, Optional
from collections.abc import Iterable
import copy
from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

def read(filename):

    energy=None
    forces=None
    dipole=None
    with open(filename, 'r') as fileobj:
        lineiter = iter(fileobj)
        for line in lineiter:
            if 'CARTESIAN COORDINATES (ANGSTROEM)' in line:
                    
                positions=[]
                symbols=''
                next(lineiter)
                
                while True:
                    
                    line=next(lineiter).split()
                    
                    if line ==[]:
                        
                        break
                    symbols+=line[0]
                    positions.append([float(word) for word in line[1:]])
                


            elif 'FINAL SINGLE POINT ENERGY' in line:
                convert = Hartree
                energy = float(line.split()[4]) * convert
 
     

    
           


        atoms=Atoms(symbols,positions=positions)    
        atoms.set_positions(positions)

        atoms.calc = SinglePointCalculator(
        atoms, energy=energy
        )
        

    return atoms

def read_hessian(label,atoms):
    if not os.path.isfile(label + '.hess'):
        raise ReadError("hess file missing.")
    """Read Forces from ORCA output file."""
    with open(f'{label}.hess', 'r') as fd:
        lines = fd.readlines()


            
    
    
    NCarts = 3 * len(atoms)
    hess=np.zeros((NCarts,NCarts))
    if len(atoms.constraints)>0:
        for l in atoms.constraints:
            if l.__class__.__name__=='FixAtoms':
                a=l.todict()
                clist=np.array(a['kwargs']['indices'])
                alist=np.delete(np.arange(0,len(atoms)),clist)
                
                NCarts = 3 * len(alist)
    for i, line in enumerate(lines):
        if '$hessian' in line:
            hess_line=i+3
    
    chunks=NCarts//5+1
    for i in range(chunks):
        for j in range(NCarts):
            
            rows=lines[round(hess_line+i*(NCarts+1)+j)].split()
            
            rows=[float(k.replace('D', 'e')) for k in rows]
        
            hess[j][i*5:i*5+len(rows)-1]=rows[1::]
    
    hess*=(Hartree / Bohr**2) 
    

    indices= np.arange(0,len(atoms))
    if 'alist' in locals():
        indices=alist

    return VibrationsData.from_2d(atoms,hess,indices)


class OrcaDynamics:
    calctype = 'optimizer'
    delete = ['force']
    keyword: Optional[str] = None
    special_keywords: Dict[str, str] = dict()

    def __init__(self, atoms, calc=None):
        self.atoms = atoms
        if calc is not None:
            self.calc = calc
        else:
            if self.atoms.calc is None:
                raise ValueError("{} requires a valid ORCA calculator "
                                 "object!".format(self.__class__.__name__))

            self.calc = self.atoms.calc

    def todict(self):
        return {'type': self.calctype,
                'optimizer': self.__class__.__name__}

    def delete_keywords(self, kwargs):
        """removes list of keywords (delete) from kwargs"""
        for d in self.delete:
            kwargs.pop(d, None)

    def set_keywords(self, kwargs):
        args = kwargs.pop(self.keyword, [])
        
        if isinstance(args, str):
            args = [args]
        elif isinstance(args, Iterable):
            args = list(args)

        for key, template in self.special_keywords.items():
            if key in kwargs:
                val = kwargs.pop(key)
                args.append(template.format(val))
        
        kwargs['jobtype'] = 'opt'

    def run(self, **kwargs):


        self.delete_keywords(kwargs)
        self.delete_keywords(self.calc.parameters)
        self.set_keywords(kwargs)

        self.calc.set(**self.keywords)
       
        self.atoms.calc = self.calc

        try:
            self.atoms.get_potential_energy()
        except OSError:
            converged = False
        else:
            converged = True

        atoms = read(self.calc.label + '.out')
        
        

        self.atoms.positions = atoms.positions
        

        self.atoms.calc=atoms.calc

        return converged


class OrcaOptimizer(OrcaDynamics):

    keywords = {'task':'opt'}