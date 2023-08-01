import numpy as np
from ase.calculators.calculator import FileIOCalculator
from ase.calculators.calculator import SCFError
import ase.units
from ase.vibrations.data import VibrationsData
import ase.io
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
            if '$molecule' in line:
             
                positions=[]
                symbols=''
                next(lineiter)
                while True:
                    
                    line=next(lineiter).split()
                    if '$end' in line:
                     
                        atoms=Atoms(symbols,positions=positions)
                        break
                    symbols+=line[0]
                    positions.append((line[1:4]))


            elif 'SCF failed to converge' in line:
                raise SCFError()
            elif 'ERROR: alpha_min' in line:
                # Even though it is not technically a SCFError:
                raise SCFError()
            elif ' Total energy in the final basis set =' in line:
                convert = ase.units.Hartree
                energy = float(line.split()[8]) * convert
            elif ' Gradient of SCF Energy' in line:
                # Read gradient as 3 by N array and transpose at the end
                gradient = [[] for _ in range(3)]
                # Skip first line containing atom numbering
                next(lineiter)
                while True:
                    # Loop over the three Cartesian coordinates
                    for i in range(3):
                        # Cut off the component numbering and remove
                        # trailing characters ('\n' and stuff)
                        line = next(lineiter)[5:].rstrip()
                        # Cut in chunks of 12 symbols and convert into
                        # strings. This is preferred over string.split() as
                        # the fields may overlap for large gradients
                        gradient[i].extend(list(map(
                            float, [line[i:i + 12]
                                    for i in range(0, len(line), 12)])))

                    # After three force components we expect either a
                    # separator line, which we want to skip, or the end of
                    # the gradient matrix which is characterized by the
                    # line ' Max gradient component'.
                    # Maybe change stopping criterion to be independent of
                    # next line. Eg. if not lineiter.next().startswith(' ')
                    if ' Max gradient component' in next(lineiter):
                        # Minus to convert from gradient to force
                        forces = np.array(gradient).T * (
                            -ase.units.Hartree / ase.units.Bohr)
                        break
            elif 'Standard Nuclear Orientation (Angstroms)' in line:
                positions=[[] for _ in range(len(atoms))]
                next(lineiter)
                next(lineiter)
                a=True
                while a:
        
                    for i in range(len(atoms)):
                        # Cut off the component numbering and remove
                        # trailing characters ('\n' and stuff)
                        line = next(lineiter)[14:].rstrip()
                        # Cut in chunks of 12 symbols and convert into
                        # strings. This is preferred over string.split() as
                        # the fields may overlap for large gradients
                        
                        positions[i].extend(list(map(
                            float, [line[i:i + 17]
                                    for i in range(0, len(line), 17)])))

                    a=False 
                
                atoms.set_positions(positions)
            elif 'Z-matrix Print:' in line:
                break
        atoms.calc = SinglePointCalculator(
        atoms, energy=energy, dipole=dipole, forces=forces,
        )
        

    return atoms

def get_vibrations(label, atoms):
    filename = label + '.out'

    with open(filename, 'r') as fileobj:
        
        fileobj = fileobj.readlines()
        hess_line = 0
        for num, line in enumerate(fileobj, 1):
            if  'Mass-Weighted Hessian Matrix' in line:
                hess_line = num
                hess = []
                NCarts = 3 * len(atoms)
                if len(atoms.constraints)>0:
                    for l in atoms.constraints:
                        if l.__class__.__name__=='FixAtoms':
                            a=l.todict()
                            clist=np.array(a['kwargs']['indices'])
                            alist=np.delete(np.arange(0,len(atoms)),clist)
                        
                            NCarts = 3 * len(alist)
                            
                i=hess_line+2
                while  any(l.isalpha() for l in fileobj[i]) == False:
                    hess.append(fileobj[i])                     #read the lines
                    i += 1            
                hess=[l for l in hess if l !='\n']              #get rid of empty separator lines
        
                hess = [hess[l:l + NCarts] for l in range(0, len(hess), NCarts)]    #identify the chunks
                hess = [[k.split() for k in l] for l in hess]                       #
                
                hess = [np.array(l, dtype=('float64')) for l in hess]
                mass_weighted_hessian = hess[0]
                for l in range(1,len(hess)):
                    if np.size(hess[l],axis=1)>0:
                        mass_weighted_hessian = np.hstack((mass_weighted_hessian, hess[l]))
                #atoms.calc.results['hessian'] = mass_weighted_hessian  
                break
    mass_weights = np.repeat(atoms.get_masses()**0.5, 3)
    if 'alist' in locals():
        mass_weights=np.repeat(atoms.get_masses()[alist]**0.5, 3)
    mass_weights_matrix = np.outer(mass_weights, mass_weights[:, np.newaxis])
    hessian = np.multiply(mass_weighted_hessian, mass_weights_matrix)*(ase.units.Hartree / ase.units.Bohr**2) #qchem uses atomic units
    indices= np.arange(0,len(atoms))
    if 'alist' in locals():
        indices=alist
    return VibrationsData.from_2d(atoms, hessian,indices=indices)


class QChemDynamics:
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
                raise ValueError("{} requires a valid QChem calculator "
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
        calc_old = self.atoms.calc
        params_old = copy.deepcopy(self.calc.parameters)

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
        
        # self.calc.parameters = params_old
        # self.calc.reset()
        # if calc_old is not None:
        #     self.atoms.calc = calc_old
        self.atoms.calc=atoms.calc
        #ase.io.write('%s.json'%(self.calc.label),atoms)
        return converged


class QChemOptimizer(QChemDynamics):

    keywords = {'jobtype':'opt'}


   


          
            