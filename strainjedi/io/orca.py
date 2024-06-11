from ase.vibrations.vibrations import VibrationsData
import numpy as np
from ase.units import Bohr, Hartree
from ase.calculators.orca import PointChargePotential
import re
import os
from ase.calculators.calculator import FileIOCalculator, Parameters, ReadError
from typing import Dict, Optional
from collections.abc import Iterable
from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointCalculator



class ORCA(FileIOCalculator):
    implemented_properties = ['energy', 'forces']

    if 'ORCA_COMMAND' in os.environ:
        command = os.environ['ORCA_COMMAND'] + ' PREFIX.inp > PREFIX.out'
    else:
        command = 'orca PREFIX.inp > PREFIX.out'

    default_parameters = dict(
        charge=0, mult=1,
        task='gradient',
        orcasimpleinput='tightscf PBE def2-SVP',
        orcablocks='%scf maxiter 200 end')

    def __init__(self, restart=None,
                 ignore_bad_restart_file=FileIOCalculator._deprecated,
                 label='orca', atoms=None, **kwargs):
        """ Modified ASE interface to ORCA 4 
        by Ragnar Bjornsson, Based on NWchem interface but simplified.
        Only supports energies and gradients (no dipole moments,
        orbital energies etc.) for now.

        For more ORCA-keyword flexibility, method/xc/basis etc.
        keywords are not used. Instead, two keywords:

            orcasimpleinput: str
                What you'd put after the "!" in an orca input file.
                Should in most cases contain "engrad" or method that
                writes the engrad file. If not (single point only),
                set the "task" parameter manually.
                Default is ``engrad tightscf PBE def2-SVP``.

            orcablock: str
                What you'd put in the "% ... end"-blocks.


        are used to define the ORCA simple-inputline and the ORCA-block input.
        This allows for more flexible use of any ORCA method or keyword
        available in ORCA instead of hardcoding stuff.

        Default parameters are:

            charge: 0

            mult: 1

            task: 'gradient'

        Point Charge IO functionality added by A. Dohn.
        """
        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, **kwargs)

        self.pcpot = None

    def set(self, **kwargs):
        changed_parameters = FileIOCalculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        p = self.parameters
        p.write(self.label + '.ase')
        p['label'] = self.label
        if self.pcpot:  # also write point charge file and add things to input
            p['pcpot'] = self.pcpot

        write_orca(atoms, **p)

    def read(self, label):
        FileIOCalculator.read(self, label)
        if not os.path.isfile(self.label + '.out'):
            raise ReadError

        with open(self.label + '.inp') as fd:
            for line in fd:
                if line.startswith('geometry'):
                    break
            symbols = []
            positions = []
            for line in fd:
                if line.startswith('end'):
                    break
                words = line.split()
                symbols.append(words[0])
                positions.append([float(word) for word in words[1:]])

        self.parameters = Parameters.read(self.label + '.ase')
        self.read_results()
    def read_results(self):
        self.read_energy()
        if self.parameters.task.find('gradient') > -1:
            self.read_forces()

    def read_energy(self):
        """Read Energy from ORCA output file."""
        with open(self.label + '.out', mode='r', encoding='utf-8') as fd:
            text = fd.read()
        # Energy:
        re_energy = re.compile(r"FINAL SINGLE POINT ENERGY.*\n")
        re_not_converged = re.compile(r"Wavefunction not fully converged")
        found_line = re_energy.finditer(text)
        for match in found_line:
            if not re_not_converged.search(match.group()):
                self.results['energy'] = float(
                    match.group().split()[-1]) * Hartree

    def read_forces(self):
        """Read Forces from ORCA output file."""
        if not os.path.isfile(self.label + '.engrad'):
            raise ReadError("Engrad file missing.")
        with open(f'{self.label}.engrad', 'r') as fd:
            lines = fd.readlines()
        getgrad = False
        gradients = []
        tempgrad = []
        for i, line in enumerate(lines):
            if line.find('# The current gradient') >= 0:
                getgrad = True
                gradients = []
                tempgrad = []
                continue
            if getgrad and "#" not in line:
                grad = line.split()[-1]
                tempgrad.append(float(grad))
                if len(tempgrad) == 3:
                    gradients.append(tempgrad)
                    tempgrad = []
            if '# The at' in line:
                getgrad = False
        self.results['forces'] = -np.array(gradients) * Hartree / Bohr

    def embed(self, mmcharges=None, **parameters):
        """Embed atoms in point-charges (mmcharges)
        """
        self.pcpot = PointChargePotential(mmcharges, label=self.label)
        return self.pcpot
        
def write_orca(atoms, **params):
    """Modified function to write ORCA input file, making optimizations possible
    """
    charge = params['charge']
    mult = params['mult']
    label = params['label']

    if 'pcpot' in params.keys():
        pcpot = params['pcpot']
        pcstring = '% pointcharges \"' +\
                   label + '.pc\"\n\n'
        params['orcablocks'] += pcstring
        pcpot.write_mmcharges(label)

    with open(label + '.inp', 'w') as fd:
        fd.write("! %s \n" % params['orcasimpleinput'])
        fd.write("%s \n" % params['orcablocks'])

        fd.write('*xyz')
        fd.write(" %d" % charge)
        fd.write(" %d \n" % mult)
        for atom in atoms:
            if atom.tag == 71:  # 71 is ascii G (Ghost)
                symbol = atom.symbol + ' : '
            else:
                symbol = atom.symbol + '   '
            fd.write(symbol +
                     str(atom.position[0]) + ' ' +
                     str(atom.position[1]) + ' ' +
                     str(atom.position[2]) + '\n')
        fd.write('*\n')
def read(filename):
    '''Read orca outputs.

    filename: str'''
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

def get_vibrations(label,atoms,indices=None):
    '''Read hessian.

    label: str
        Filename w/o .log.
    atoms: class
        Structure of which the frequency analysis was performed.
    indices: lst
        Indices of unconstrained atoms.
    Returns:
        VibrationsData object. 
    '''
    if indices==None:
        indices = range(len(atoms))
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
        '''
        atoms: class
            Structure with orca calculator'''
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
    '''
    Allowing ase to use Orca geometry optimizations
    
    '''
    keywords = {'task':'opt'}
