import sys

from ase.io.gaussian import _compare_merge_configs
from ase.vibrations.vibrations import VibrationsData
import re
import numpy as np
from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.units import Hartree, Bohr
from copy import deepcopy
from ase.units import Bohr, Hartree
from ase.io.gaussian import _format_output_type, _xc_to_method, _check_problem_methods,_pop_link0_params,_format_addsec,_format_basis_set,_format_method_basis,_format_route_params,_get_extra_section_params,_get_molecule_spec
import copy
from ase.calculators.calculator import FileIOCalculator
import ase.calculators.gaussian as gaussian

_re_l716 = re.compile(r'^\s*\(Enter .+l716.exe\)$')
_re_forceblock = re.compile(r'^\s*Center\s+Atomic\s+Forces\s+\S+\s*$')
_re_atom = re.compile(
    r'^\s*\S+\s+(\S+)\s+(?:\S+\s+)?(\S+)\s+(\S+)\s+(\S+)\s*$'
)
        

def read_gaussian_out(label, index=-1):
    '''
    modified for reading gaussian geometry optimizations
   label: str
        filename w/o .log.
    '''

    with open(label+'.log','r') as fd:
        
        configs = []
        atoms = None
        energy = None
        dipole = None
        forces = None
        for line in fd:
            line = line.strip()
            if line.startswith(r'1\1\GINC'):
                # We've reached the "archive" block at the bottom, stop parsing
                break

            if (line == 'Input orientation:'
                    or line == 'Z-Matrix orientation:'
                    or line == "Standard orientation:"):
                if atoms is not None:
                    atoms.calc = SinglePointCalculator(
                        atoms, energy=energy, dipole=dipole, forces=forces,
                    )
                    _compare_merge_configs(configs, atoms)
                atoms = None
                #energy = None
                dipole = None
                forces = None

                numbers = []
                positions = []
                pbc = np.zeros(3, dtype=bool)
                cell = np.zeros((3, 3))
                npbc = 0
                # skip 4 irrelevant lines
                for _ in range(4):
                    fd.readline()
                while True:
                    match = _re_atom.match(fd.readline())
                    if match is None:
                        break
                    number = int(match.group(1))
                    pos = list(map(float, match.group(2, 3, 4)))
                    if number == -2:
                        pbc[npbc] = True
                        cell[npbc] = pos
                        npbc += 1
                    else:
                        numbers.append(max(number, 0))
                        positions.append(pos)
                atoms = Atoms(numbers, positions, pbc=pbc, cell=cell)
            elif (line.startswith('Energy=')
                    or line.startswith('SCF Done:')):
                # Some semi-empirical methods (Huckel, MINDO3, etc.),
                # or SCF methods (HF, DFT, etc.)
                energy = float(line.split('=')[1].split()[0].replace('D', 'e'))
                energy *= Hartree
            elif (line.startswith('E2 =') or line.startswith('E3 =')
                    or line.startswith('E4(') or line.startswith('DEMP5 =')
                    or line.startswith('E2(')):
                # MP{2,3,4,5} energy
                # also some double hybrid calculations, like B2PLYP
                energy = float(line.split('=')[-1].strip().replace('D', 'e'))
                energy *= Hartree
            elif line.startswith('Wavefunction amplitudes converged. E(Corr)'):
                # "correlated method" energy, e.g. CCSD
                energy = float(line.split('=')[-1].strip().replace('D', 'e'))
                energy *= Hartree
            elif _re_l716.match(line):
                # Sometimes Gaussian will print "Rotating derivatives to
                # standard orientation" after the matched line (which looks like
                # "(Enter /opt/gaussian/g16/l716.exe)", though the exact path
                # depends on where Gaussian is installed). We *skip* the dipole
                # in this case, because it might be rotated relative to the input
                # orientation (and also it is numerically different even if the
                # standard orientation is the same as the input orientation).
                line = fd.readline().strip()
                if not line.startswith('Dipole'):
                    continue
                dip = line.split('=')[1].replace('D', 'e')
                tokens = dip.split()
                dipole = []
                # dipole elements can run together, depending on what method was
                # used to calculate them. First see if there is a space between
                # values.
                if len(tokens) == 3:
                    dipole = list(map(float, tokens))
                elif len(dip) % 3 == 0:
                    # next, check if the number of tokens is divisible by 3
                    nchars = len(dip) // 3
                    for i in range(3):
                        dipole.append(float(dip[nchars * i:nchars * (i + 1)]))
                else:
                    # otherwise, just give up on trying to parse it.
                    dipole = None
                    continue
                # this dipole moment is printed in atomic units, e-Bohr
                # ASE uses e-Angstrom for dipole moments.
                dipole = np.array(dipole) * Bohr
            elif _re_forceblock.match(line):
                # skip 2 irrelevant lines
                fd.readline()
                fd.readline()
                forces = []
                while True:
                    match = _re_atom.match(fd.readline())
                    if match is None:
                        break
                    forces.append(list(map(float, match.group(2, 3, 4))))
                forces = np.array(forces) * Hartree / Bohr
        if atoms is not None:
            atoms.calc = SinglePointCalculator(
                atoms, energy=energy, dipole=dipole, forces=forces,
            )
            _compare_merge_configs(configs, atoms)
        
        return configs[index]

def get_vibrations(label,atoms,indices=None):
    """
    Read hessian.

    label: str
        filename w/o .log.
    atoms: class
        Structure of which the frequency analysis was performed.

    Returns:
        VibrationsData object.
    """
    if indices is None:
        indices = range(len(atoms))
    imaginary_freq_pattern = r'\**\s+(\d+)\s+imaginary frequencies \(negative Signs\)\s*\**'
    _re_hessblock = re.compile(r'^\s*Force\s+constants\s+in\s+Cartesian\s+coordinates:\s*$') #TODO not used
    output = label+'.log'
    with open(output, 'r') as fd:
        content = fd.read()
        match = re.search(imaginary_freq_pattern, content)
        if match:
            print(f'Found {match.group(1)} imaginary frequencies in {output}. Jedi Analysis can not be performed.')
            sys.exit()
        lines = content.splitlines()

    hess_line = 0
    NCarts = 3 * len(atoms)
    if len(atoms.constraints) > 0:
        for l in atoms.constraints:
            if l.__class__.__name__ == 'FixAtoms':
                a = l.todict()
                clist = np.array(a['kwargs']['indices'])
                alist = np.delete(np.arange(0, len(atoms)), clist)
                NCarts = 3 * len(alist)
    hess = np.zeros((NCarts, NCarts))
    for num, line in enumerate(lines, 1):
        if 'Force constants in Cartesian coordinates:' in line:
            hess_line = num+1
    
    chunks = NCarts//5+1
    for i in range(chunks):
        for j in range(NCarts-i*5):
            # TODO is there any occasion where it actually needs round()? isn't int() also possible
            rows = lines[round(hess_line+i*(NCarts+1)-sum(np.linspace(0, i-1, i)*5)+j)].split()
            rows = [float(k.replace('D', 'e')) for k in rows]
            hess[j+i*5][i*5:i*5+len(rows)-1]=rows[1::]
    hess = hess+hess.T-np.diag(np.diag(hess))
    hess *= (Hartree / Bohr**2)
    return VibrationsData.from_2d(atoms, hess, indices)



def write_gaussian_in(fd, atoms, properties=['energy'],
                      method=None, basis=None, fitting_basis=None,
                      output_type='P', basisfile=None, basis_set=None,
                      xc=None, charge=None, mult=None, extra=None,
                      ioplist=None, addsec=None, spinlist=None,
                      zefflist=None, qmomlist=None, nmagmlist=None,
                      znuclist=None, radnuclearlist=None,
                      **params):
    '''
    Generates a Gaussian input file, function modified from ASE

    Parameters
    -----------
    fd: file-like
        where the Gaussian input file will be written
    atoms: Atoms
        Structure to write to the input file
    properties: list
        Properties to calculate
    method: str
        Level of theory to use, e.g. ``hf``, ``ccsd``, ``mp2``, or ``b3lyp``.
        Overrides ``xc`` (see below).
    xc: str
        Level of theory to use. Translates several XC functionals from
        their common name (e.g. ``PBE``) to their internal Gaussian name
        (e.g. ``PBEPBE``).
    basis: str
        The basis set to use. If not provided, no basis set will be requested,
        which usually results in ``STO-3G``. Maybe omitted if basisfile is set
        (see below).
    fitting_basis: str
        The name of the fitting basis set to use.
    output_type: str
        Level of output to record in the Gaussian
        output file - this may be ``N``- normal or ``P`` -
        additional.
    basisfile: str
        The name of the basis file to use. If a value is provided, basis may
        be omitted (it will be automatically set to 'gen')
    basis_set: str
        The basis set definition to use. This is an alternative
        to basisfile, and would be the same as the contents
        of such a file.
    charge: int
        The system charge. If not provided, it will be automatically
        determined from the ``Atoms`` object’s initial_charges.
    mult: int
        The system multiplicity (``spin + 1``). If not provided, it will be
        automatically determined from the ``Atoms`` object’s
        ``initial_magnetic_moments``.
    extra: str
        Extra lines to be included in the route section verbatim.
        It should not be necessary to use this, but it is included for
        backwards compatibility.
    ioplist: list
        A collection of IOPs definitions to be included in the route line.
    addsec: str
        Text to be added after the molecular geometry specification, e.g. for
        defining masses with ``freq=ReadIso``.
    spinlist: list
        A list of nuclear spins to be added into the nuclear
        propeties section of the molecule specification.
    zefflist: list
        A list of effective charges to be added into the nuclear
        propeties section of the molecule specification.
    qmomlist: list
        A list of nuclear quadropole moments to be added into
        the nuclear propeties section of the molecule
        specification.
    nmagmlist: list
        A list of nuclear magnetic moments to be added into
        the nuclear propeties section of the molecule
        specification.
    znuclist: list
        A list of nuclear charges to be added into the nuclear
        propeties section of the molecule specification.
    radnuclearlist: list
        A list of nuclear radii to be added into the nuclear
        propeties section of the molecule specification.
    params: dict
        Contains any extra keywords and values that will be included in either
        the link0 section or route section of the gaussian input file.
        To be included in the link0 section, the keyword must be one of the
        following: ``mem``, ``chk``, ``oldchk``, ``schk``, ``rwf``,
        ``oldmatrix``, ``oldrawmatrix``, ``int``, ``d2e``, ``save``,
        ``nosave``, ``errorsave``, ``cpu``, ``nprocshared``, ``gpucpu``,
        ``lindaworkers``, ``usessh``, ``ssh``, ``debuglinda``.
        Any other keywords will be placed (along with their values) in the
        route section.
    '''

    params = deepcopy(params)

    if properties is None:
        properties = ['energy']

    output_type = _format_output_type(output_type)

    # basis can be omitted if basisfile is provided
    if basis is None:
        if basisfile is not None or basis_set is not None:
            basis = 'gen'

    # determine method from xc if it is provided
    if method is None:
        if xc is not None:
            method = _xc_to_method.get(xc.lower(), xc)

    # If the user requests a problematic method, rather than raising an error
    # or proceeding blindly, give the user a warning that the results parsed
    # by ASE may not be meaningful.
    if method is not None:
        _check_problem_methods(method)

    # determine charge from initial charges if not passed explicitly
    if charge is None:
        charge = atoms.get_initial_charges().sum()

    # determine multiplicity from initial magnetic moments
    # if not passed explicitly
    if mult is None:
        mult = atoms.get_initial_magnetic_moments().sum() + 1

    # set up link0 arguments
    out = []
    params, link0_list = _pop_link0_params(params)
    out.extend(link0_list)

    # begin route line
    # note: unlike in old calculator, each route keyword is put on its own
    # line.
    out.append(_format_method_basis(output_type, method, basis, fitting_basis))

    # If the calculator's parameter dictionary contains an isolist, we ignore
    # this - it is up to the user to attach this info as the atoms' masses
    # if they wish for it to be used:
    params.pop('isolist', None)

    # Any params left will belong in the route section of the file:
    out.extend(_format_route_params(params))

    if ioplist is not None:
        out.append('IOP(' + ', '.join(ioplist) + ')')

    # raw list of explicit keywords for backwards compatibility
    if extra is not None:
        out.append(extra)

    # Add 'force' iff the user requested forces, since Gaussian crashes when
    # 'force' is combined with certain other keywords such as opt and irc.
    if 'forces' in properties and 'force' not in params:
        out.append('force')

    # header, charge, and mult
    out += ['', 'Gaussian input prepared by ASE', '',
            '{:.0f} {:.0f}'.format(charge, mult)]

    # make dict of nuclear properties:
    nuclear_props = {'spin': spinlist, 'zeff': zefflist, 'qmom': qmomlist,
                     'nmagm': nmagmlist, 'znuc': znuclist,
                     'radnuclear': radnuclearlist}
    nuclear_props = {k: v for k, v in nuclear_props.items() if v is not None}

    # atomic positions and nuclear properties:
    molecule_spec = _get_molecule_spec(atoms, nuclear_props)
    for line in molecule_spec:
        out.append(line)

    out.extend(_format_basis_set(basis, basisfile, basis_set))
    out.pop()       # modified here so there is one linebreak less, otherwise the addsec is ignored
    out.extend(_format_addsec(addsec))

    out += ['', '']
    with open(fd, 'w') as f:
        f.write('\n'.join(out))
    f.close()




class GaussianDynamics(gaussian.GaussianDynamics):


    def run(self, **kwargs):
        calc_old = self.atoms.calc
        params_old = copy.deepcopy(self.calc.parameters)

        self.delete_keywords(kwargs)
        self.delete_keywords(self.calc.parameters)
        self.set_keywords(kwargs)

        self.calc.set(**kwargs)
        self.atoms.calc = self.calc

        try:
            self.atoms.get_potential_energy()
        except OSError:
            converged = False
        else:
            converged = True

        atoms = read_gaussian_out(self.calc.label)
        self.atoms.cell = atoms.cell
        self.atoms.positions = atoms.positions

        #self.calc.parameters = params_old
        #self.calc.reset()
        #if calc_old is not None:
         #   self.atoms.calc = calc_old

        return converged


class GaussianOptimizer(GaussianDynamics):
    '''
    Allowing ase to use Gaussian geometry optimizations
    
    '''
    keyword = 'opt'
    special_keywords = {
        'fmax': '{}',
        'steps': 'maxcycle={}',
    }


class GaussianIRC(GaussianDynamics):
    keyword = 'irc'
    special_keywords = {
        'direction': '{}',
        'steps': 'maxpoints={}',
    }


class Gaussian(gaussian.Gaussian):
    '''Modified Gaussian calculator using modified parsers'''

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        write_gaussian_in(self.label + '.com', atoms, properties=properties,
               **self.parameters)

    def read_results(self):
        output = read_gaussian_out(self.label)
        self.atoms=output
        self.calc = output.calc
        self.results = output.calc.results