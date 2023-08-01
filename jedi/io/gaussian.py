from ase.io.gaussian import *
from ase.vibrations.vibrations import VibrationsData
def read_gaussian_out(fd, index=-1):
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

def get_vibrations(label,atoms):
    _re_hessblock = re.compile(r'^\s*Force\s+constants\s+in\s+Cartesian\s+coordinates:\s*$')
    output = label+'.log'
    with open(output,'r') as fd:
        lines=fd.readlines()
    
    hess_line = 0
    NCarts = 3 * len(atoms)
    if len(atoms.constraints)>0:
        for l in atoms.constraints:
            if l.__class__.__name__=='FixAtoms':
                a=l.todict()
                clist=np.array(a['kwargs']['indices'])
                alist=np.delete(np.arange(0,len(atoms)),clist)
                
                NCarts = 3 * len(alist)
    hess=np.zeros((NCarts,NCarts))
    for num, line in enumerate(lines, 1):
        if 'Force constants in Cartesian coordinates:' in line:
            hess_line=num+1
    
    chunks=NCarts//5+1
    for i in range(chunks):
        for j in range(NCarts-i*5):

            rows=lines[round(hess_line+i*(NCarts+1)-sum(np.linspace(0,i-1,i)*5)+j)].split()
    
            rows=[float(k.replace('D', 'e')) for k in rows]
      
            hess[j+i*5][i*5:i*5+len(rows)-1]=rows[1::]
    hess=hess+hess.T-np.diag(np.diag(hess))
    hess*=(Hartree / Bohr**2) 
    return VibrationsData.from_2d(atoms,hess,indices)