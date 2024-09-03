import ase
import numpy as np
import os
from pathlib import Path
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from typing import Dict, Optional, Union, Literal, Sequence, List
from ase import Atoms
from ase import Atom
from ase import neighborlist
from ase.units import Hartree, Bohr, mol, kcal
from ase.data import covalent_radii, chemical_symbols
from strainjedi.colors import colors
from strainjedi.print_config import header, energy_comparison, atoms_listing
from strainjedi.quotes import quotes
from strainjedi import __version__
from strainjedi.jedi import Jedi


class JediAtoms(Jedi):

    E_atoms = None
    first_call = True
    def run(self, ase_units: bool = False,
            printout_save: bool = True,
            label: Union[str] = None,
            indices: Union[List[int]] = None,
            weighting: Optional[Literal['ric','poly']] = None,
            r_cut: Union[float] = None):
        """Runs the analysis. Calls all necessary functions to get the needed values.

        Args:
            ase_units: boolean
                True: eV for energies, Å for lengths
                False: kcal/mol for energies, Bohr for lengths
                Default: False
            printout_save: boolean
                True: saves printout as file
                False: doesn't save printout
                Default: True
            label: str
                label for saved printout file, E_atoms_{label}
                None: saves file as E_atoms or E_atoms_part for partial analysis
                Default: None
            indices: list
                list of indices of a substructure if desired
                Default: None
            weighting: boolean
                True: weighting function is used
                False: no weighting function
                Default: False
            r_cut: float
                used r_cut value for weighting function
                Default: None

        Returns:
            Indices, strain, energy in every atom
        """

        self.ase_units = ase_units
        # get necessary data
        self.indices=np.arange(0,len(self.atoms0))
        if indices:
            self.indices=indices
        if weighting == 'poly' and r_cut is None:
            raise TypeError("Please specify r_cut when weighting is set to 'poly'")
        delta_q = self.get_delta_q(weighting,r_cut)
        self.get_b_matrix(weighting,r_cut)
        B = self.B
        self.get_hessian()
        H_cart = self.H  # Hessian of optimized (ground state) structure

        if len(self.atoms0) != H_cart.shape[0]/3:
            raise ValueError('Hessian has not the fitting shape, possibly a partial hessian. Please try partial_analysis')
        try:
            all_E_geometries = self.get_energies()
        except:
            all_E_geometries = self.energies
        E_geometries = all_E_geometries[0]

        B_transp = np.transpose(B)

        # Calculate the pseudoinverse of the B-Matrix and its transposed
        B_plus = np.linalg.pinv(B, 0.0001)
        B_transp_plus = np.linalg.pinv(B_transp, 0.0001)

        # Calculate the P-Matrix (eq. 4 in Helgaker's paper)
        P = np.dot(B, B_plus)

        H_q = P.dot(B_transp_plus).dot(H_cart).dot(B_plus).dot(P)

        # Get the energy stored in every coordinate
        E_M = np.sum(0.5*(delta_q*H_q).T*delta_q,axis=1)
        self.E_atoms=np.sum(E_M.reshape(-1, len(self.atoms0)), axis=1)
        E_atoms_total = sum(self.E_atoms[self.indices])

        if ase_units==True:
            self.E_atoms*=Hartree
            E_atoms_total*=Hartree
            delta_q*=Bohr
        elif ase_units == False:
            self.E_atoms *= mol/kcal*Hartree
            E_atoms_total *= mol/kcal*Hartree

        proc_geom_atoms = (E_atoms_total / E_geometries - 1) * 100

        self.printout(E_geometries,E_atoms_total,proc_geom_atoms,r_cut,ase_units=self.ase_units)
        if not label:
            filename = 'E_atoms'
            if indices:
                filename += '_special'
        else:
            filename = f"E_atoms_{label}"
        if printout_save is True:
            self.printout(E_geometries,E_atoms_total,proc_geom_atoms,r_cut,ase_units=self.ase_units,save=True,file=filename)
        pass

    def get_bonds(self, mol):
        '''Gets list of bonds in mol

        '''
        mol = mol
        indices = np.arange(0,len(self.atoms0))

        cutoff = neighborlist.natural_cutoffs(mol, mult=self.covf)  ## cutoff for covalent bonds see Bakken et al.
        bl = np.vstack(neighborlist.neighbor_list('ij', a=mol, cutoff=cutoff)).T  # determine covalent bonds
        bl = bl[bl[:, 0] < bl[:, 1]]  # remove double mentioned
        bl, counts = np.unique(bl, return_counts=True, axis=0)
        if ~ np.all(counts == 1) and JediAtoms.first_call:
            print('\nunit cell too small hessian not calculated for self interaction\n'
                  'jedi analysis for a finite system consisting of the cell will be conducted')
            JediAtoms.first_call = False
        bl = np.atleast_2d(bl)
        if len(indices) != len(mol):
            bl = bl[np.all([np.in1d(bl[:, 0], indices), np.in1d(bl[:, 1], indices)], axis=0)]

        return bl

    def poly_weighting(self,delta_q,r_cut,indices=None):
        if indices is None:
            indices = np.arange(0,len(self.atoms0))
        bonds = self.get_bonds(self.atoms0)
        cutoff = neighborlist.natural_cutoffs(self.atoms0, mult=self.covf)
        r_g1 = max(neighborlist.neighbor_list('d', self.atomsF, cutoff=cutoff))
        if r_cut < r_g1:
            raise TypeError("r_cut needs to be bigger than the biggest distance from an atom to it's neighbor atom")
        for row in range(self.dF.shape[0]):
            for col in range(self.dF.shape[1]):
                r_gi = self.dF[row][col]
                r = (r_gi - r_g1) / (r_cut - r_g1)
                if row == col:
                    pass
                elif any((bond[0] == indices[row] and bond[1] == indices[col]) or (bond[0] == indices[col] and bond[1] == indices[row]) for bond in bonds):
                    delta_q[row][col] *= 1
                elif r > 1.0:
                    delta_q[row][col] *= 0
                elif r <= 0.5:
                    delta_q[row][col] *= (1 - 6 * r ** 2 + 6 * r ** 3)
                elif r >= 0.5:
                    delta_q[row][col] *= (2 - 6 * r + 6 * r ** 2 - 2 * r ** 3)

        return delta_q

    def ric_weighting(self,delta_q,indices=None):
        if indices is None:
            indices = np.arange(0,len(self.atoms0))
        rim_list = Jedi.get_rims(self,self.atomsF)
        j = Jedi(self.atoms0,self.atomsF,self.H)
        j.indices = indices
        delta_rics = j.get_delta_q()
        for row in range(self.dF.shape[0]):
            for col in range(self.dF.shape[1]):
                if row == col:
                    pass
                elif any((bl[0] == indices[row] and bl[1] == indices[col]) or (bl[0] == indices[col] and bl[1] == indices[row]) for bl in rim_list[0]):
                    delta_q[row][col] *= 1
                elif any((ba[0] == indices[row] and ba[2] == indices[col]) or (ba[0] == indices[col] and ba[2] == indices[row]) for idx, ba in enumerate(rim_list[2])):
                    idx =+ 1
                    if delta_rics[len(rim_list[0])+idx] <= 0.1 and delta_rics[len(rim_list[0])+idx] >= -0.1:
                        delta_q[row][col] *= 0
                    else:
                        delta_q[row][col] *= 0.5
                elif any((da[0] == indices[row] and da[3] == indices[col]) or (da[0] == indices[col] and da[3] == indices[row]) for idx, da in enumerate(rim_list[3])):
                    if delta_rics[len(rim_list[0])+len(rim_list[2])+idx] <= 0.1 and delta_rics[len(rim_list[0])+len(rim_list[2])+idx] >= -0.1:
                        delta_q[row][col] *= 0
                    else:
                        delta_q[row][col] *= 0.5
                else:
                    delta_q[row][col] *= 0

        return delta_q

    def get_delta_q(self,weighting,r_cut,indices=None):
        mic = False
        if self.atomsF.get_pbc().any() == True:
            mic = True
        self.d0 = self.atoms0.get_all_distances(mic=mic)
        self.dF = self.atomsF.get_all_distances(mic=mic)
        if indices:
            self.d0 = self.d0[np.ix_(indices, indices)]
            self.dF = self.dF[np.ix_(indices, indices)]
        delta_q = self.dF - self.d0
        if weighting == 'poly':
            delta_q = self.poly_weighting(delta_q,r_cut,indices).flatten() / Bohr
        elif weighting == 'ric':
            delta_q = self.ric_weighting(delta_q,indices).flatten() / Bohr
        else:
            delta_q = delta_q.flatten() / Bohr

        return delta_q

    def get_b_matrix(self,weighting,r_cut,indices=None):
        if indices is None:
            indices = np.arange(0,len(self.atomsF))
        mic = False
        if self.atomsF.get_pbc().any() == True:
            mic = True
        B = np.empty((len(indices)**2, 3 * len(indices)))
        pos0 = self.atoms0.positions.copy()
        for idx, i in enumerate(indices):
            for j in range(3):
                a = self.atoms0.copy()
                pos = pos0.copy()
                pos[i][j] -= 0.005
                a.positions = pos
                d_minus = a.get_all_distances(mic=mic)
                pos[i][j] += 2 * 0.005
                a.positions = pos
                d_plus = a.get_all_distances(mic=mic)
                delta_q = d_minus - d_plus
                if len(indices) < len(self.atoms0):
                    delta_q = delta_q[np.ix_(indices, indices)]
                if weighting == 'poly':
                    delta_q = self.poly_weighting(delta_q, r_cut,indices).flatten()
                elif weighting == 'ric':
                    delta_q = self.ric_weighting(delta_q, indices).flatten()
                else:
                    delta_q = delta_q.flatten()
                derivatives = delta_q / 0.01
                derivatives = np.reshape(derivatives, (len(indices)**2))
                B[0:len(indices)**2, idx * 3 + j] = derivatives
        self.B = B

        return B

    def printout(self,
                 E_geometries: float,
                 E_atoms_total: float,
                 proc_geom_atoms: float,
                 weighting: Optional[Literal['ric','poly']] = None,
                 r_cut: float = None,
                 ase_units: bool = False,
                 save: bool = False,
                 file: str = 'E_atoms'):
        '''
        Printout of analysis of stored strain energy in the bonds.
        '''
        output = []
        # Header
        output.append("\n")
        output.append('{:^{header}}'.format("************************************************", **header))
        output.append('{:^{header}}'.format("*                 JEDI ANALYSIS                *", **header))
        output.append('{:^{header}}'.format("*       Judgement of Energy DIstribution       *", **header))
        output.append('{:^{header}}'.format("************************************************", **header))
        output.append('{:^{header}}'.format(f"version {__version__}\n", **header))

        # Comparison of total energies
        if not ase_units:
            output.append('{0:>{column1}}''{1:^{column2}}''{2:^{column3}}'
                          .format(" ", "Strain Energy (kcal/mol)", "Deviation (%)", **energy_comparison))
        elif ase_units:
            output.append('{0:>{column1}}''{1:^{column2}}''{2:^{column3}}'
                          .format(" ", "Strain Energy (eV)", "Deviation (%)", **energy_comparison))

        output.append('{0:<{column1}}' '{1:^{column2}.4f}' '{2:^{column3}}'
                      .format("Ab Initio", E_geometries, "-", **energy_comparison))
        # TODO save proc_e_rims as std decimal number and have print command use .2%
        output.append('{0:<{column1}}''{1:^{column2}.4f}''{2:^{column3}.2f}'
                      .format("JEDI_ATOMS", E_atoms_total, proc_geom_atoms, **energy_comparison))
        if weighting == 'poly':
            output.append('{0:<{column1}}' '{1:^{column2}}' '{2:^{column3}}'
                          .format(f"(r_cut = {r_cut})","","", **energy_comparison))

        # strain in the atoms
        if not ase_units:
            output.append(
                '{0:^{column1}}''{1:^{column2}}''{2:^{column3}}''{3:^{column4}}'
                .format("Atom No.", "Element", "Percentage", "Energy (kcal/mol)", **atoms_listing))
        elif ase_units:
            output.append(
                '{0:^{column1}}''{1:^{column2}}''{2:^{column3}}''{3:^{column4}}'
                .format("Atom No.", "Element", "Percentage", "Energy (eV)", **atoms_listing))


        for i, k in enumerate(self.E_atoms[self.indices]):
            output.append(
                '{0:^{column1}}''{1:^{column2}}''{2:^{column3}.2f}''{3:^{column4}.2f}'
                .format(self.indices[i],
                        self.atoms0.symbols[self.indices[i]],
                        k/E_atoms_total*100,
                        k,
                        **atoms_listing))

        if save is False:
            print("\n".join(output))
            print("\n"+quotes())
        else:
            with open(file, 'w') as f:
                f.writelines("\n".join(output))

    def partial_analysis(self, indices: Union[List[int]] = None,
                         ase_units: bool = False,
                         printout_save: bool = True,
                         label: Union[str] = None,
                         weighting: bool = True,
                         r_cut: Union[float] = None):

        """Runs the analysis. Calls all necessary functions to get the needed values.

        Args:
            indices:
                list of indices of a substructure
            ase_units: boolean
                True: eV for energies, Å for lengths
                False: kcal/mol for energies, Bohr for lengths
                Default: False
            printout_save: boolean
                True: saves printout as file
                False: doesn't save printout
                Default: True
            label: str
                label for saved printout file, E_atoms_{label}
                None: saves file as E_atoms or E_atoms_part for partial analysis
                Default: None
            indices: list
                list of indices of a substructure if desired
                Default: None
            weighting: boolean
                True: weighting function is used
                False: no weighting function
                Default: False
            r_cut: float
                used r_cut value for weighting function
                Default: None
        Returns:
            Indices, strain, energy in every atom
        """
        self.ase_units = ase_units
        # get necessary data
        self.indices=indices

        if weighting is True and r_cut is None:
            raise TypeError("Please specify r_cut when weighting is set to True")
        delta_q = self.get_delta_q(weighting, r_cut, indices)
        self.get_hessian()
        H_cart = self.H         #Hessian of optimized (ground state) structure
        self.get_b_matrix(weighting, r_cut, indices=indices)
        B = self.B

        if len(indices) != H_cart.shape[0]/3:
            raise ValueError('Hessian has not the fitting shape')

        try:
            all_E_geometries = self.get_energies()
        except:
            all_E_geometries = self.energies
        E_geometries = all_E_geometries[0]

        B_transp = np.transpose(B)

        # Calculate the pseudoinverse of the B-Matrix and its transposed
        B_plus = np.linalg.pinv(B, 0.0001)
        B_transp_plus = np.linalg.pinv(B_transp, 0.0001)

        # Calculate the P-Matrix (eq. 4 in Helgaker's paper)
        P = np.dot(B, B_plus)

        H_q = P.dot(B_transp_plus).dot(H_cart).dot(B_plus).dot(P)

        # Get the energy stored in every coordinate
        E_M = np.sum(0.5 * (delta_q * H_q).T * delta_q, axis=1)
        self.E_atoms = np.sum(E_M.reshape(-1, len(indices)), axis=1)
        E_nan = np.full((len(self.atoms0)), np.nan)
        E_nan[indices] = self.E_atoms
        self.E_atoms = E_nan
        E_atoms_total = np.nansum(self.E_atoms)

        if ase_units==True:
            self.E_atoms*=Hartree
            E_atoms_total*=Hartree
            delta_q*=Bohr
        elif ase_units == False:
            self.E_atoms*=mol/kcal*Hartree
            E_atoms_total*=mol/kcal*Hartree

        proc_geom_atoms = (E_atoms_total / E_geometries - 1) * 100

        self.printout(E_geometries, E_atoms_total, proc_geom_atoms, r_cut, ase_units=self.ase_units)
        if not label:
            filename = 'E_atoms_partial'
        else:
            filename = f"E_atoms_{label}"
        if printout_save is True:
            self.printout(E_geometries, E_atoms_total, proc_geom_atoms, r_cut, ase_units=self.ase_units, save=True,
                          file=filename)

    def vmd_gen(self,
                des_colors: Optional[Dict] = None,
                box: bool = False,
                bonds_out_of_box: bool = False,
                man_strain: Optional[float] = None,
                colorbar: bool = True,
                label: Union[Path, str] = 'vmd',
                incl_coloring: Optional[str] = None):
        """Generates all scripts and files to save the values for the color coding

        Args:
            des_colors: (dict)
                key: order number, value: [R,G,B]
            box: boolean
                True: draw box
                False: ignore box
            man_strain: float
                reference value for the strain energy used in the color scale
                default: 'None'
            colorbar: boolean
                draw colorbar or not
            label: string
                name of folder for the created files
            incl_coloring: str
                2 inclusive coloring options, otherwise green to red gradient
                "cyan": cyan to red gradient
                "magma": matplotlib magma gradient
                default: 'None'
        """
        if isinstance(label, str):
            destination_dir = Path(label)
        elif isinstance(label, Path):
            destination_dir = label
        else:
            raise TypeError("Please specify the directory (label) to write vmd scripts to as Path or string")
        destination_dir.mkdir(parents=True, exist_ok=True)
        #########################
        #       Basic stuff     #
        #########################

        if self.ase_units == False:
            unit = "kcal/mol"
        elif self.ase_units == True:
            unit = "eV"
        pbc_flag = False
        if self.atomsF.get_pbc().any() == True:
            pbc_flag = True
        self.atomsF.write(destination_dir / 'xF.xyz')

        E_atoms = self.E_atoms

        # Write some basic stuff to the tcl scripts

        output = []
        output.append(f'\n# Load a molecule\nmol new {{{destination_dir.resolve() / "xF.xyz"}}}\n\n')
        output.append('# Change bond radii and various resolution parameters\nmol representation cpk 0.8 0.0 30 '
                      '5\nmol representation bonds 0.2 30\n\n')

        output.append('# Change the color of the graphical representation 0 to white\ncolor change rgb 0 1.00 1.00 '
                      '1.00\n')
        output.append('# The background should be white ("blue" has the colorID 0, which we have changed to '
                      'white)\ncolor Display Background blue\n\n')
        output.append('# Define the other colorIDs\n')

        # Define colorcodes for various atomtypes

        # from .colors import colors
        if des_colors is not None:
            for i in des_colors:
                colors[i] = des_colors[i]  # desired colors overwrite the standard ones

        symbols = np.unique(self.atomsF.get_chemical_symbols())
        symbols = symbols[symbols != 'H']  # get all symbols except H, H is white

        N_colors_atoms = len(symbols)
        N_colors = 32 - N_colors_atoms - 1  # vmd only supports 32 colors for modcolor

        # Generate the color-code and write it to the tcl scripts

        colorbar_colors = []

        # get green to red gradient
        if incl_coloring is None:
            for i in range(N_colors):
                R_value = float(i) / (N_colors / 2)
                if R_value > 1:
                    R_value = 1
                if N_colors % 2 == 0:
                    G_value = 2 - float(i + 1) / (N_colors / 2)
                if N_colors % 2 != 0:
                    G_value = 2 - float(i) / (N_colors / 2)
                if G_value > 1:
                    G_value = 1

                B_value = 0

                output.append(
                    '%1s%5i%10.6f%10.6f%10.6f%1s' % ("color change rgb", i + 1, R_value, G_value, B_value, "\n"))
                colorbar_colors.append((R_value, G_value, B_value))

        elif incl_coloring == "cyan":
            # get cyan to red gradient
            for i in range(N_colors):
                R_value = float(i) / (N_colors / 2)
                if R_value > 1:
                    R_value = 1
                B_value = 2 - float(i + 1) / (N_colors / 2)
                if B_value > 1:
                    B_value = 1
                if i <= (N_colors / 2):
                    G_value = ((N_colors / 2) - i) / (N_colors / 2)
                if i > (N_colors / 2):
                    G_value = 0

                output.append(
                    '%1s%5i%10.6f%10.6f%10.6f%1s' % ("color change rgb", i + 1, R_value, G_value, B_value, "\n"))
                colorbar_colors.append((R_value, G_value, B_value))

        elif incl_coloring == "magma":
            # get magma gradient from matplotlib
            gradient = np.linspace(0, 1, N_colors)
            cmap = cm.get_cmap('magma').reversed()
            cut_off_blue = 0.175
            cut_off_beige = 0.15
            adjusted_gradient = gradient * (1 - cut_off_blue - cut_off_beige) + cut_off_beige
            colors_rgb = cmap(adjusted_gradient)
            R_values = colors_rgb[:, 0]
            G_values = colors_rgb[:, 1]
            B_values = colors_rgb[:, 2]
            for i in range(N_colors):
                output.append('%1s%5i%10.6f%10.6f%10.6f%1s' % (
                    "color change rgb", i + 1, R_values[i], G_values[i], B_values[i], "\n"))
                colorbar_colors.append((R_values[i], G_values[i], B_values[i]))

        # add color code for axes and box
        output.append(
            '%1s%5i%10.6f%10.6f%10.6f%1s' % ("color change rgb", 32, float(0), float(0), float(0), "\n"))  # black
        output.append(
            '%1s%5i%10.6f%10.6f%10.6f%1s' % ("color change rgb", 1039, float(1), float(0), float(0), "\n"))  # red
        output.append(
            '%1s%5i%10.6f%10.6f%10.6f%1s' % ("color change rgb", 1038, float(0), float(1), float(0), "\n"))  # green
        output.append(
            '%1s%5i%10.6f%10.6f%10.6f%1s' % ("color change rgb", 1037, float(0), float(0), float(1), "\n"))  # blue
        output.append('%1s%5i%10.6f%10.6f%10.6f%1s' % (
            "color change rgb", 1036, float(0.25), float(0.75), float(0.75), "\n"))  # cyan
        output.append('''color Axes X 1039
color Axes Y 1038
color Axes Z 1037
color Axes Origin 1036
color Axes Labels 32
''')
        # define color of atoms with the color code above
        for j in range(N_colors_atoms):
            output.append('\n\nmol representation cpk 0.7 0.0 30 5')
            output.append('\nmol addrep top')
            output.append('\n%s%i%s' % ("mol modstyle ", j + 1, " top cpk"))
            output.append('\n%s%i%s%i%s' % ("mol modcolor ", j + 1, " top {colorid ", N_colors + j + 1, "}"))
            output.append('\n%s%i%s%s%s' % ("mol modselect ", j + 1, " top {name ", symbols[j], "}"))

        #########################
        #	Binning		#
        #########################

        # Welcome
        print("\n\nCreating tcl scripts for generating color-coded structures in VMD...")

        # Create an array that stores the atom as the first entry The energy will be added as the second entry.
        E_array = np.full((len(self.E_atoms)), np.nan)

        # Create an array that stores only the energies in the coordinate of interest and print some information
        # Get rid of ridiculously small values and treat diatomic molecules explicitly
        # (in order to create a unified picture, we have to create all these arrays in any case)

        if E_atoms.max() <= 0.001:
            E_atoms = np.zeros(len(self.indices))

        if len(E_atoms) > len(self.indices):  # for partial_analysis or run with indices
            E_array[self.indices] = E_atoms[self.indices]
        else:
            E_array = E_atoms
        E_array = np.vstack((np.arange(len(self.atoms0)), E_array))
        print("\nProcessing atoms...")

        # get bonds that reach out of the unit cell
        if pbc_flag == True and bonds_out_of_box == True:
            E_array_pbc = np.empty((2, 0))

            from ase.data.vdw import vdw_radii  # for long range bonds
            cutoff = [vdw_radii[atom.number] * self.vdwf for atom in self.atomsF]
            ex_bl = np.vstack(neighborlist.neighbor_list('ij', a=self.atomsF, cutoff=cutoff)).T
            ex_bl = np.hstack((ex_bl, neighborlist.neighbor_list('S', a=self.atomsF, cutoff=cutoff)))
            ex_bl = np.hstack((ex_bl, neighborlist.neighbor_list('D', a=self.atomsF, cutoff=cutoff)))
            atoms_ex_cell = ex_bl[(ex_bl[:, 2] != 0) | (ex_bl[:, 3] != 0) | (
                    ex_bl[:, 4] != 0)]  # determines which nearest neighbors are outside the unit cell
            mol = self.atomsF.copy()  # a extended cell is needed for vmd since it does not show intercellular bonds
            mol.wrap()  # wrap molecule important for atoms close to the boundaries
            bondscheck = self.get_bonds(self.atomsF)

            for i in range(len(atoms_ex_cell)):
                pos_ex_atom = mol.get_positions()[int(atoms_ex_cell[i, 0])] + atoms_ex_cell[i,
                                                                              5:8]  # get positions of cell external atoms by adding the vector
                # if pos_ex_atom in mol.positions:
                original_rim = [int(atoms_ex_cell[i, 0]),
                                int(atoms_ex_cell[
                                        i, 1])]  # get the indices of the corresponding atoms inside the cell
                original_rim.sort()  # needs to be sorted because rim list only covers one direction
                if len(np.where(np.all(mol.positions == pos_ex_atom, axis=1))[0]) > 0:
                    ex_ind = np.where(np.all(mol.positions == pos_ex_atom, axis=1))[0][0]
                else:
                    ex_ind = len(mol)
                    if len(np.where(np.all(original_rim == bondscheck, axis=1))[0]) > 0:
                        mol.append(Atom(symbol=mol.symbols[int(atoms_ex_cell[i, 1])],
                                        position=pos_ex_atom))  # append to the virtual atoms object

                if len(np.where(np.all(original_rim == bondscheck, axis=1))[0]) > 0:
                    E_array_value = E_array[1][int(atoms_ex_cell[i, 1])]
                    E_array_pbc = np.append(E_array_pbc, [[ex_ind], [E_array_value]],
                                            axis=1)  # add to bond list with auxillary index

            mol.write(destination_dir / 'xF.xyz')  # save the modified structure with auxilliary atoms for vmd
            E_array = np.hstack((E_array, E_array_pbc))

        # Store the maximum energy in a variable for later call
        max_energy = float(np.nanmax(E_array, axis=1)[1])  # maximum energy in one atom

        # Generate the binning windows by splitting E_array into N_colors equal windows

        if man_strain == None:

            binning_windows = np.linspace(0, np.nanmax(E_array, axis=1)[1], num=N_colors)
        else:
            binning_windows = np.linspace(0, float(man_strain), num=N_colors)

        if box:
            output.append("\n\n# Adding a pbc box")
            output.append('\npbc set {%f %f %f %f %f %f}' % (
                self.atomsF.cell.cellpar()[0], self.atomsF.cell.cellpar()[1], self.atomsF.cell.cellpar()[2],
                self.atomsF.cell.cellpar()[3], self.atomsF.cell.cellpar()[4], self.atomsF.cell.cellpar()[5]))
            output.append("\npbc box -color 32")
        output.append("\n\n# Adding a representation with the appropriate colorID for each atom")
        # Calculate which binning_windows value is closest to the bond-percentage and do the output

        for i, b in zip(E_array[0], E_array[1]):
            if np.isnan(b):
                colorID = 32  # black
            else:
                colorID = np.abs(binning_windows - b).argmin() + 1

            output.append('\n\nmol representation cpk 0.7 0.0 30 5')
            output.append('\nmol addrep top')
            output.append('\n%s%i%s' % ("mol modstyle ", N_colors_atoms + i + 1, " top cpk"))
            output.append(
                '\n%s%i%s%i%s' % ("mol modcolor ", N_colors_atoms + i + 1, " top {colorid ", colorID, "}"))
            output.append(
                '\n%s%i%s%s%s' % ("mol modselect ", N_colors_atoms + i + 1, " top {index ", int(i), "}\n"))
        f = open(destination_dir / 'atoms.vmd', 'w')
        f.writelines(output)
        f.close()

        # colorbar
        if colorbar == True:
            min = 0.000

            if man_strain == None:
                max = np.nanmax(E_array, axis=1)[1]
            else:
                max = man_strain

            # highresolution colorbar with matplotlib
            import matplotlib.pyplot as plt
            from matplotlib.colorbar import ColorbarBase
            from matplotlib.colors import LinearSegmentedColormap, Normalize
            plt.rc('font', size=20)
            fig = plt.figure()
            ax = fig.add_axes([0.05, 0.08, 0.1, 0.9])
            cmap_name = 'my_list'
            cmap = LinearSegmentedColormap.from_list(cmap_name, colorbar_colors, N=N_colors)
            cb = ColorbarBase(ax, orientation='vertical',
                              cmap=cmap,
                              norm=Normalize(min, round(max, 3)),
                              label=unit,
                              ticks=np.round(np.linspace(min, max, 8), decimals=3))

            fig.savefig(destination_dir / 'atomscolorbar.pdf', bbox_inches='tight')

        if man_strain == None:
            print(f"Maximum energy in  atom {int(np.nanargmax(E_atoms))}: {float(max_energy):.3f} {unit}.")

    def pov_gen(self, colorbar: bool = True,
                box: bool = False,
                bonds_out_of_box: bool = False,
                man_strain: Optional[float] = None,
                label: Union[Path, str] = 'pov',
                incl_coloring: Optional[Literal['cyan', 'magma']] = None,
                view_dir: Optional[Union[Literal['x', 'y', 'z'], Atoms]] = None,
                zoom: float = 1.,
                metal: Optional[list] = None,
                tex: str = 'vmd',
                radii: float = 1.,
                scale_radii: Optional[float] = None,
                bond_color: tuple = (0.75,0.75,0.75),
                light: Optional[Union[Sequence[float], np.ndarray]] = None,
                background: str = 'White',
                bondradius: float = .1,
                pixelwidth: int = 2000,
                aspectratio: Optional[float] = None,
                run_pov: bool = True):
        """Generates POV object for atoms object

                Args:
                    colorbar: boolean
                        True: save colorbar
                        default: False
                    box: boolean
                        True: draw box
                        False: ignore box
                        default: False
                    bonds_out_of_box: boolean
                        True: shows atoms for bonds that reach out of the box
                        default: 'False'
                    man_strain: float
                        reference value for the strain energy used in the color scale
                        default: 'None'
                    label: string or Path
                        name of folder for the created files
                        default: 'pov'
                    incl_coloring: str
                        2 inclusive coloring options, otherwise green to red gradient
                        'cyan': cyan to red gradient
                        'magma': costumed matplotlib magma gradient
                        default: 'None'
                    view_dir: str
                        camera view
                        'x': from the x-axis at the y,z-plane
                        'y': from the y-axis at the z,x-plane
                        'z': from the z-axis at the x,y-plane
                        None: for all three view directions
                        Atoms: rotated Atoms object for customed view direction
                        default: None
                    zoom: float
                        change the zoom
                        default: 1.0
                    metal: list
                        a list of atom indices that should be treated as metals without bonds, a metal texture and a
                        bigger scaled radius
                        default: None
                    tex: str
                        a texture to use for the atoms
                        default = 'vmd'
                    radii: float or list
                        atomic radii. if a single value is given, it is interpreted as a multiplier for the covalent radii
                        in ase.data. if a list of len(atoms) is given, it is interpreted as individual atomic radii
                        default: 1.0
                    scale_radii: float
                        float or list with floats that scale the atomic radii
                        default: 0.5
                    bond_color: tuple
                        color for the bonds as a (r,g,b) tuple
                        None: (0.75,0.75,0.75) is used
                        default: None
                    light: tuple
                        position of the light source as a (x,y,z) tuple
                        None: light source has the same direction as the camera
                        default: None
                    background: str
                        background color
                        default: 'White'
                    bondradius: float
                        radii to use in drawing bonds
                        default: 0.1
                    pixelwidth: int
                        width in pixels of the final image. Note that the height is set by the aspect ratio
                        (controlled by carmera_right_up).
                        default: 2000
                    aspectratio: float
                        controls the aspect ratio
                        None: aspect ratio is calculated by carmera_right_up
                        default: None
                    run_pov: boolean
                        True: png is created directly
                        False: pov is saved and needs to be rendered by povray
                        default: True
                """

        from strainjedi.io.iopov import POV
        from matplotlib.colors import LinearSegmentedColormap
        import builtins

        if isinstance(label, str):
            destination_dir = Path(label)
        elif isinstance(label, Path):
            destination_dir = label
        else:
            raise TypeError("Please specify the directory (label) to write pov scripts to as Path or string")
        destination_dir.mkdir(parents=True, exist_ok=True)

        if type(view_dir) is ase.Atoms:
            atoms_f = view_dir
        else:
            atoms_f = self.atomsF.copy()
        pbc_flag = False
        if self.atomsF.get_pbc().any() == True:
            pbc_flag = True

        # Welcome
        print("\n\nCreating pov script for generating color-coded structures in POV-Ray...")

        # find bonds in molecule
        bonds = self.get_bonds(self.atomsF)

        # get color gradients
        if incl_coloring is None:
            # get green red gradient
            grad_colors = [(0, 1, 0), (1, 1, 0), (1, 0, 0)]
            color_positions = [0, 0.5, 1]
            cmap = LinearSegmentedColormap.from_list('custom_cmap', list(zip(color_positions, grad_colors)))
        elif incl_coloring == 'cyan':
            # get cyan red gradient
            grad_colors = [(0, 1, 1), (1, 0, 1), (1, 0, 0)]
            color_positions = [0, 0.5, 1]
            cmap = LinearSegmentedColormap.from_list('custom_cmap', list(zip(color_positions, grad_colors)))
        elif incl_coloring == 'magma':
            # get magma gradient
            magma = cm.get_cmap('magma')
            magma_r = magma(np.linspace(1, 0, 256))
            cut_off_beige = int(256 * 0.15)
            cut_off_blue = -int(256 * 0.175)
            magma_r_cut = magma_r[cut_off_beige:cut_off_blue]
            cmap = LinearSegmentedColormap.from_list('custom_cmap', magma_r_cut)

        E_atoms = self.E_atoms
        # Get rid of ridiculously small values
        if E_atoms.max() <= 0.001:
            E_atoms = np.zeros(len(self.indices))

        # get an E_array with only the information from coordinates of interest
        E_array = np.full((len(self.atoms0)), np.nan)
        if len(E_atoms) > len(self.indices):  # for partial_analysis or run with indices
            E_array[self.indices] = E_atoms[self.indices]
        else:
            E_array = E_atoms
        E_array = np.vstack((np.arange(len(self.atoms0)), E_array))

        pbc_bonds = []
        # delete bonds that reach out of the unit cell for bonds_out_of_box==False
        if pbc_flag == True:
            pbc_bonds_mask = []
            dF = self.atomsF.get_all_distances()
            dF_mic = self.atomsF.get_all_distances(mic=True)
            for i in bonds:
                if round(dF[i[0]][i[1]],5) != round(dF_mic[i[0]][i[1]],5):
                    pbc_bonds_mask.append(False)
                else:
                    pbc_bonds_mask.append(True)
            pbc_bonds = bonds[~np.array(pbc_bonds_mask)]
            bonds = np.delete(bonds,np.where(~np.array(pbc_bonds_mask))[0],axis=0)

        # get atoms for bonds that reach out of the unit cell and their energies
        if pbc_flag == True and bonds_out_of_box == True:
            E_array_pbc = np.empty((2, 0))
            from ase.data.vdw import vdw_radii  # for long range bonds
            cutoff = [vdw_radii[atom.number] * self.vdwf for atom in self.atomsF]
            ex_bl = np.vstack(neighborlist.neighbor_list('ij', a=self.atomsF, cutoff=cutoff)).T
            ex_bl = np.hstack((ex_bl, neighborlist.neighbor_list('S', a=self.atomsF, cutoff=cutoff)))
            ex_bl = np.hstack((ex_bl, neighborlist.neighbor_list('D', a=self.atomsF, cutoff=cutoff)))
            atoms_ex_cell = ex_bl[(ex_bl[:, 2] != 0) | (ex_bl[:, 3] != 0) | (
                    ex_bl[:, 4] != 0)]  # determines which nearest neighbors are outside the unit cell
            atoms_f.wrap()  # wrap molecule important for atoms close to the boundaries
            bondscheck = self.get_bonds(self.atomsF)

            for i in range(len(atoms_ex_cell)):
                pos_ex_atom = atoms_f.get_positions()[int(atoms_ex_cell[i, 0])] + atoms_ex_cell[i,
                                                                                  5:8]  # get positions of cell external atoms by adding the vector
                original_rim = [int(atoms_ex_cell[i, 0]),
                                int(atoms_ex_cell[
                                        i, 1])]  # get the indices of the corresponding atoms inside the cell
                original_rim.sort()  # needs to be sorted because rim list only covers one direction
                if len(np.where(np.all(atoms_f.positions == pos_ex_atom, axis=1))[0]) > 0:
                    ex_ind = np.where(np.all(atoms_f.positions == pos_ex_atom, axis=1))[0][0]
                else:
                    ex_ind = len(atoms_f)
                    if len(np.where(np.all(original_rim == bondscheck, axis=1))[0]) > 0:
                        atoms_f.append(Atom(symbol=atoms_f.symbols[int(atoms_ex_cell[i, 1])],
                                            position=pos_ex_atom))  # append to the virtual atoms object
                        bonds = np.delete(bonds, np.where(np.all(original_rim == bonds, axis=1))[0], axis=0)
                        bonds = np.vstack((bonds, [[int(atoms_ex_cell[i, 0]), ex_ind]]))
                if len(np.where(np.all(original_rim == bondscheck, axis=1))[0]) > 0:
                    E_array_value = E_array[1][int(atoms_ex_cell[i, 1])]
                    E_array_pbc = np.append(E_array_pbc, [[ex_ind], [E_array_value]],
                                            axis=1)  # add to bond list with auxillary index

            E_array = np.hstack((E_array, E_array_pbc))

        # Store the maximum energy in a variable for later call
        max_energy = float(np.nanmax(E_array, axis=1)[1])  # maximum energy in one atom

        # get atom color for specific energy value from color gradient
        atom_colors = np.array([])
        for i, b in zip(E_array[0], E_array[1]):
            if np.isnan(b):
                atom_color = (0.000, 0.000, 0.000)  # black
            else:
                if man_strain is None:
                    normalized_energy = b / max_energy
                else:
                    normalized_energy = b / float(man_strain)
                color = cmap(normalized_energy)
                atom_color = (color[0], color[1], color[2])
            atom_colors = np.append(atom_colors, atom_color)
            atom_colors = atom_colors.reshape(-1, 3)

        # radii to distinguish different elements
        atomic_numbers = atoms_f.get_atomic_numbers()
        z = np.array(list(set(atomic_numbers)))
        atom_radii = None
        legend = False
        if len(z) > 1:
            legend = True
            atom_radii = {i: covalent_radii[i] for i in z}
            atom_radii = dict(sorted(atom_radii.items(), key=lambda item: item[1]))
            radii_values = np.linspace(0.3, 0.8, len(z))
            atom_radii = {k: v for k, v in zip(atom_radii.keys(), radii_values)}
            radii = np.array([atom_radii[num] for num in atomic_numbers])

        # generating pov object with specified view direction, write .pov file and run it
        if metal:
            tex = [tex] * len(atoms_f)
            scales = np.array([0.5] * len(atoms_f))
            for idx in metal:
                tex[idx] = 'chrome'
                scales[idx] = 1.0

        cell = None
        if type(view_dir) is ase.Atoms:
            atoms_rotated = view_dir
            positions = atoms_rotated.get_positions()
            center = np.mean(positions, axis=0)
            if box and pbc_flag is True:
                cell = atoms_rotated.cell
                center = 0.5 * cell[0] + 0.5 * cell[1] + 0.5 * cell[2]
            camwidth = [(-(builtins.max(atoms_rotated.positions[:, 0]) - center[0]) - 1, 0., 0.),
                        (0., builtins.max(atoms_rotated.positions[:, 1]) - center[1] + 1, 0.)]
            if cell is not None:
                camwidth = [
                    (-0.5 * abs(center[0] - builtins.min(atoms_rotated.positions[:, 0]) + 1.0), 0., 0.),
                    (0., 0.5 * abs(center[1] - builtins.max(atoms_rotated.positions[:, 1])) + 1.0, 0.)]
            location = center.copy()
            location[2] += 60. * zoom
            direction = center.copy()
            direction[2] += 10.
            if light is None:
                light = center.copy()
                light[2] += 70.
            pov = POV(atoms_rotated,
                      tex=tex,
                      radii=radii,
                      scale_radii=scale_radii,
                      bond_colors=bond_color,
                      atom_colors=atom_colors,
                      cameralocation=location,
                      look_at=center,
                      camera_right_up=camwidth,
                      cameradirection=direction,
                      area_light=[light, 'White', 1.7, 1.7, 3, 3],
                      background=background,
                      bondatoms=bonds,
                      pbc_bondatoms=pbc_bonds,
                      metal=metal,
                      bondradius=bondradius,
                      pixelwidth=pixelwidth,
                      aspectratio=aspectratio,
                      cell=cell,
                      legend=legend,
                      legend_atoms=atom_radii,
                      legend_color=bond_color
                      )
            if run_pov is True:
                pov.write(f'{label}.png', label)
            else:
                pov.write(f'{label}.pov', label)
        else:
            view_list = ['x', 'y', 'z']
            for view in view_list:
                if view_dir == view or view_dir is None:
                    if view == 'x':
                        atoms_rotated = atoms_f.copy()
                        atoms_rotated.rotate(90, '-z', rotate_cell=True)
                        atoms_rotated.rotate(90, '-x', rotate_cell=True)
                        positions = atoms_rotated.get_positions()
                        center = np.mean(positions, axis=0)
                        if box and pbc_flag is True:
                            cell = atoms_rotated.cell
                            center = 0.5 * cell[0] + 0.5 * cell[1] + 0.5 * cell[2]
                    elif view == 'y':
                        atoms_rotated = atoms_f.copy()
                        atoms_rotated.rotate(90, 'z', rotate_cell=True)
                        atoms_rotated.rotate(90, 'y', rotate_cell=True)
                        positions = atoms_rotated.get_positions()
                        center = np.mean(positions, axis=0)
                        if box and pbc_flag is True:
                            cell = atoms_rotated.cell
                            center = 0.5 * cell[0] + 0.5 * cell[1] + 0.5 * cell[2]
                    elif view == 'z':
                        atoms_rotated = atoms_f.copy()
                        positions = atoms_rotated.get_positions()
                        center = np.mean(positions, axis=0)
                        if box and pbc_flag is True:
                            cell = atoms_rotated.cell
                            center = 0.5 * cell[0] + 0.5 * cell[1] + 0.5 * cell[2]
                    camwidth = [(-(builtins.max(atoms_rotated.positions[:, 0]) - center[0]) - 1, 0., 0.),
                                (0., builtins.max(atoms_rotated.positions[:, 1]) - center[1] + 1, 0.)]
                    if cell is not None:
                        camwidth = [
                            (-0.5 * abs(center[0] - builtins.min(atoms_rotated.positions[:, 0]) + 1.0), 0., 0.),
                            (0., 0.5 * abs(center[1] - builtins.max(atoms_rotated.positions[:, 1])) + 1.0, 0.)]
                    location = center.copy()
                    location[2] += 60.
                    direction = center.copy()
                    direction[2] += 10. * zoom
                    if light is None:
                        light = center.copy()
                        light[2] += 70.
                    pov = POV(atoms_rotated,
                              tex=tex,
                              radii=radii,
                              scale_radii=scale_radii,
                              bond_colors=bond_color,
                              atom_colors=atom_colors,
                              cameralocation=location,
                              look_at=center,
                              camera_right_up=camwidth,
                              cameradirection=direction,
                              area_light=[light, 'White', 1.7, 1.7, 3, 3],
                              background=background,
                              bondatoms=bonds,
                              pbc_bondatoms=pbc_bonds,
                              metal=metal,
                              bondradius=bondradius,
                              pixelwidth=pixelwidth,
                              aspectratio=aspectratio,
                              cell=cell,
                              legend=legend,
                              legend_atoms=atom_radii,
                              legend_color=bond_color
                              )
                    if run_pov is True:
                        pov.write(f'{label}.png', label)
                    else:
                        pov.write(f'{label}.pov', label)

        if self.ase_units == False:
            unit = "kcal/mol"
        elif self.ase_units == True:
            unit = "eV"

        # colorbar
        if colorbar is True:
            min = 0.000

            if man_strain is None:
                max = np.nanmax(E_array, axis=1)[1]
            else:
                max = man_strain
            # high resolution colorbar with matplotlib
            plt.rc('font', size=20)
            fig = plt.figure()
            ax = fig.add_axes([0.05, 0.08, 0.1, 0.9])
            cb = ColorbarBase(ax, orientation='vertical',
                              cmap=cmap,
                              norm=Normalize(min, round(max, 3)),
                              label=unit,
                              ticks=np.round(np.linspace(min, max, 8), decimals=3))
            fig.savefig(destination_dir / 'atomscolorbar.pdf', bbox_inches='tight')

        if man_strain is None:
            print(f"Maximum energy in  atom {int(np.nanargmax(E_atoms))}: {float(max_energy):.3f} {unit}.")
