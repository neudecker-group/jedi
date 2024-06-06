import numpy as np
import os
from pathlib import Path
import matplotlib.cm as cm
from typing import Dict, Optional, Union
from ase.atoms import Atom
import ase.neighborlist
from ase.units import Hartree, Bohr, mol, kcal
from strainjedi.colors import colors
from strainjedi.jedi import Jedi


class JediAtoms(Jedi):

    E_atoms=None

    def run(self, ase_units=False, indices=None):
        """Runs the analysis. Calls all necessary functions to get the needed values.

        Args:
            indices:
                list of indices of a substructure if desired
            ase_units: boolean
                flag to get eV for energies Å for lengths otherwise it is kcal/mol, Bohr
        Returns:
            Indices, strain, energy in every RIM
        """
        self.ase_units = ase_units
        # get necessary data
        self.indices=np.arange(0,len(self.atoms0))
        if indices:
            self.indices=indices


        self.get_hessian()
        H_cart = self.H         #Hessian of optimized (ground state) structure
        delta_x= self.get_delta_x()

        if len(self.atoms0) != H_cart.shape[0] / 3:
            raise ValueError(
                'Hessian has not the fitting shape, possibly a partial hessian. Please try partial_analysis')
        try:
            all_E_geometries = self.get_energies()
        except:
            all_E_geometries = self.energies
        E_geometries = all_E_geometries[0]


    # Get the energy stored in every coordinate (take care to get the right multiplication for a diatomic molecule)
        E_coords = np.sum(0.5*(delta_x*H_cart).T*delta_x,axis=1)
        self.E_atoms=np.sum(E_coords.reshape(-1, 3), axis=1)
        if ase_units==True:

            self.E_atoms*=Hartree
            delta_x*=Bohr
        elif ase_units == False:
            self.E_atoms *= mol/kcal*Hartree
        self.printout(E_geometries)
        pass

    def get_delta_x(self):
        from ase.geometry import find_mic
        from rmsd import kabsch_fit
        if self.atomsF.get_pbc().any() is False:
            minimised_atomsF = kabsch_fit(self.atomsF.positions,
                                          self.atoms0.positions)  # to get rid of strain for translation and rotation
            delta_x = minimised_atomsF - self.atoms0.positions
            return delta_x.flatten() / Bohr
        else:
            delta_x = self.atomsF.positions - self.atoms0.positions
            return find_mic(delta_x, self.atomsF.cell)[0].flatten() / Bohr  # find minimal delta_x under pbc conditions

    def printout(self, E_geometries, save=False):
        '''
        Printout of analysis of stored strain energy in the bonds.
        '''
        #############################################
        #	    	   Output section	        	#
        #############################################
        # Header
        output = []
        output.append("\n ************************************************")
        output.append("\n *                 JEDI ANALYSIS                *")
        output.append("\n *       Judgement of Energy DIstribution       *")
        output.append("\n ************************************************\n")

        # Comparison of total energies
        if self.ase_units==False:
            output.append("\n                   Strain Energy (kcal/mol)  Deviation (%)")
        elif self.ase_units==True:
            output.append("\n                   Strain Energy (eV)        Deviation (%)")

        E_atoms_total = sum(self.E_atoms[self.indices])
        output.append("\n      Ab initio   " + "{:14.8f}".format(E_geometries) + "                -")
        output.append(
            '\n{:17s}{:15.8f}{:22.2f}'.format("      JEDI", E_atoms_total, (E_atoms_total / E_geometries - 1) * 100))

        delta_x = self.get_delta_x().reshape(-1, 3)
        # strain in the bonds

        if self.ase_units == False:
            output.append(
                "\n Atom No.   Element    delta_x (a.u.)                      Percentage         Energy (kcal/mol)")
        elif self.ase_units == True:
            output.append("\n Atom No.   Element    delta_x (Å)                         Percentage         Energy (eV)")

        for i, k in enumerate(self.E_atoms[self.indices]):
            output.append(
                '\n{:>3d}         {:<2s}          {:>7.3f}   {:>7.3f}   {:>7.3f}       {:>8.2f}%       {:16.7f}'.format(
                    self.indices[i],
                    self.atoms0.symbols[self.indices[i]],
                    delta_x[i][0],
                    delta_x[i][1],
                    delta_x[i][2],
                    k / E_atoms_total * 100,
                    k
                ))

        if save is False:
            print(*output)
        else:
            f = open('E_atoms', 'w')
            f.writelines(output)
            f.close()

    def partial_analysis(self, indices, ase_units=False):

        """Runs the analysis. Calls all necessary functions to get the needed values.

        Args:
            indices:
                list of indices of a substructure
            ase_units: boolean
                flag to get eV for energies å fo lengths otherwise it is kcal/mol, Bohr
        Returns:
            Indices, strain, energy in every RIM
        """
        self.ase_units = ase_units
        # get necessary data
        self.indices=indices


        self.get_hessian()
        H_cart = self.H         #Hessian of optimized (ground state) structure
        #get strain in coordinates
        i = np.repeat(np.atleast_2d(indices),3,axis=0)*3
        i[1]+=1
        i[2]+=2
        i = i.ravel('F')
        delta_x= self.get_delta_x()[i]
#        print(delta_x)

        if len(indices) != H_cart.shape[0]/3:
            raise ValueError('Hessian has not the fitting shape')
        try:
            all_E_geometries = self.get_energies()
        except:
            all_E_geometries = self.energies
        E_geometries = all_E_geometries[0]


    # Get the energy stored in every coordinate (take care to get the right multiplication for a diatomic molecule)
        E_coords = np.sum(0.5*(delta_x*H_cart).T*delta_x,axis=1)
        self.E_atoms=np.sum(E_coords.reshape(-1, 3), axis=1)
        if ase_units==True:

            self.E_atoms*=Hartree
            delta_x*=Bohr
        elif ase_units == False:
            self.E_atoms *= mol / kcal * Hartree
        E_atoms = np.full((len(self.atoms0)), np.nan)
        E_atoms[self.indices] = self.E_atoms
        self.E_atoms = E_atoms
        self.printout(E_geometries)

    def get_bonds(self, mol):
        '''Gets list of bonds in mol

        '''
        mol = mol

        indices = self.indices
        cutoff = ase.neighborlist.natural_cutoffs(mol, mult=self.covf)  ## cutoff for covalent bonds see Bakken et al.
        bl = np.vstack(ase.neighborlist.neighbor_list('ij', a=mol, cutoff=cutoff)).T  # determine covalent bonds

        bl = bl[bl[:, 0] < bl[:, 1]]  # remove double metioned
        bl, counts = np.unique(bl, return_counts=True, axis=0)
        if ~ np.all(counts == 1):
            print('unit cell too small hessian not calculated for self interaction \
                   jedi analysis for a finite system consisting of the cell will be conducted')
        bl = np.atleast_2d(bl)

        if len(indices) != len(mol):
            bl = bl[np.all([np.in1d(bl[:, 0], indices), np.in1d(bl[:, 1], indices)], axis=0)]

        return bl

    def vmd_gen(self,
                des_colors: Optional[Dict] = None,
                box: bool = False,
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
        output.append(f'\n# Load a molecule\nmol new {destination_dir.resolve() / "xF.xyz"}\n\n')
        output.append('# Change bond radii and various resolution parameters\nmol representation cpk 0.8 0.0 30 '
                      '5\nmol representation bonds 0.2 30\n\n')


        output.append('# Change the color of the graphical representation 0 to white\ncolor change rgb 0 1.00 1.00 '
                      '1.00\n')
        output.append('# The background should be white ("blue" has the colorID 0, which we have changed to '
                      'white)\ncolor Display Background blue\n\n')
        output.append('# Define the other colorIDs\n')


        # Define colorcodes for various atomtypes

        #from .colors import colors
        if des_colors is not None:
            for i in des_colors:
                colors[i] = des_colors[i]         #desired colors overwrite the standard ones

        symbols = np.unique(self.atomsF.get_chemical_symbols())
        symbols = symbols[symbols != 'H']           #get all symbols except H, H is white

        N_colors_atoms = len(symbols)
        N_colors = 32 - N_colors_atoms - 1           #vmd only supports 32 colors for modcolor


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
        #define color of atoms with the color code above
        for j in range(N_colors_atoms):
            output.append('\n\nmol representation cpk 0.7 0.0 30 5')
            output.append('\nmol addrep top')
            output.append('\n%s%i%s' % ("mol modstyle ", j+1, " top cpk"))
            output.append('\n%s%i%s%i%s' % ("mol modcolor ", j+1, " top {colorid ", N_colors+j+1, "}"))
            output.append('\n%s%i%s%s%s' % ("mol modselect ", j+1, " top {name ", symbols[j], "}"))





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
        if pbc_flag == True:
            E_array_pbc = np.empty((2, 0))

            from ase.data.vdw import vdw_radii  # for long range bonds
            cutoff = [vdw_radii[atom.number] * self.vdwf for atom in self.atomsF]
            ex_bl = np.vstack(ase.neighborlist.neighbor_list('ij', a=self.atomsF, cutoff=cutoff)).T
            ex_bl = np.hstack((ex_bl, ase.neighborlist.neighbor_list('S', a=self.atomsF, cutoff=cutoff)))
            ex_bl = np.hstack((ex_bl, ase.neighborlist.neighbor_list('D', a=self.atomsF, cutoff=cutoff)))
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
                                int(atoms_ex_cell[i, 1])]  # get the indices of the corresponding atoms inside the cell
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

            mol.write('xF.xyz')  # save the modified structure with auxilliary atoms for vmd
            E_array = np.hstack((E_array, E_array_pbc))

        # Store the maximum energy in a variable for later call
        max_energy = float(np.nanmax(E_array, axis=1)[1])  # maximum energy in one atom

        # Generate the binning windows by splitting E_array into N_colors equal windows

        if man_strain == None:

            binning_windows = np.linspace(0, np.nanmax(E_array, axis=1)[1], num=N_colors )
        else:
            binning_windows = np.linspace(0, float(man_strain), num=N_colors )

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
            output.append('\n%s%i%s' % ("mol modstyle ", N_colors_atoms+i+1, " top cpk"))
            output.append('\n%s%i%s%i%s' % ("mol modcolor ", N_colors_atoms+i+1, " top {colorid ", colorID, "}"))
            output.append('\n%s%i%s%s%s' % ("mol modselect ", N_colors_atoms+i+1, " top {index ", int(i), "}\n"))
        f = open(destination_dir / 'atoms.vmd', 'w')
        f.writelines(output)
        f.close()

        #colorbar
        if colorbar==True:
            min=0.000

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
            print(f"Maximum energy in  atom {int(np.argmax(E_atoms) + 1)}: {float(max_energy):.3f} {unit}.")

        # save printout in folder
        try:
            all_E_geometries = self.get_energies()
        except:
            all_E_geometries = self.energies
        E_geometries = all_E_geometries[0]
        self.printout(E_geometries, save=True)

        os.chdir('..')
        pass

    def pov_gen(self, colorbar=True, box=False, man_strain=None, label='pov', incl_coloring=None, view_dir=None,
                zoom=1., tex='vmd',
                radii=1., scale_radii=None, bond_colors=None, cameratype='perspective', cameralocation=(0., 0., 20.),
                look_at=(0., 0., 0.), camera_right_up=[(-8., 0., 0.), (0., 6., 0.)], cameradirection=(0., 0., 10.),
                light=None,
                background='White', bondradius=.1, pixelwidth=2000, aspectratio=None):
        '''Generates POV object for atoms object

                Args:
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
                    view_dir: str
                        camera view
                        'x': from the x-axis at the y,z-plane
                        'y': from the y-axiy at the z,x-plane
                        'z': from the z-axis at the x,y-plane
                    kwargs:
                        see iopov.py
                '''

        from strainjedi.io.iopov import POV
        from matplotlib.colors import LinearSegmentedColormap
        import builtins

        try:  # ToDo: pov_gen ohne chdir
            os.mkdir(label)
        except:
            pass
        os.chdir(label)

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

        # get atoms for bonds that reach out of the unit cell and their energies
        if pbc_flag == True:
            E_array_pbc = np.empty((2, 0))

            from ase.data.vdw import vdw_radii  # for long range bonds
            cutoff = [vdw_radii[atom.number] * self.vdwf for atom in self.atomsF]
            ex_bl = np.vstack(ase.neighborlist.neighbor_list('ij', a=self.atomsF, cutoff=cutoff)).T
            ex_bl = np.hstack((ex_bl, ase.neighborlist.neighbor_list('S', a=self.atomsF, cutoff=cutoff)))
            ex_bl = np.hstack((ex_bl, ase.neighborlist.neighbor_list('D', a=self.atomsF, cutoff=cutoff)))
            atoms_ex_cell = ex_bl[(ex_bl[:, 2] != 0) | (ex_bl[:, 3] != 0) | (
                        ex_bl[:, 4] != 0)]  # determines which nearest neighbors are outside the unit cell
            atoms_f.wrap()  # wrap molecule important for atoms close to the boundaries
            bondscheck = self.get_bonds(self.atomsF)

            for i in range(len(atoms_ex_cell)):
                pos_ex_atom = atoms_f.get_positions()[int(atoms_ex_cell[i, 0])] + atoms_ex_cell[i,
                                                                                  5:8]  # get positions of cell external atoms by adding the vector
                # if pos_ex_atom in mol.positions:
                original_rim = [int(atoms_ex_cell[i, 0]),
                                int(atoms_ex_cell[i, 1])]  # get the indices of the corresponding atoms inside the cell
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

        # generating pov object with specified view direction, write .pov file and run it
        positions = atoms_f.get_positions()
        center = np.mean(positions, axis=0)
        cell = None
        if box and pbc_flag is True:
            cell = atoms_f.cell
            center = 0.5 * cell[0] + 0.5 * cell[1] + 0.5 * cell[2]
        if view_dir is False:
            if light is None:
                light = cameralocation
            if look_at == 'center':
                look_at = center
                cameralocation += center
                cameradirection += center
                light += cameralocation
            pov = POV(atoms_f,
                      tex=tex,
                      radii=radii,
                      scale_radii=scale_radii,
                      bond_colors=bond_colors,
                      atom_colors=atom_colors,
                      cameratype=cameratype,
                      cameralocation=cameralocation,
                      look_at=look_at,
                      camera_right_up=camera_right_up,
                      cameradirection=cameradirection,
                      area_light=[light, 'White', 1.7, 1.7, 3, 3],
                      background=background,
                      bondatoms=bonds,
                      bondradius=bondradius,
                      pixelwidth=pixelwidth,
                      aspectratio=aspectratio,
                      cell=cell
                      )
            pov.write(f'{label}.png')
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
                        camwidth = [(-0.5 * abs(center[0] - builtins.min(atoms_rotated.positions[:, 0]) + 2.0), 0., 0.),
                                    (0., 0.5 * abs(center[1] - builtins.max(atoms_rotated.positions[:, 1])) + 1., 0.)]
                    location = center.copy()
                    location[2] += 60. * zoom
                    direction = center.copy()
                    direction[2] += 10.
                    light = center.copy()
                    light[2] += 20.
                    pov = POV(atoms_rotated,
                              tex=tex,
                              radii=radii,
                              scale_radii=scale_radii,
                              bond_colors=bond_colors,
                              atom_colors=atom_colors,
                              cameratype=cameratype,
                              cameralocation=location,
                              look_at=center,
                              camera_right_up=camwidth,
                              cameradirection=direction,
                              area_light=[light, 'White', 1.7, 1.7, 3, 3],
                              background=background,
                              bondatoms=bonds,
                              bondradius=bondradius,
                              pixelwidth=pixelwidth,
                              aspectratio=aspectratio,
                              cell=cell
                              )
                    pov.write(f'{label}_{view}.png')

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
            import matplotlib.pyplot as plt
            from matplotlib.colorbar import ColorbarBase
            from matplotlib.colors import Normalize
            plt.rc('font', size=20)
            fig = plt.figure()
            ax = fig.add_axes([0.05, 0.08, 0.1, 0.9])
            cb = ColorbarBase(ax, orientation='vertical',
                              cmap=cmap,
                              norm=Normalize(min, round(max, 3)),
                              label=unit,
                              ticks=np.round(np.linspace(min, max, 8), decimals=3))
            fig.savefig('atomscolorbar.pdf', bbox_inches='tight')

        if man_strain is None:
            print(f"Maximum energy in  atom {int(np.argmax(E_atoms) + 1)}: {float(max_energy):.3f} {unit}.")

        # save printout in folder
        try:
            all_E_geometries = self.get_energies()
        except:
            all_E_geometries = self.energies
        E_geometries = all_E_geometries[0]
        self.printout(E_geometries, save=True)

        os.chdir('..')
        pass
