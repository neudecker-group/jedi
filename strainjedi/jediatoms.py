import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union
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
                flag to get eV for energies å fo lengths otherwise it is kcal/mol, Bohr
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

        if len(self.atoms0) != H_cart.shape[0]/3:
            raise ValueError('Hessian has not the fitting shape, possibly a partial hessian. Please try partial_analysis')
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
        return (self.atomsF.positions.flatten()-self.atoms0.positions.flatten())/Bohr

    def printout(self,E_geometries):
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
        output.append("\n      Ab initio     " + "%.8f" % E_geometries + "                  -")
        output.append('\n%5s%16.8f%21.2f' % (" JEDI           ", E_atoms_total, (E_atoms_total / E_geometries-1)*100))


        # strain in the bonds

        if self.ase_units == False:
            output.append("\n Atom No.       Element                              Percentage    Energy (kcal/mol)")
        elif self.ase_units == True:
            output.append("\n Atom No.       Element                              Percentage    Energy (eV)")


        for i, k in enumerate(self.E_atoms[self.indices]):
            output.append('\n%6i%7s%-11s%9.1f%17.7f' % (self.indices[i], " ", self.atoms0.symbols[self.indices[i]], k/E_atoms_total, k))
        print(*output)

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
            self.E_atoms *= mol/kcal*Hartree
        self.printout(E_geometries)

    def vmd_gen(self,
                des_colors: Optional[Dict] = None,
                box: bool = False,
                man_strain: Optional[float] = None,
                colorbar: bool = True,
                label: Union[Path, str] = 'vmd'):
        """
        Generates vmd scripts and files to save the values for the color coding

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

        #get green to red gradient
        for i in range(N_colors):
            R_value = float(i)/(N_colors/2)
            if R_value > 1:
                R_value = 1
            if N_colors % 2 == 0:
                G_value = 2 - float(i+1)/(N_colors/2)
            if N_colors % 2 != 0:
                G_value = 2 - float(i)/(N_colors/2)
            if G_value > 1:
                G_value = 1

            B_value = 0

            output.append('%1s%5i%10.6f%10.6f%10.6f%1s' % ("color change rgb", i+1, R_value, G_value, B_value, "\n"))
            colorbar_colors.append((R_value, G_value, B_value))

        # add color codes of atoms
        for j in range(N_colors_atoms):
            output.append('%1s%5i%10.6f%10.6f%10.6f%1s'
                          % ("color change rgb",
                             N_colors+j+1,
                             float(colors[symbols[j]][0]),
                             float(colors[symbols[j]][1]),
                             float(colors[symbols[j]][2]), "\n"))

        #add color code for axes and box
        output.append('%1s%5i%10.6f%10.6f%10.6f%1s' % ("color change rgb", 32, float(0), float(0), float(0), "\n"))#black
        output.append('%1s%5i%10.6f%10.6f%10.6f%1s' % ("color change rgb", 1039, float(1), float(0), float(0), "\n"))#red
        output.append('%1s%5i%10.6f%10.6f%10.6f%1s' % ("color change rgb", 1038, float(0), float(1), float(0), "\n"))#green
        output.append('%1s%5i%10.6f%10.6f%10.6f%1s' % ("color change rgb", 1037, float(0), float(0), float(1), "\n"))#blue
        output.append('%1s%5i%10.6f%10.6f%10.6f%1s' % ("color change rgb", 1036, float(0.25), float(0.75), float(0.75), "\n"))#cyan
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


        # Create an array that stores the atom as the first entry The energy will be added as the secondd entry.
        E_array = np.full((len(self.E_atoms)),np.nan)



        # Create an array that stores only the energies in the coordinate of interest and print some information
        # Get rid of ridiculously small values and treat diatomic molecules explicitly
        # (in order to create a unified picture, we have to create all these arrays in any case)


        if E_atoms.max() <= 0.001:
            E_atoms = np.zeros(len(self.indices))
        E_array[list([*self.indices])]=E_atoms[self.indices] if  len(self.indices) != len(E_atoms) else E_atoms
        E_array=np.vstack((np.arange(len(self.indices)),E_array[self.indices]))
        print("\nProcessing atoms...")

    # Store the maximum energy in a variable for later call

        max_energy = float(np.nanmax(E_array, axis=1)[1])  # maximum energy in one bond

    # Generate the binning windows by splitting bond_E_array into N_colors equal windows


        if man_strain == None:

            binning_windows = np.linspace(0, np.nanmax(E_array, axis=1)[1], num=N_colors )
        else:
            binning_windows = np.linspace(0, float(man_strain), num=N_colors )





        if box  :

            output.append("\n\n# Adding a pbc box")
            output.append('\npbc set {%f %f %f %f %f %f}'
                          %(self.atomsF.cell.cellpar()[0],
                            self.atomsF.cell.cellpar()[1],
                            self.atomsF.cell.cellpar()[2],
                            self.atomsF.cell.cellpar()[3],
                            self.atomsF.cell.cellpar()[4],
                            self.atomsF.cell.cellpar()[5]))
            output.append("\npbc box -color 32")
        output.append("\n\n# Adding a representation with the appropriate colorID for each atom")
            # Calculate which binning_windows value is closest to the bond-percentage and do the output


        for i, b in zip(E_array[0],E_array[1]):
            if np.isnan(b):
                colorID = 32                       #black
            else:
                colorID = np.abs( binning_windows - b ).argmin() + 1

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


            #highresolution colorbar with matplotlib
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
                                        norm=Normalize(min,round(max,3)),
                                        label=unit,
                                        ticks=np.round(np.linspace(min, max, 8),decimals=3))

            fig.savefig(destination_dir / 'atomscolorbar.pdf', bbox_inches='tight')

        if man_strain==None:
            print("\nAdding all energies for the stretch, bending and torsion of the bond with maximum strain...")
            print(f"Maximum energy in  atom {int(np.argmax(E_atoms)+1)}: {float(max_energy):.3f} {unit}.")
