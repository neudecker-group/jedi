from strainjedi.visualization.colors import colors
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LinearSegmentedColormap, Normalize

class VMDVisualizer:
    """Generates VMD TCL scripts"""
    def __init__(self, visualization_data, mapper, output_dir, energy_unit = "kcal/mol"):
        self.visualization_data = visualization_data
        self.mapper = mapper
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.energy_unit = energy_unit


    def write_inputs(self, box):
        for mode in self.visualization_data.keys():
            self.mode = mode
            self.atomsF = self.visualization_data[self.mode]['bond_data']['atoms']
            self.symbols = np.unique([a.symbol for a in self.atomsF if a.symbol != 'H'])
            self.atomsF.write(self.output_dir / 'xF.xyz')
            
            lines = []
            lines.extend(self.write_header())
            lines.extend(self.write_color_definitions())
            lines.extend(self.write_atom_bond_representations())

            if self.atomsF.pbc.any() and box:
                lines.extend(self.draw_box())
            lines.extend(self.vmd_colorbar())
            self.pdf_colorbar()

            with open(self.output_dir / f'{mode}.vmd', 'w') as f:
                f.write('\n'.join(lines))
                       
    def write_header(self):
        lines = [
            f'\n# Load molecule',
            f'mol new {{{self.output_dir.resolve() / "xF.xyz"}}} type xyz\n',
            '\n# Change bond radii and various resolution parameters',
            'mol representation cpk 0.8 0.0 30 5',
            'mol representation bonds 0.2 30\n',
            '\n# Change the drawing method of the first graphical representation to CPK',
            'mol modstyle 0 top cpk',
            '\n# Color only H atoms white',
            'mol modselect 0 top {name H}',
            '\n# Change the color of the graphical representation 0 to white',
            'color change rgb 0 1.00 1.00 1.00',
            'mol modcolor 0 top {colorid 0}',
            '\n# The background should be white ("blue" has the colorID 0, which we have changed to white)',
            'color Display Background blue\n'
        ]
        return lines
    
    def write_color_definitions(self):
        lines = ['\n# Define the other colorIDs']

        self.cmap = self.visualization_data[self.mode]["color_data"]["colormap"]
        self.n_colors = 32 - len(self.symbols) - 1

        self.colorlist = self.mapper.generate_colors(self.cmap, self.n_colors)

        for i, rgb in enumerate(self.colorlist):
            lines.append(f'color change rgb {i+1:5d} {rgb[0]:10.6f} {rgb[1]:10.6f} {rgb[2]:10.6f}')


        atom_colors = self.visualization_data[self.mode]['color_data']['atom_colors']
        unique_atom_colors = {}
        for i, atom in enumerate(self.atomsF):
            if atom.symbol != 'H' and atom.symbol not in unique_atom_colors:
                unique_atom_colors[atom.symbol] = atom_colors[i]

        for i, symbol in enumerate(self.symbols):
            rgb = unique_atom_colors[symbol]
            lines.append(f'\ncolor change rgb {self.n_colors+i+1:5d} {rgb[0]:10.6f} {rgb[1]:10.6f} {rgb[2]:10.6f}')

        lines.append('\ncolor change rgb 32 0.000000 0.000000 0.000000')  # Black for NaN
        lines.append('\ncolor change rgb 1039 1.000000 0.000000 0.000000')  # Red for X-axis
        lines.append('\ncolor change rgb 1038 0.000000 1.000000 0.000000')  # Green for Y-axis
        lines.append('\ncolor change rgb 1037 0.000000 0.000000 1.000000')  # Blue for Z-axis
        lines.append('\ncolor change rgb 1036 0.250000 0.750000 0.750000')  # Cyan for origin
        lines.append('\ncolor Axes X 1039')
        lines.append('color Axes Y 1038')
        lines.append('color Axes Z 1037')
        lines.append('color Axes Origin 1036')
        lines.append('color Axes Labels 32')

        return lines
    
    def write_atom_bond_representations(self):
        lines = []

        # Atom representations
        for i, symbol in enumerate(self.symbols):
            colorID = self.n_colors + i + 1
            lines.append('\nmol representation cpk 0.7 0.0 30 5')
            lines.append('mol addrep top')
            lines.append(f'mol modstyle {i+1} top cpk')
            lines.append(f'mol modcolor {i+1} top {{colorid {colorID}}}')
            lines.append(f'mol modselect {i+1} top {{name {symbol}}}')

        # Bond representations
        bonds = self.visualization_data[self.mode]['bond_data']['bonds']
        norm_energies = self.visualization_data[self.mode]['bond_data']['norm_energies']
        custom_bonds = self.visualization_data[self.mode]['bond_data']['custom_bonds']
        norm_custom_energies = self.visualization_data[self.mode]['bond_data']['norm_custom_energies']
        self.max_strain = self.visualization_data[self.mode]['color_data']['max_strain']


        binning_windows = np.linspace(0, 1, num=self.n_colors)
        n_atom_reps = len(self.symbols)

        # regular bonds
        for i in range(len(bonds)):
            bond = bonds[i]
            energy = norm_energies[i]

            if np.isnan(energy):
                colorID = 32 # black
            else:
                colorID = np.abs(binning_windows - energy).argmin() + 1

            
            rep_id = n_atom_reps + i + 1
            lines.append(f'mol addrep top')
            lines.append(f'mol modstyle {rep_id} top bonds')
            lines.append(f'mol modcolor {rep_id} top {{colorid {colorID}}}')
            lines.append(f'mol modselect {rep_id} top {{index {int(bond[0])} {int(bond[1])}}}\n')

        # Custom bonds
        if len(custom_bonds) > 0:
            lines.append('\n# Custom bonds (dashed lines)\n')
            
            for i in range(len(custom_bonds)):
                bond = custom_bonds[i]
                energy = norm_custom_energies[i]
                
                # Bin energy
                if np.isnan(energy):
                    colorID = 32
                else:
                    colorID = np.abs(binning_windows - energy).argmin() + 1
                
                lines.append(f'\nset x [[atomselect top "index {int(bond[0])} {int(bond[1])}"] get {{x y z}}]')
                lines.append('set a [lindex $x 0]')
                lines.append('set b [lindex $x 1]')
                lines.append(f'draw color {colorID}')
                lines.append('draw line $a $b width 3 style dashed')

        return lines
    
    def vmd_colorbar(self):
        """Generate TCL code for interactive colorbar in VMD."""
       
        lines = [f"""display update off
display resetview
variable bar_mol
                 


set old_top [molinfo top]
set bar_mol [mol new]
mol top $bar_mol
                 
#bar can be fixed with mol fix 'molid of the bar' 

                 

# We want to draw relative to the location of the top mol so that the bar 
# will always show up nicely.
set center [molinfo $old_top get center]
set center [regsub -all {[{}]} $center ""]
set center [split $center]
set min 0.0
set max 2.219368753453463
set length 30.0
set width [expr $length / 6]

# draw the color bar
set start_y [expr 1 + [lindex $center 1] ]

set use_x [expr 1 + [lindex $center 0] ]

set use_z [expr 1+ [lindex $center 2 ]]

set step [expr $length / 29]

set label_num 8

for {{set colorid 1 }} {{ $colorid <= {self.n_colors} }} {{incr colorid 1 }} {{
    draw color $colorid
    set cur_y [ expr $start_y + ($colorid -0.5 ) * $step ]
    draw line "$use_x $cur_y $use_z"  "[expr $use_x+$width] $cur_y $use_z" width 10000
}}
# draw the labels
set coord_x [expr (1.1*$width)+$use_x];
set step_size [expr $length / $label_num]
set color_step [expr 29/$label_num]
set value_step [expr ($max - $min ) / double ($label_num)]


for {{set i 0}} {{$i <= $label_num }} {{ incr i 1}} {{
    set cur_color_id 32
    draw color $cur_color_id
    set coord_y [expr $start_y+$i * $step_size ]
    set cur_text [expr $min + $i * $value_step ]
    draw text  " $coord_x $coord_y $use_z"  [format %6.3f  $cur_text]
}}
draw text " $coord_x [expr $coord_y + $step_size] $use_z"   "kcal/mol"
# re-set top
mol top $old_top
display update on 
"""]
        return lines
    
    def draw_box(self):
        lines = []
        lines.append("\n\n# Adding a pbc box")
        lines.append('\npbc set {%f %f %f %f %f %f}'
                     % (self.atomsF.cell.cellpar()[0],
                        self.atomsF.cell.cellpar()[1],
                        self.atomsF.cell.cellpar()[2],
                        self.atomsF.cell.cellpar()[3],
                        self.atomsF.cell.cellpar()[4],
                        self.atomsF.cell.cellpar()[5]))
        lines.append("\npbc box -color 32")
        lines.append("\n\n# Adding a representation with the appropriate colorID for each bond")

        return lines

    def pdf_colorbar(self):
        plt.rc('font', size=20)
        fig = plt.figure()
        ax = fig.add_axes([0.05, 0.08, 0.1, 0.9])
        cmap_name = 'my_list'
        cmap = LinearSegmentedColormap.from_list(cmap_name, self.colorlist, N=self.n_colors)
        ColorbarBase(ax,
                     orientation='vertical',
                     cmap=cmap,
                     norm=Normalize(0.,round(self.max_strain,3)),
                     label=self.energy_unit,
                     ticks=np.round(np.linspace(0, self.max_strain, 8), decimals=3))

        fig.savefig(f'{self.output_dir / self.mode}colorbar.pdf', bbox_inches='tight')