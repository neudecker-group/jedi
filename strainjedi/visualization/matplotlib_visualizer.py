import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.colorbar import ColorbarBase


class MatplotlibVisualizer:
    """Generates simple visualization with matplotlib."""
    def __init__(self, visualization_data, mapper, output_dir, energy_unit="kcal/mol"):
        self.visualization_data = visualization_data
        self.mapper = mapper
        self.output_dir = Path(output_dir)
        self.energy_unit = energy_unit

    def run(self, show, show_indices, box=False):
        for mode in self.visualization_data.keys():
            self.mode = mode
            self.atomsF = self.visualization_data[self.mode]['bond_data']['atoms']
            self.symbols = np.unique([a.symbol for a in self.atomsF if a.symbol != 'H'])
            pos = self.atomsF.get_positions()
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_axis_off()
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor(None)
            ax.yaxis.pane.set_edgecolor(None)
            ax.zaxis.pane.set_edgecolor(None)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

            self.plot_bonds(ax, pos)
            self.plot_atoms(ax, pos, show_indices)

            if box and self.atomsF.pbc.any():
                self.plot_pbc_box(ax)

            self._set_3d_equal_aspect(ax, pos)
            self.colorbar(fig)

            ax.view_init(60, 0, 90)

            #plt.tight_layout()
            if show:
                plt.show()
            else:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(self.output_dir / f'{self.mode}_matplotlib.pdf', bbox_inches='tight')
            plt.close()

    def plot_bonds(self, ax, pos):
        bonds = self.visualization_data[self.mode]['bond_data']['bonds']
        custom_bonds = self.visualization_data[self.mode]['bond_data']['custom_bonds']
        norm_energies = self.visualization_data[self.mode]['bond_data']['norm_energies']
        norm_custom_energies = self.visualization_data[self.mode]['bond_data']['norm_custom_energies']
        self.max_strain = self.visualization_data[self.mode]['color_data']['max_strain']
        colormap = self.visualization_data[self.mode]['color_data']['colormap']

        colorlist = self.mapper.generate_colors(colormap, 256)
        self.cmap = LinearSegmentedColormap.from_list('strain', colorlist)

        for bond, energy in zip(bonds, norm_energies):
            p1, p2 = pos[int(bond[0])], pos[int(bond[1])]

            if np.isnan(energy):
                color = 'black'
            else:
                color = self.cmap(energy)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    color=color, linewidth=5)

        for bond, energy in zip(custom_bonds, norm_custom_energies):
            p1, p2 = pos[int(bond[0])], pos[int(bond[1])]

            if np.isnan(energy):
                color = 'black'
            else:
                color = self.cmap(energy)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    color=color, linewidth=2, linestyle='--')

    def plot_atoms(self, ax, pos, show_indices):
        atom_colors = self.visualization_data[self.mode]['color_data']['atom_colors']
        symbols = self.atomsF.get_chemical_symbols()
        
        for idx, (p, (sym, col)) in enumerate(zip(pos, zip(symbols, atom_colors))):
            size = 100 if sym == 'H' else 200
            ax.scatter(p[0], p[1], p[2], color=col, s=size, linewidths=0.5, edgecolor='0.3')
            if show_indices:
                ax.text(p[0], p[1], p[2], str(idx),
                        fontsize=6 if sym == 'H' else 8,
                        color='k',
                        ha='center',
                        va='center',
                        zorder=100)

    def plot_pbc_box(self, ax):
        cell = self.atomsF.get_cell()
        # The 8 corners of the parallelepiped defined by the 3 cell vectors
        a, b, c = cell[0], cell[1], cell[2]
        origin = np.array([0.0, 0.0, 0.0])

        corners = np.array([
            origin,          # 0
            a,               # 1
            b,               # 2
            a + b,           # 3
            c,               # 4
            a + c,           # 5
            b + c,           # 6
            a + b + c,       # 7
        ])

        # 12 edges of the parallelepiped
        edges = [
            (0, 1), (0, 2), (0, 4),  # from origin
            (1, 3), (1, 5),           # from a
            (2, 3), (2, 6),           # from b
            (4, 5), (4, 6),           # from c
            (3, 7), (5, 7), (6, 7),   # to a+b+c
        ]

        for i, j in edges:
            p1, p2 = corners[i], corners[j]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    color='black', linewidth=0.8, linestyle='-', alpha=0.5)

    def colorbar(self, fig):
        cbar_ax = fig.add_axes([0.8, 0.2, 0.06, 0.6])
        ColorbarBase(cbar_ax,
                     orientation='vertical',
                     cmap=self.cmap,
                     norm=Normalize(0., round(self.max_strain, 3)),
                     label=self.energy_unit,
                     ticks=np.round(np.linspace(0, self.max_strain, 8), decimals=3))

    def _set_3d_equal_aspect(self, ax, positions):
        """Equal aspect for 3D."""
        # Include cell corners in the range calculation when PBC box is shown
        all_points = positions
        if self.atomsF.pbc.any():
            cell = self.atomsF.get_cell()
            a, b, c = cell[0], cell[1], cell[2]
            origin = np.array([0.0, 0.0, 0.0])
            corners = np.array([origin, a, b, a + b, c, a + c, b + c, a + b + c])
            all_points = np.vstack([positions, corners])

        max_range = np.array([
            all_points[:, 0].max() - all_points[:, 0].min(),
            all_points[:, 1].max() - all_points[:, 1].min(),
            all_points[:, 2].max() - all_points[:, 2].min()
        ]).max() / 2.0

        mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
        mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
        mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)