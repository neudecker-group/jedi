import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d
from ase.build import molecule
from ase.data import covalent_radii, atomic_numbers
from ase.data.colors import jmol_colors
from ase.neighborlist import natural_cutoffs, neighbor_list


def detect_bonds_ase(atoms, bond_factor=1.1):
    """
    Detect bonds using ASE's neighborlist and natural_cutoffs.
    bond_factor: multiplicative factor for covalent radii sums.
    """
    cutoffs = natural_cutoffs(atoms, mult=bond_factor)
    i_array, j_array = neighbor_list('ij', atoms, cutoff=cutoffs)
    bonds = np.sort(np.vstack((i_array, j_array)).T, axis=1)
    bonds = np.unique(bonds, axis=0)
    return bonds


def plot_atoms_points_3d_option_c(atoms, bonds,
                                  bond_color='lightgrey',
                                  bond_color_map=None,
                                  radius_type='covalent',
                                  figsize=(8, 8),
                                  show_indices=False,
                                  bond_linewidth=3.0,
                                  halo_alpha=0.6,
                                  halo_factor=1.4):
    """
    Plot atoms with opaque cores and transparent halos to hide bonds inside atoms.
    Indices are drawn last using projection to always appear on top.
    """
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    # Radii selection
    if radius_type == 'covalent':
        radii = np.array([covalent_radii[atomic_numbers[sym]] for sym in symbols])
    elif radius_type == 'vdw':
        from ase.data import vdw_radii
        radii = np.array([vdw_radii[atomic_numbers[sym]] for sym in symbols])
    else:
        raise ValueError("radius_type must be 'covalent' or 'vdw'")
    atom_colors = [jmol_colors[atomic_numbers[sym]] for sym in symbols]
    # Scale sizes for scatter (points²)
    size_factor = 15
    core_sizes = (2 * radii * size_factor) ** 2
    halo_sizes = (2 * radii * size_factor * halo_factor) ** 2
    # Figure setup
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()
    ax.view_init(elev=20, azim=30)
    # --- Draw bonds first ---
    for (i, j) in bonds:
        if bond_color_map and (i, j) in bond_color_map:
            c = bond_color_map[(i, j)]
        elif bond_color_map and (j, i) in bond_color_map:
            c = bond_color_map[(j, i)]
        else:
            c = bond_color
        xs, ys, zs = zip(positions[i], positions[j])
        ax.plot(xs, ys, zs, color=c, linewidth=bond_linewidth)
    # --- Draw halos ---
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               s=halo_sizes, c=atom_colors, edgecolors='none',
               alpha=halo_alpha, depthshade=False)
    # --- Draw opaque cores ---
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               s=core_sizes, c=atom_colors, edgecolors='k',
               alpha=1.0, depthshade=False)
    # --- Draw atom indices last so they appear on top ---
    if show_indices:
        for idx, pos in enumerate(positions):
            # Project 3D position to 2D screen coordinates
            x2d, y2d, _ = proj3d.proj_transform(pos[0], pos[1], pos[2], ax.get_proj())
            ax.annotate(str(idx),
                        xy=(x2d, y2d),
                        xytext=(0, 0),
                        textcoords='offset points',
                        ha='center', va='center',
                        fontsize=8,
                        color='black',
                        zorder=10)  # ensure top layer
    plt.tight_layout()
    return fig, ax


# ------------------ Example usage ------------------
if __name__ == '__main__':
    # Create benzene molecule
    atoms = molecule('C6H6')
    # Detect bonds with ASE
    bonds = detect_bonds_ase(atoms, bond_factor=1.1)  # 10% tolerance
    # Find a specific C–H bond
    carbons = [i for i, sym in enumerate(atoms.get_chemical_symbols()) if sym == 'C']
    third_c = carbons[2]
    nearest_h = next((j for i, j in bonds if i == third_c and atoms[j].symbol == 'H'), None)
    if nearest_h is None:
        nearest_h = next((i for i, j in bonds if j == third_c and atoms[i].symbol == 'H'), None)
    custom_bonds = {(third_c, nearest_h): 'red'}
    # Plot with indices turned on
    fig, ax = plot_atoms_points_3d_option_c(atoms,
                                            bonds,
                                            bond_color='lightgrey',
                                            bond_color_map=custom_bonds,
                                            radius_type='covalent',
                                            figsize=(8, 8),
                                            show_indices=False,  # <-- indices ON
                                            bond_linewidth=6.0,
                                            halo_alpha=0.6,
                                            halo_factor=1.2)
    plt.show()
