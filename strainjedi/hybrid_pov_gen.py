from strainjedi.jedi import Jedi
from strainjedi.jediatoms import JediAtoms
from strainjedi.io.iopov import POV
import numpy as np
from typing import Any, Dict, Optional, Union, List, Sequence, Literal
from pathlib import Path
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import builtins
from ase.atoms import Atoms
import ase.io
from ase.data import covalent_radii
from ase.data.colors import jmol_colors


def hybrid_pov_gen(jedi: Jedi = None,
                   jediatoms: JediAtoms = None,
                   color_unanalysed: tuple = (0.75,0.75,0.75),
                   colorbar: bool = True,
                   box: bool = False,
                   man_strain: Optional[float] = None,
                   label: Union[Path, str] = 'pov',
                   incl_coloring: Optional[Literal['cyan', 'magma']] = None,
                   view_dir: Optional[Union[Literal['x', 'y', 'z'], Atoms]] = None,
                   zoom: float = 1.,
                   metal: Optional[list] = None,
                   tex: Union[str, list, np.ndarray] = 'vmd',
                   radii: float = 1.,
                   scale_radii: Optional[float] = None,
                   light: Optional[Union[Sequence[float], np.ndarray]] = None,
                   background: str = 'White',
                   bondradius: float = .1,
                   pixelwidth: int = 2000,
                   aspectratio: Optional[float] = None,
                   run_pov: bool = True):
    """Generates POV object for atoms object and hybrid jedi + jediatoms analysis coloring
                Args:
                    jedi: Jedi object
                        partial jedi analysis of the Atoms object
                    jediatoms: JediAtoms object
                        partial jediatoms analysis of the Atoms object
                    color_unanalysed: tuple
                        color for jediatoms bonds and jedi atoms as a (r,g,b) tuple
                        Default: (0.8,0.8,0.8)
                    box: boolean
                        True: draw box
                        False: ignore box
                        default: False
                    man_strain: float
                        reference value for the strain energy used in the color scale
                        default: 'None'
                    colorbar: boolean
                        True: save colorbar
                        default: False
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
                    tex: str or list
                        a texture to use for the atoms, either a single value or a list
                        of len(atoms),
                        default = 'vmd'
                    radii: float or list
                        atomic radii. if a single value is given, it is interpreted as a multiplier for the covalent radii
                        in ase.data. if a list of len(atoms) is given, it is interpreted as individual atomic radii
                        default: 1.0
                    scale_radii: float
                        float or list with floats that scale the atomic radii
                        default: 0.5
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

    if isinstance(label, str):
        destination_dir = Path(label)
    elif isinstance(label, Path):
        destination_dir = label
    else:
        raise TypeError("Please specify the directory (label) to write vmd scripts to as Path or string")
    destination_dir.mkdir(parents=True, exist_ok=True)

    if type(view_dir) is ase.Atoms:
        atoms_f = view_dir
    else:
        atoms_f = jedi.atomsF.copy()
    pbc_flag = False
    if atoms_f.get_pbc().any() == True:
        pbc_flag = True

    #test if same units for jedi and jediatoms are used
    if jedi.ase_units != jediatoms.ase_units:
        raise TypeError("Units for jedi and jediatoms analysis have to be the same")
    else:
        ase_units = jedi.ase_units

    # test if jedi and jediatoms indices don't intersect
    for i in jedi.indices:
        for k in jediatoms.indices:
            if i == k:
                raise TypeError("Please choose indices for the jedi and jediatoms partial analysis that don't intersect")

    # Welcome
    print("\n\nCreating pov script for generating color-coded structures in POV-Ray...")

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

    ## get rims from jedi analysis ##
    rim_list = jedi.rim_list
    if len(jedi.proc_E_RIMs) == 0:
        jedi.run()

    bl = []
    ba = []
    da = []

    # Bond lengths (a molecule has at least one bond):
    for i in rim_list[0]:
        numbers = [int(i[0]), int(i[1])]
        bl.append(numbers)

    # custom bonds
    for i in rim_list[1]:
        numbers = [int(i[0]), int(i[1])]
        bl.append(numbers)

    # Bond angles:
    for i in rim_list[2]:
        numbers = [int(i[0]), int(i[1]), int(i[2])]
        ba.append(numbers)

    # Dihedral angles:
    for i in rim_list[3]:
        numbers = [int(n) for n in i]
        da.append(numbers)

    # percental energy of RIMs
    E_RIMs = jedi.E_RIMs

    # for substructure analysis
    if len(jedi.indices) < len(jedi.atomsF):  # if there are only values for a substructure
        p_indices = jedi.indices
        jedi.indices = range(len(jedi.atomsF))

        rim = jedi.get_common_rims().copy()  # get rims of whole structure to show the whole structure

        for i in range(2):
            if rim[i].shape[0] == 0:
                break

            rim[i] = np.ascontiguousarray(rim[i])
            a = np.array(rim_list[i]).view(
                [('', np.array(rim_list[i]).dtype)] * np.array(rim_list[i]).shape[1]).ravel()
            b = np.array(rim[i]).view(
                [('', np.array(rim_list[i]).dtype)] * np.array(rim_list[i]).shape[1]).ravel()
            rim[i] = np.setxor1d(a, b)
            rim[i] = rim[i].view(np.array(rim_list[i]).dtype).reshape(-1, 2)  # get unconsidered rims
            nan = np.full((len(rim[i]), 1), np.nan)  # nan for special color (black)
            rim[i] = np.hstack((rim[i], nan))  # stack unanalyzed rims for later vmd visualization
        bond_E_array_app = rim

        jedi.indices = p_indices

    # Create an array that stores the bond connectivity as the first two entries.
    # The energy will be added as the third entry.
    E_jedi = np.full((len(bl), 3), np.nan)
    for i in range(len(bl)):
        E_jedi[i][0] = bl[i][0]
        E_jedi[i][1] = bl[i][1]

    # Bonds
    if len(bl) == 1:
        E_bl = E_RIMs
    else:
        E_bl = E_RIMs[0:len(bl)]

    # Bendings
    E_ba = E_RIMs[len(bl):len(bl) + len(ba)]

    # Torsions (handle stdout separately)
    E_da = E_RIMs[len(bl) + len(ba):len(bl) + len(ba) + len(da)]

    # Map onto the bonds (create "all" on the fly and treat diatomic molecules explicitly)
    # Bonds (trivial)
    for i in range(len(bl)):
        if len(bl) == 1:
            E_jedi[i][2] = E_bl[i]
        else:  # TODO if and else are equal?
            E_jedi[i][2] = E_bl[i]

    # Bendings
    for i in range(len(ba)):
        for j in range(len(bl)):
            # look for the right connectivity
            if ((ba[i][0] == bl[j][0] and ba[i][1] == bl[j][1])
                    or (ba[i][0] == bl[j][1] and ba[i][1] == bl[j][0])
                    or (ba[i][1] == bl[j][0] and ba[i][2] == bl[j][1])
                    or (ba[i][1] == bl[j][1] and ba[i][2] == bl[j][0])):
                E_jedi[j][2] += 0.5 * E_ba[i]
                if np.isnan(E_jedi[j][2]):
                    E_jedi[j][2] = 0.5 * E_ba[i]

    # Torsions
    for i in range(len(da)):
        for j in range(len(bl)):
            if ((da[i][0] == bl[j][0] and da[i][1] == bl[j][1])
                    or (da[i][0] == bl[j][1] and da[i][1] == bl[j][0])
                    or (da[i][1] == bl[j][0] and da[i][2] == bl[j][1])
                    or (da[i][1] == bl[j][1] and da[i][2] == bl[j][0])
                    or (da[i][2] == bl[j][0] and da[i][3] == bl[j][1])
                    or (da[i][2] == bl[j][1] and da[i][3] == bl[j][0])):
                E_jedi[j][2] += (float(1) / 3) * E_da[i]
                if np.isnan(E_jedi[j][2]):
                    E_jedi[j][2] = (float(1) / 3) * E_da[i]

    custom_E_array = E_jedi[len(rim_list[0]):len(bl)]
    bond_E_array = E_jedi[0:len(rim_list[0])]

    if len(jedi.indices) < len(jedi.atomsF):
        # stack bonds that were neglected before to show the whole structure
        bond_E_array = np.vstack((bond_E_array, bond_E_array_app[0]))
        try:
            custom_E_array = np.vstack((custom_E_array, bond_E_array_app[1]))
        except:
            pass

    bonds = bond_E_array[:, 0:2].astype(int)
    custom_bonds = custom_E_array[:, 0:2].astype(int)
    pbc_bonds = None
    custom_pbc_bonds = None
    # delete bonds that reach out of the unit cell for bonds_out_of_box==False
    if pbc_flag == True:
        pbc_bonds_mask = []
        custom_pbc_bonds_mask = []
        dF = atoms_f.get_all_distances()
        dF_mic = atoms_f.get_all_distances(mic=True)
        for i in bonds:
            if round(dF[i[0]][i[1]], 5) != round(dF_mic[i[0]][i[1]], 5):
                pbc_bonds_mask.append(False)
            else:
                pbc_bonds_mask.append(True)
        for i in custom_bonds:
            if round(dF[i[0]][i[1]], 5) != round(dF_mic[i[0]][i[1]], 5):
                custom_pbc_bonds_mask.append(False)
            else:
                custom_pbc_bonds_mask.append(True)
        pbc_bonds = bonds[~np.array(pbc_bonds_mask)]
        pbc_bond_E_array = np.delete(bond_E_array, np.where(np.array(pbc_bonds_mask))[0], axis=0)
        bonds = np.delete(bonds, np.where(~np.array(pbc_bonds_mask))[0], axis=0)
        bond_E_array = np.delete(bond_E_array, np.where(~np.array(pbc_bonds_mask))[0], axis=0)
        # custom_pbc_bonds = custom_bonds[~np.array(custom_pbc_bonds_mask)]
        # custom_pbc_bond_E_array = np.delete(custom_E_array, np.where(np.array(custom_pbc_bonds_mask))[0], axis=0)
        # custom_bonds = np.delete(custom_bonds, np.where(~np.array(custom_pbc_bonds_mask))[0], axis=0)
        # custom_E_array = np.delete(custom_E_array, np.where(~np.array(custom_pbc_bonds_mask))[0], axis=0)

    ## jediatoms ##
    E_atoms = jediatoms.E_atoms
    # Get rid of ridiculously small values
    if E_atoms.max() <= 0.001:
        E_atoms = np.zeros(len(jediatoms.indices))

    # get an E_array with only the information from coordinates of interest
    E_jediatoms = np.full((len(jediatoms.atoms0)), np.nan)
    if len(E_atoms) > len(jediatoms.indices):  # for partial_analysis or run with indices
        E_jediatoms[jediatoms.indices] = E_atoms[jediatoms.indices]
    else:
        E_jediatoms = E_atoms
    E_jediatoms = np.vstack((np.arange(len(jediatoms.atoms0)), E_jediatoms))

    # Store the maximum energy in a variable for later call
    max_energy_j = float(np.nanmax(E_jedi, axis=0)[2])  # maximum energy in one bond
    max_energy_ja = float(np.nanmax(E_jediatoms, axis=1)[1])  # maximum energy in one atom
    max_energy = float(builtins.max([max_energy_j,max_energy_ja]))
    for row in E_jedi:
        if max_energy_j in row:
            atom_1_max_energy = int(row[0])
            atom_2_max_energy = int(row[1])

    # get atom color for specific energy value from color gradient
    bond_colors = np.array([])
    for i, b in enumerate(bond_E_array):
        if np.isnan(b[2]):
            bond_color = (0.000, 0.000, 0.000)  # black
            for idx in jediatoms.indices:
                if b[0] == idx or b[1] == idx:
                    bond_color = color_unanalysed
        else:
            if man_strain is None:
                normalized_energy = b[2] / max_energy
            else:
                normalized_energy = b[2] / float(man_strain)
            color = cmap(normalized_energy)
            bond_color = (color[0], color[1], color[2])
        bond_colors = np.append(bond_colors, bond_color)
        bond_colors = bond_colors.reshape(-1, 3)

    if pbc_flag == True:
        for i, b in enumerate(pbc_bond_E_array):
            if np.isnan(b[2]):
                bond_color = (0.000, 0.000, 0.000)  # black
                for idx in jediatoms.indices:
                    if b[0] == idx or b[1] == idx:
                        bond_color = color_unanalysed
            else:
                if man_strain is None:
                    normalized_energy = b[2] / max_energy
                else:
                    normalized_energy = b[2] / float(man_strain)
                color = cmap(normalized_energy)
                bond_color = (color[0], color[1], color[2])
            bond_colors = np.append(bond_colors, bond_color)
            bond_colors = bond_colors.reshape(-1, 3)

    for i in custom_E_array:
        if np.isnan(i[2]):
            bond_color = (0.000, 0.000, 0.000)  # black
        else:
            if man_strain is None:
                normalized_energy = i[2] / max_energy
            else:
                normalized_energy = i[2] / float(man_strain)
            color = cmap(normalized_energy)
            bond_color = (color[0], color[1], color[2])
        bond_colors = np.append(bond_colors, bond_color)
        bond_colors = bond_colors.reshape(-1, 3)

    # if pbc_flag == True:
    #     for i, b in enumerate(custom_pbc_bond_E_array):
    #         if np.isnan(b[2]):
    #             bond_color = (0.000, 0.000, 0.000)  # black
    #             for idx in jediatoms.indices:
    #                 if b[0] == idx or b[1] == idx:
    #                     bond_color = color_unanalysed
    #         else:
    #             if man_strain is None:
    #                 normalized_energy = b[2] / max_energy
    #             else:
    #                 normalized_energy = b[2] / float(man_strain)
    #             color = cmap(normalized_energy)
    #             bond_color = (color[0], color[1], color[2])
    #         bond_colors = np.append(bond_colors, bond_color)
    #         bond_colors = bond_colors.reshape(-1, 3)

    ## atom colors ##
    # get atom color for specific energy value from color gradient
    atom_colors = np.array([])
    for i, b in zip(E_jediatoms[0], E_jediatoms[1]):
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

    for i in jedi.indices:
        atom_colors[i] = color_unanalysed

    # radii to distinguish different elements
    atomic_nrs = atoms_f.get_atomic_numbers()
    z = np.array(list(set(atomic_nrs)))
    atom_radii = None
    legend = False
    if len(z) > 1:
        legend = True
        atom_radii = {i: covalent_radii[i] for i in z}
        atom_radii = dict(sorted(atom_radii.items(), key=lambda item: item[1]))
        radii_values = np.linspace(0.3, 0.8, len(z))
        atom_radii = {k: v for k, v in zip(atom_radii.keys(), radii_values)}
        radii = np.array([atom_radii[num] for num in atomic_nrs])

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
                  atom_colors=atom_colors,
                  bond_colors=bond_colors,
                  cameralocation=location,
                  look_at=center,
                  camera_right_up=camwidth,
                  cameradirection=direction,
                  area_light=[light, 'White', 1.7, 1.7, 3, 3],
                  background=background,
                  bondatoms=bonds,
                  pbc_bondatoms=pbc_bonds,
                  custom_bondatoms=custom_bonds,
                  custom_pbc_bondatoms=custom_pbc_bonds,
                  metal=metal,
                  bondradius=bondradius,
                  pixelwidth=pixelwidth,
                  aspectratio=aspectratio,
                  cell=cell,
                  legend=legend,
                  legend_atoms=atom_radii,
                  legend_color=color_unanalysed
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
                          atom_colors=atom_colors,
                          bond_colors=bond_colors,
                          cameralocation=location,
                          look_at=center,
                          camera_right_up=camwidth,
                          cameradirection=direction,
                          area_light=[light, 'White', 1.7, 1.7, 3, 3],
                          background=background,
                          bondatoms=bonds,
                          pbc_bondatoms=pbc_bonds,
                          custom_bondatoms=custom_bonds,
                          custom_pbc_bondatoms=custom_pbc_bonds,
                          metal=metal,
                          bondradius=bondradius,
                          pixelwidth=pixelwidth,
                          aspectratio=aspectratio,
                          cell=cell,
                          legend=legend,
                          legend_atoms=atom_radii,
                          legend_color=color_unanalysed
                          )
                if run_pov is True:
                    pov.write(f'{label}.png', label)
                else:
                    pov.write(f'{label}.pov', label)

    if ase_units == False:
        unit = "kcal/mol"
    elif ase_units == True:
        unit = "eV"

    # colorbar
    if colorbar is True:
        min = 0.000

        if man_strain is None:
            max = max_energy
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
        fig.savefig(destination_dir / 'colorbar.pdf', bbox_inches='tight')

    print("\nAdding all energies for the stretch, bending and torsion of the bond with maximum strain...")
    if man_strain is None:
        print(f"Maximum energy in bond between atoms "
              f"{atom_1_max_energy} and {atom_2_max_energy}: {float(max_energy_j):.3f} {unit}.")
        print(f"Maximum energy in  atom {int(np.nanargmax(E_jediatoms[0],axis=0))}: {float(max_energy_ja):.3f} {unit}.")

    # printout(jedi.E_geometries,jedi.E_RIMs_total,jediatoms.E_atoms_total,jedi.proc_geom_RIMs,jediatoms.proc_geom_atoms)
             # jedi.rim_list,jedi.atomsF,jedi.proc_E_RIMs,jedi.E_RIMs,)

# def printout(E_geometries: float,
#              E_RIMs_total: float,
#              E_atoms_total: float,
#              proc_geom_RIMs: float,
#              proc_geom_atoms: float,
#              ase_units: bool = False,
#              file: str = 'E_hybrid'):
#     '''
#     Printout of analysis of stored strain energy in the bonds and atoms.
#     '''
#     output = []
#     # Header
#     output.append("\n")
#     output.append('{:^{header}}'.format("************************************************", **header))
#     output.append('{:^{header}}'.format("*                 JEDI ANALYSIS                *", **header))
#     output.append('{:^{header}}'.format("*       Judgement of Energy DIstribution       *", **header))
#     output.append('{:^{header}}'.format("************************************************", **header))
#     output.append('{:^{header}}'.format(f"version {__version__}\n", **header))
#
#     # Comparison of total energies
#     if not ase_units:
#         output.append('{0:>{column1}}''{1:^{column2}}''{2:^{column3}}'
#                       .format(" ", "Strain Energy (kcal/mol)", "Deviation (%)", **energy_comparison))
#     elif ase_units:
#         output.append('{0:>{column1}}''{1:^{column2}}''{2:^{column3}}'
#                       .format(" ", "Strain Energy (eV)", "Deviation (%)", **energy_comparison))
#
#     output.append('{0:<{column1}}' '{1:^{column2}.4f}' '{2:^{column3}}'
#                   .format("Ab Initio", E_geometries, "-", **energy_comparison))
#     # TODO save proc_e_rims as std decimal number and have print command use .2%
#     output.append('{0:<{column1}}''{1:^{column2}.4f}''{2:^{column3}.2f}'
#                   .format("JEDI", E_RIMs_total, proc_geom_RIMs, **energy_comparison))
#     output.append('{0:<{column1}}''{1:^{column2}.4f}''{2:^{column3}.2f}'
#                   .format(f"JEDI_ATOMS", E_atoms_total, proc_geom_atoms, **energy_comparison))
#     output.append('{0:<{column1}}''{1:^{column2}.4f}''{2:^{column3}.2f}'
#                   .format("Hybrid", E_RIMs_total+E_atoms_total, proc_geom_RIMs+proc_geom_atoms, **energy_comparison))

    # # strain in the bonds
    # if not ase_units:
    #     output.append(
    #         '{0:^{column1}}''{1:^{column2}}''{2:^{column3}}''{3:^{column4}}''{4:^{column5}}'
    #         .format("RIC No.", "RIC type", "indices", "Percentage", "Energy (kcal/mol)", **rims_listing))
    # elif ase_units:
    #     output.append(
    #         '{0:^{column1}}''{1:^{column2}}''{2:^{column3}}''{3:^{column4}}''{4:^{column5}}'
    #         .format("RIC No.", "RIC type", "indices", "Percentage", "Energy (eV)", **rims_listing))
    # rics_dict = {0: "bond",
    #              1: "custom"}
    # ric_counter = 0
    # for ric_type, rim in rics_dict.items():
    #     for k in rim_list[ric_type]:
    #         ind = f"{atoms.symbols[k[0]]}{k[0]}  {atoms.symbols[k[1]]}{k[1]}"
    #         # TODO save proc_e_rims as std decimal number and have print command use .2%
    #         output.append(
    #             '{0:^{column1}}''{1:^{column2}}''{2:^{column3}}''{3:^{column4}.2f}''{4:^{column5}.4f}'
    #             .format(ric_counter + 1,
    #                     rim,
    #                     ind,
    #                     proc_E_RIMs[ric_counter],
    #                     E_RIMs[ric_counter],
    #                     **rims_listing))
    #         ric_counter += 1
    #
    # # strain in the atoms
    # if not ase_units:
    #     output.append(
    #         '{0:^{column1}}''{1:^{column2}}''{2:^{column3}}''{3:^{column4}}'
    #         .format("Atom No.", "Element", "Percentage", "Energy (kcal/mol)", **atoms_listing))
    # elif ase_units:
    #     output.append(
    #         '{0:^{column1}}''{1:^{column2}}''{2:^{column3}}''{3:^{column4}}'
    #         .format("Atom No.", "Element", "Percentage", "Energy (eV)", **atoms_listing))
    # for i, k in enumerate(E_atoms[indices]):
    #     output.append(
    #         '{0:^{column1}}''{1:^{column2}}''{2:^{column3}.2f}''{3:^{column4}.2f}'
    #         .format(self.indices[i],
    #                 self.atoms0.symbols[self.indices[i]],
    #                 k/E_atoms_total*100,
    #                 k,
    #                 **atoms_listing))

    # with open(file, 'w') as f:
    #     f.writelines("\n".join(output))
