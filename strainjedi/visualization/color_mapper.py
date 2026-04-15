import warnings

import ase.neighborlist
import numpy as np
from ase.atoms import Atom
from ase.data.vdw import vdw_radii
from matplotlib import cm, colormaps
from matplotlib.colors import Colormap

from strainjedi.visualization.colors import colors as symbol_colors


class ColorMapper:
    def __init__(self, jedi_instance):
        self.j = jedi_instance
        self.atoms0 = jedi_instance.atoms0
        self.atomsF = jedi_instance.atomsF
        self.rim_list = jedi_instance.rim_list
        self.E_RIMs = jedi_instance.E_RIMs
        self.proc_E_RIMs = jedi_instance.proc_E_RIMs
        self.custom_bonds = jedi_instance.custom_bonds
        self.ase_units = jedi_instance.ase_units
        self.indices = jedi_instance.indices
        self.vdwf = jedi_instance.vdwf
        self.atoms_vis = jedi_instance.atomsF

    def assign_atom_colors(self):
        atom_colors = []
        for atom in self.atoms_vis:
            atom_colors.append(symbol_colors.get(atom.symbol))
        return atom_colors

    def map_to_bonds(self, mode):

        self.bl, self.ba, self.da = [], [], []

        for i in self.rim_list[0]:
            self.bl.append([int(i[0]), int(i[1])])

        for i in self.rim_list[1]:
            self.bl.append([int(i[0]), int(i[1])])

        for i in self.rim_list[2]:
            self.ba.append([int(i[0]), int(i[1]), int(i[2])])

        for i in self.rim_list[3]:
            self.da.append([int(i[0]), int(i[1]), int(i[2]), int(i[3])])

        E_array = np.full((len(self.bl), 3), np.nan)
        for i in range(len(self.bl)):
            E_array[i][0] = self.bl[i][0]
            E_array[i][1] = self.bl[i][1]

        E_array = self.get_energies_per_bond(E_array, mode)
        E_array[np.isnan(E_array[:, 2]), 2] = 0.0

        n_regular = len(self.rim_list[0])
        n_custom = len(self.rim_list[1])

        if self.atomsF.pbc.any():
            E_array, n_regular, n_custom = self.handle_pbc(E_array, n_regular, n_custom)

        if len(self.indices) < len(self.atomsF):
            E_array = self.add_unanalyzed_bonds(E_array)

        regular_bonds = E_array[:n_regular]
        custom_bonds_section = E_array[n_regular : n_regular + n_custom]
        pbc_bonds = E_array[n_regular + n_custom :]

        all_regular = (
            np.vstack([regular_bonds, pbc_bonds])
            if len(pbc_bonds) > 0
            else regular_bonds
        )

        return {
            "bonds": all_regular[:, :2].astype(int),
            "energies": all_regular[:, 2],
            "custom_bonds": custom_bonds_section[:, :2].astype(int)
            if len(custom_bonds_section) > 0
            else np.empty((0, 2), dtype=int),
            "custom_energies": custom_bonds_section[:, 2]
            if len(custom_bonds_section) > 0
            else np.array([]),
            "atoms": self.atoms_vis,
            "split_bonds": getattr(self, "split_bonds", False),
            "pbc_split_bonds": getattr(self, "pbc_split_bonds", []),
        }

    def get_energies_per_bond(self, E_array, mode):
        if mode in ["bl", "all"]:
            E_bl = self.E_RIMs[0 : len(self.bl)]
            for i in range(len(E_bl)):
                E_array[i][2] = E_bl[i]

        if mode in ["ba", "all"]:
            E_ba = self.E_RIMs[len(self.bl) : (len(self.bl) + len(self.ba))]
            for i, angle in enumerate(self.ba):
                for j, bond in enumerate(self.bl):
                    if self.check_bond_in_angle(bond, angle):
                        if np.isnan(E_array[j][2]):
                            E_array[j][2] = 0.5 * E_ba[i]
                        else:
                            E_array[j][2] += 0.5 * E_ba[i]

        if mode in ["da", "all"]:
            E_da = self.E_RIMs[(len(self.bl) + len(self.ba)) : len(self.E_RIMs)]
            for i, dihedral in enumerate(self.da):
                for j, bond in enumerate(self.bl):
                    if self.check_bond_in_dihedral(bond, dihedral):
                        if np.isnan(E_array[j][2]):
                            E_array[j][2] = (float(1) / 3) * E_da[i]
                        else:
                            E_array[j][2] += (float(1) / 3) * E_da[i]

        return E_array

    def check_bond_in_angle(self, bond, angle):
        return (
            (angle[0] == bond[0] and angle[1] == bond[1])
            or (angle[0] == bond[1] and angle[1] == bond[0])
            or (angle[1] == bond[0] and angle[2] == bond[1])
            or (angle[1] == bond[1] and angle[2] == bond[0])
        )

    def check_bond_in_dihedral(self, bond, dihedral):
        return (
            (dihedral[0] == bond[0] and dihedral[1] == bond[1])
            or (dihedral[0] == bond[1] and dihedral[1] == bond[0])
            or (dihedral[1] == bond[0] and dihedral[2] == bond[1])
            or (dihedral[1] == bond[1] and dihedral[2] == bond[0])
            or (dihedral[2] == bond[0] and dihedral[3] == bond[1])
            or (dihedral[2] == bond[1] and dihedral[3] == bond[0])
        )

    def add_unanalyzed_bonds(self, E_array):

        self.j.indices = np.arange(0, len(self.atomsF))
        full_rims = self.j.get_rims(self.atomsF)
        analyzed_bonds = set(
            tuple(sorted([int(b[0]), int(b[1])])) for b in E_array[:, :2]
        )

        unanalyzed = []
        for bond in full_rims[0]:
            bond_tuple = tuple(sorted([int(bond[0]), int(bond[1])]))
            if bond_tuple not in analyzed_bonds:
                unanalyzed.append([bond[0], bond[1], np.nan])

        if len(unanalyzed) > 0:
            E_array = np.vstack((E_array, unanalyzed))

        return E_array

    def handle_pbc(self, E_array, n_regular, n_custom):

        mol = self.atomsF.copy()
        for atom in mol:
            atom.tag = -1

        rim_list = self.rim_list
        n_orig = len(self.atomsF)

        bondscheck = (
            rim_list[0][:, (0, 1)] if n_regular > 0 else np.empty((0, 2), dtype=int)
        )
        customcheck = (
            rim_list[1][:, (0, 1)] if n_custom > 0 else np.empty((0, 2), dtype=int)
        )

        cutoff = [vdw_radii[atom.number] * self.vdwf for atom in self.atomsF]

        # Get all neighbor pairs with shift and displacement vectors
        ex_bl = np.vstack(
            ase.neighborlist.neighbor_list("ij", a=self.atomsF, cutoff=cutoff)
        ).T
        ex_bl = np.hstack(
            (ex_bl, ase.neighborlist.neighbor_list("S", a=self.atomsF, cutoff=cutoff))
        )
        ex_bl = np.hstack(
            (ex_bl, ase.neighborlist.neighbor_list("D", a=self.atomsF, cutoff=cutoff))
        )

        atoms_ex_cell = ex_bl[
            (ex_bl[:, 2] != 0) | (ex_bl[:, 3] != 0) | (ex_bl[:, 4] != 0)
        ]

        # Build energy lookups
        translate = {}
        for row in E_array[:n_regular]:
            key = (int(min(row[0], row[1])), int(max(row[0], row[1])))
            translate[key] = row[2]

        ctranslate = {}
        for row in E_array[n_regular : n_regular + n_custom]:
            key = (int(min(row[0], row[1])), int(max(row[0], row[1])))
            ctranslate[key] = row[2]

        bond_pbc = []
        custom_pbc = []
        regular_to_remove = set()
        custom_to_remove = set()
        pbc_split_bonds = []

        for i in range(len(atoms_ex_cell)):
            atom_i = int(atoms_ex_cell[i, 0])
            atom_j = int(atoms_ex_cell[i, 1])
            disp = atoms_ex_cell[i, 5:8]

            # Aux position: wrapped pos of atom_i + displacement vector
            pos_ex_atom = mol.get_positions()[atom_i] + disp

            original_rim = sorted([atom_i, atom_j])
            original_tuple = tuple(original_rim)

            is_regular = (
                n_regular > 0
                and len(np.where(np.all(original_rim == bondscheck, axis=1))[0]) > 0
            )
            is_custom = (
                n_custom > 0
                and len(np.where(np.all(original_rim == customcheck, axis=1))[0]) > 0
            )

            if not is_regular and not is_custom:
                continue

            if self.split_bonds:
                if is_regular:
                    energy = translate.get(original_tuple, np.nan)
                    pbc_split_bonds.append(
                        {
                            "atom_index": atom_i,
                            "dummy_position": pos_ex_atom,
                            "energy": energy,
                            "original_bond": original_tuple,
                        }
                    )
                    regular_to_remove.add(original_tuple)

                elif is_custom:
                    energy = translate.get(original_tuple, np.nan)
                    pbc_split_bonds.append(
                        {
                            "atom_index": atom_i,
                            "dummy_position": pos_ex_atom,
                            "energy": energy,
                            "original_bond": original_tuple,
                            "is_custom": True,
                        }
                    )
                    custom_to_remove.add(original_tuple)

            else:
                # Only reuse aux atoms (index >= n_orig), never originals
                existing_aux = []
                for idx in range(n_orig, len(mol)):
                    if np.allclose(mol.positions[idx], pos_ex_atom, atol=1e-6):
                        existing_aux.append(idx)

                if len(existing_aux) > 0:
                    ex_ind = existing_aux[0]
                else:
                    ex_ind = len(mol)
                    mol.append(
                        Atom(
                            symbol=mol.symbols[atom_j], position=pos_ex_atom, tag=atom_j
                        )
                    )

                if is_regular:
                    energy = translate.get(original_tuple, np.nan)
                    bond_pbc.append([atom_i, ex_ind, energy])
                    regular_to_remove.add(original_tuple)

                elif is_custom:
                    energy = ctranslate.get(original_tuple, np.nan)
                    custom_pbc.append([atom_i, ex_ind, energy])
                    custom_to_remove.add(original_tuple)

        # Remove original cross-cell regular bonds
        regular_section = E_array[:n_regular]
        if regular_to_remove:
            keep_mask = np.ones(len(regular_section), dtype=bool)
            for idx, row in enumerate(regular_section):
                key = (int(min(row[0], row[1])), int(max(row[0], row[1])))
                if key in regular_to_remove:
                    keep_mask[idx] = False
            regular_section = regular_section[keep_mask]
        n_regular = len(regular_section)

        # Remove original cross-cell custom bonds
        orig_n_regular = len(self.rim_list[0])
        custom_section = E_array[orig_n_regular : orig_n_regular + n_custom]
        if custom_to_remove and n_custom > 0:
            keep_mask = np.ones(len(custom_section), dtype=bool)
            for idx, row in enumerate(custom_section):
                key = (int(min(row[0], row[1])), int(max(row[0], row[1])))
                if key in custom_to_remove:
                    keep_mask[idx] = False
            custom_section = custom_section[keep_mask]
        if not self.split_bonds and custom_pbc:
            custom_section = (
                np.vstack([custom_section, np.array(custom_pbc)])
                if len(custom_section) > 0
                else np.array(custom_pbc)
            )
        n_custom = len(custom_section)

        # Rebuild E_array
        parts = [regular_section]
        if n_custom > 0:
            parts.append(custom_section)
        if bond_pbc:
            parts.append(np.array(bond_pbc))
        E_array = np.vstack(parts) if len(parts) > 1 else parts[0]

        self.atoms_vis = mol
        self.pbc_split_bonds = pbc_split_bonds if self.split_bonds else []

        return E_array, n_regular, n_custom

    def generate_colors(self, colormap, n_colors):
        colors = []
        gradient = np.linspace(0, 1, n_colors)

        if isinstance(colormap, Colormap):
            colors = colormap(gradient)[:, :3].tolist()

        elif colormap == "green_red":
            for i in range(n_colors):
                R = min(1.0, float(i) / (n_colors / 2))
                G = (
                    min(1.0, 2 - float(i + 1) / (n_colors / 2))
                    if n_colors % 2 == 0
                    else min(1.0, 2 - float(i) / (n_colors / 2))
                )
                colors.append([R, G, 0.0])

        elif colormap == "cyan_red":
            for i in range(n_colors):
                R = min(1.0, float(i) / (n_colors / 2))
                B = min(1.0, 2 - float(i + 1) / (n_colors / 2))
                G = (
                    ((n_colors / 2) - i) / (n_colors / 2)
                    if i <= (n_colors / 2)
                    else 0.0
                )
                colors.append([R, G, B])

        elif colormap == "magma":
            cmap = cm.get_cmap("magma").reversed()
            adjusted = gradient * (1 - 0.175 - 0.15) + 0.15
            colors = cmap(adjusted)[:, :3].tolist()

        elif colormap in list(colormaps):
            colors = cm.get_cmap(colormap)(gradient)[:, :3].tolist()

        else:
            raise ValueError(
                f"Unknown color scheme: {colormap}. "
                f"Pass a string or a matplotlib Colormap object."
            )

        return np.array(colors)

    def get_visualization_data(
        self, modes, colormap="green_red", man_strain=None, split_bonds=True
    ):
        visualization_data = {}
        self.split_bonds = split_bonds

        for m in modes:
            bond_data = self.map_to_bonds(m)
            atom_colors = self.assign_atom_colors()

            all_energies = np.concatenate(
                [bond_data["energies"], bond_data["custom_energies"]]
            )
            max_strain = man_strain if man_strain else np.nanmax(all_energies)

            if max_strain == 0 or np.isnan(max_strain) or max_strain is None:
                warnings.warn(
                    f"Mode '{m}' has no strain energy (max_strain = {max_strain})",
                    UserWarning,
                )
                max_strain = 1.0  # dummy value to avoid division by zero

            bond_data["norm_energies"] = np.clip(
                bond_data["energies"] / max_strain, 0, 1
            )
            bond_data["norm_custom_energies"] = np.clip(
                bond_data["custom_energies"] / max_strain, 0, 1
            )

            for pbc_bond in bond_data["pbc_split_bonds"]:
                norm_energy = np.clip(pbc_bond["energy"] / max_strain, 0, 1)
                pbc_bond["norm_energy"] = norm_energy

            color_data = {
                "atom_colors": atom_colors,
                "max_strain": max_strain,
                "colormap": colormap,
            }

            visualization_data[m] = {"bond_data": bond_data, "color_data": color_data}

        return visualization_data
