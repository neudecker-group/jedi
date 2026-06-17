import warnings
from pathlib import Path

import ase.geometry
import ase.neighborlist
import ase.units
import ase.vibrations
import numpy as np
from ase.data.vdw import vdw_radii
from ase.utils import jsonable
from numpy.typing import NDArray
from typing_extensions import Any, deprecated

from strainjedi import bmatrix, constants, reporting, rics, utils
from strainjedi.visualization import ColorMapper, MatplotlibVisualizer, VMDVisualizer


@jsonable("jedi")
class Jedi:
    def __init__(
        self,
        atoms0: ase.atoms.Atoms,
        atomsF: ase.atoms.Atoms,
        modes: ase.vibrations.data.VibrationsData,
        epot: NDArray | None = None,
    ):  # indices=None
        """
        atoms0: class
            Atoms object of relaxed structure with calculated energy.
        atomsF: class
            Atoms object of strained structure with calculated energy.
        modes: class
            VibrationsData object with hessian of relaxed structure.
        epot: np.array or None
            Vector containing (f - i) endiff., final, initial energy or None. Default: None.
        """
        # validate the Hessian's atom order (must match atoms0); if not, permute Hessian to match.
        hessian, ok = utils.validate_hessian(modes, atoms0)
        if not ok:
            warnings.warn(
                "Atoms in VibrationsData object were not fitting atoms0. "
                "I have reordered your Hessian to match, but you may wish to check it."
            )

        # Internal state (name-mangled attributes only)
        self.__H = hessian / (ase.units.Hartree / ase.units.Bohr**2)

        self.__atoms0 = atoms0  # ref state
        self.__atomsF = atomsF  # strained state
        self.__covf = constants.COVALENCY_FACTOR
        self.__vdwf = constants.VAN_DER_WAALS_FACTOR

        self.__indices = np.arange(0, len(self.__atoms0))
        self.__custom_bonds = None  # list of custom added bonds

        self.__rim_list = rics.intersect(
            rics.calculate(self.__atoms0, self.__indices, self.__custom_bonds),
            rics.calculate(self.__atomsF, self.__indices, self.__custom_bonds),
        )

        self.__B = bmatrix.calculate(self.__atoms0, self.__rim_list)
        self.__q0, self.__qF, self.__delta_q = rics.subtract(self.__atoms0, self.__atomsF, self.__rim_list)

        self.__energies = epot  # optional energies of the geometries (legacy)
        self.__deltaE: float = 0.0  # energy difference between geometries

        self.__proc_E_RIMs = None  # list of procentual energy stored in single RIMs
        self.__part_rim_list = None  # rim list for election of atoms (legacy/unused)
        self.__E_RIMs = None  # list of energies stored in the rims
        self.__E_RIMs_total = None  # sum of E_rims

        self.__visualization_data = None  # last visualization dataset

    @property
    def H(self):
        return self.__H

    @property
    def atoms0(self):
        return self.__atoms0

    @property
    def atomsF(self):
        return self.__atomsF

    @property
    def rics0(self):
        return rics.calculate(self.__atoms0, self.__indices, self.__custom_bonds)

    @property
    def ricsF(self):
        return rics.calculate(self.__atomsF, self.__indices, self.__custom_bonds)

    @property
    def covf(self):
        return self.__covf

    @property
    def vdwf(self):
        return self.__vdwf

    @property
    def indices(self):
        return self.__indices

    @property
    def custom_bonds(self):
        return self.__custom_bonds

    @property
    def rim_list(self):
        return self.__rim_list

    @property
    def B(self):
        return self.__B

    @property
    def q0(self):
        return self.__q0

    @property
    def qF(self):
        return self.__qF

    @property
    def delta_q(self):
        return self.__delta_q

    @property
    def energies(self):
        return self.__energies

    @property
    def deltaE(self):
        return self.__deltaE

    @property
    def proc_E_RIMs(self):
        return self.__proc_E_RIMs

    @property
    def part_rim_list(self):
        return self.__part_rim_list

    @property
    def E_RIMs(self):
        return self.__E_RIMs

    @property
    def E_RIMs_total(self):
        return self.__E_RIMs_total

    @property
    def visualization_data(self):
        return self.__visualization_data

    def todict(self) -> dict[str, Any]:
        """make it saveable with .write()"""
        return {
            "atoms0": self.__atoms0,
            "atomsF": self.__atomsF,
            "hessian": self.__H,
            "bmatrix": self.__B,
            "delta_q": self.__delta_q,
            "rim_list": self.__rim_list,
            "energies": self.__energies,
            "indices": self.__indices,
            "E_RIMS": self.__E_RIMs,
            "proc_E_RIMS": self.__proc_E_RIMs,
            "custom_bonds": self.__custom_bonds,
        }

    @classmethod
    def fromdict(cls, data: dict[str, Any]) -> "Jedi":
        """
        Reconstruct a Jedi object from a dictionary.
        Raises ValueError if required fields are missing or have incorrect types.
        """
        atoms0 = data["atoms0"]
        atomsF = data["atomsF"]

        # Try to reconstruct VibrationsData if possible
        if "modes" in data and isinstance(data["modes"], ase.vibrations.VibrationsData):
            modes = data["modes"]
        elif "hessian" in data:
            indices = data.get("indices", None)
            if indices is not None:
                modes = ase.vibrations.VibrationsData.from_2d(atoms0, data["hessian"], indices)
            else:
                modes = ase.vibrations.VibrationsData.from_2d(atoms0, data["hessian"])
        else:
            raise ValueError("No valid vibration data ('modes' or 'hessian') found in the dictionary!")

        epot = data.get("energies", None)
        cl = cls(atoms0, atomsF, modes, epot=epot)

        # Fill additional attributes if present
        for attr, key in [
            ("_Jedi__H", "hessian"),
            ("_Jedi__B", "bmatrix"),
            ("_Jedi__delta_q", "delta_q"),
            ("_Jedi__rim_list", "rim_list"),
            ("_Jedi__indices", "indices"),
            ("_Jedi__E_RIMs", "E_RIMS"),
            ("_Jedi__proc_E_RIMs", "proc_E_RIMS"),
            ("_Jedi__custom_bonds", "custom_bonds"),
            ("_Jedi__energies", "energies"),
        ]:
            if key in data and data[key] is not None:
                setattr(cl, attr, data[key])

        return cl

    def run(self, indices=None, ase_units=False, printout: bool = True):
        """Runs the analysis. Calls all necessary functions to get the needed values.

        Args:
            indices:
                list of indices of a substructure if desired
            ase_units: boolean
                flag to get eV for energies å fo lengths and degree for angles otherwise it is kcal/mol, Bohr and radians
        Returns:
            Indices, strain, energy in every RIM
        """

        # get necessary data
        if len(self.__atoms0) != self.__H.shape[0] / 3:
            raise ValueError(
                "Hessian has not the fitting shape, possibly a partial hessian. Please try partial_analysis"
            )

        self.__deltaE = self.get_energies()

        (
            self.__proc_E_RIMs,
            self.__E_RIMs,
            self.__E_RIMs_total,
            proc_geom_RIMs,
            self.__delta_q,
        ) = jedi_analysis(
            self.__atoms0,
            self.__rim_list,
            self.__B,
            self.__H,
            self.__delta_q,
            self.__deltaE,
            use_ase_units=ase_units,
        )

        if indices:  # get only rims of interest
            self.post_process(indices)
            self.__E_RIMs_total = float(np.sum(self.__E_RIMs))
            proc_geom_RIMs = 100 * (float(np.sum(self.__E_RIMs)) - self.__deltaE) / self.__deltaE

        if printout:
            reporting.jedi_printout(
                self.__atoms0,
                self.__rim_list,
                self.__delta_q,
                self.__deltaE,
                self.__E_RIMs_total,
                proc_geom_RIMs,
                self.__proc_E_RIMs,
                self.__E_RIMs,
                ase_units=ase_units,
            )

    def get_energies(self) -> float:
        """
        Returns the difference in potential energies of self.atoms0 and self.atomsF in kcal/mol.
        """
        e0 = self.__atoms0.get_potential_energy()  # [eV]
        eF = self.__atomsF.get_potential_energy()  # [eV]
        return eF - e0

    def visualize(
        self,
        visualizer="mpl",
        colormap="green_red",
        output_dir: Path | str = "visualization",
        single_mode: str | None = None,
        man_strain: float | None = None,
        show: bool = False,
        show_indices: bool = False,
        box: bool = False,
        split_bonds: bool = True,
        ase_units: bool = False,
    ):
        """
        Args:
            visualizer: ('mpl' or 'vmd')
                Defines visualizer used for visualization.
            colormap: (str or matplotlib Colormap object)
                Color scheme for strain energy mapping. Built-in options are
                'green_red', 'cyan_red', 'magma', or any matplotlib colormap
                name (e.g. 'viridis', 'inferno'). Alternatively, a custom
                matplotlib Colormap object can be passed directly.
                default: 'green_red'
            output_dir: (str or Path)
                Directory to save the visualization output files.
                default: 'visualization'
            single_mode: (str or None)
                Restrict visualization to a single mode. Options are 'bl'
                (bond lengths), 'ba' (bond angles), 'da' (dihedral angles),
                or 'all' (combined). If None, all modes are generated.
                default: None
            man_strain: (float or None)
                Manual reference value for the maximum of the color scale.
                If None, the maximum strain energy in the data is used.
                default: None
            show: (bool)
                Display the plot interactively. Only works with the
                Matplotlib visualizer.
                default: False
            show_indices: (bool)
                Display atom indices. Only works with the
                Matplotlib visualizer.
            split_bonds: (bool)
                Display PBC bonds as half-bonds at each side of the unit cell.
            box: (bool)
                Draw the unit cell box. Only applies to periodic structures.
                default: False
            ase_units: (bool)
                Whether to use Angstrom and eV instead of kcal/mol and Bohr.
                default: False
        """
        if self.__proc_E_RIMs is None or len(self.__proc_E_RIMs) == 0:
            raise ValueError("Analysis has not been run. Jedi.run() must be called before Jedi.visualize()")

        if show and visualizer != "mpl":
            warnings.warn("'show=True' only works for the Matplotlib Visualizer and will be ignored.", UserWarning)

        if show_indices and visualizer != "mpl":
            warnings.warn(
                "'show_indices=True' only works for the Matplotlib Visualizer and will be ignored.",
                UserWarning,
            )

        energy_unit = "eV" if ase_units else "kcal/mol"

        valid_modes = ["bl", "ba", "da", "all"]
        if not single_mode:
            mode_list = valid_modes
        elif single_mode in valid_modes:
            mode_list = [single_mode]
        else:
            raise ValueError(f"Unknown mode '{single_mode}'. single_mode must be in: {valid_modes} or None")

        mapper = ColorMapper(self, ase_units)
        self.__visualization_data = mapper.get_visualization_data(mode_list, colormap, man_strain, split_bonds)

        if visualizer == "mpl":
            vis = MatplotlibVisualizer(self.__visualization_data, mapper, output_dir, energy_unit)
            vis.run(show, show_indices, box)
        elif visualizer == "vmd":
            vis = VMDVisualizer(self.__visualization_data, mapper, output_dir, energy_unit)
            vis.write_inputs(box)
        else:
            raise ValueError("Unknown visualizer. Visualizer must be 'mpl' or 'vmd'.")

    @deprecated("Use Jedi.visualize(visualizer='vmd') instead.")
    def vmd_gen(
        self,
        _des_colors: dict | None = None,
        _box: bool = False,
        man_strain: float | None = None,
        modus: str | None = None,
        _colorbar: bool = True,
        label: Path | str = "vmd",
        incl_coloring: str | None = None,
    ):
        """
        Args:
            des_colors: (dict)
                key: order number, value: [R,G,B]
            box: boolean
                True: draw box
                False: ignore box
            man_strain: float
                reference value for the strain energy used in the color scale
                default: 'None'
            modus: str
                defines where to use the man_strain
                default: 'None'
            colorbar: boolean
                draw colorbar or not
            label: string or pathlib.Path
                name of folder for the created files
            incl_coloring: str
                2 inclusive coloring options, otherwise green to red gradient
                "cyan": cyan to red gradient
                "magma": matplotlib magma gradient
                default: 'None'
        """
        if not incl_coloring:
            incl_coloring = "green_red"
        self.visualize(
            visualizer="vmd",
            colormap=incl_coloring,
            output_dir=label,
            single_mode=modus,
            man_strain=man_strain,
        )

    def partial_analysis(self, indices, ase_units=False):
        """
        Analyse a substructure with given indices.

        Args:
            indices:
                list of indices of atoms in desired substructure
        """
        # for calculation with partial hessian
        if 3 * len(indices) < len(self.__H):
            raise ValueError("to little indices for the given hessian")

        has_custom_bonds = False
        if self.__custom_bonds is not None:
            old_custom_bonds = self.__custom_bonds.copy()
            has_custom_bonds = True
            self.__custom_bonds = self.__custom_bonds[np.isin(self.__custom_bonds, indices).all(axis=1)]

        rim_list = rics.calculate(self.__atoms0, self.__indices, self.__custom_bonds)
        if len(rim_list) == 0:
            raise ValueError("Chosen indexlist has no rims")

        self.__B = bmatrix.calculate(self.__atoms0, rim_list, indices=self.__indices)
        # set B matrix values of not considered atoms to 0
        self.__B = bmatrix.restrict(self.__B, indices)

        self.__q0, self.__qF, self.__delta_q = rics.subtract(self.__atoms0, self.__atomsF, rim_list)

        self.__deltaE = self.get_energies()

        (
            self.__proc_E_RIMs,
            self.__E_RIMs,
            self.__E_RIMs_total,
            _proc_geom_RIMs,
            self.__delta_q,
        ) = jedi_analysis(
            self.__atomsF,
            rim_list,
            self.__B,
            self.__H,
            self.__delta_q,
            self.__deltaE,
            use_ase_units=ase_units,
        )

        # get values of rims inside the substructure
        self.post_process(indices)
        self.__E_RIMs_total = float(np.sum(self.__E_RIMs))
        proc_geom_RIMs = 100 * (float(np.sum(self.__E_RIMs)) - self.__deltaE) / self.__deltaE

        reporting.jedi_printout(
            self.__atoms0,
            self.__rim_list,
            self.__delta_q,
            self.__deltaE,
            self.__E_RIMs_total,
            proc_geom_RIMs,
            self.__proc_E_RIMs,
            self.__E_RIMs,
            ase_units=ase_units,
        )

        if has_custom_bonds:
            self.__custom_bonds = old_custom_bonds  # restore the user input

    def post_process(self, indices):
        """
        Get only the values of RICs inside a defined substructure.

        Args:
            indices:
                list of indices of atoms in desired substructure
        Returns:
            Values for analyzed RIMs in the defined substructure
        """
        # get rims with only the considered atoms
        self.__indices = indices
        rim_list = self.__rim_list

        cbonds_flag = False
        if self.__custom_bonds is not None:
            custom_bonds = self.__custom_bonds.copy()
            cbonds_flag = True
            self.__custom_bonds = self.__custom_bonds[np.isin(self.__custom_bonds, indices).all(axis=1)]

        rim_p = rics.intersect(
            rics.calculate(self.__atoms0, indices, self.__custom_bonds),
            rics.calculate(self.__atomsF, indices, self.__custom_bonds),
        )

        ind = []
        rim_list_c = []  # preparing for stacking rim_list to be able to use np.unique

        for i in range(4):  # rim_list is always of length 4
            if rim_list[i].shape == (0,):
                rim_list_c.append([])
            else:
                if rim_p[i].shape[0] > 0:
                    rim_list_c.append(np.vstack((rim_list[i], rim_p[i])))
                else:
                    rim_list_c.append(rim_list[i])

            _, z = np.unique(rim_list_c[-1], return_counts=True, axis=0)
            ind.append(np.where(z > 1)[0])  # get indices where ric is in both sets

        for i in range(4):
            ind[i] = ind[i] + np.sum([p.shape[0] for p in rim_list[0:i]])  # get correct indices for the stacked array
        ind = np.hstack(ind).astype(int)

        # Keep rim_list aligned with filtered E_RIMs/delta_q.
        # ind is a "flat" index over the concatenation of rim_list[0], rim_list[1], rim_list[2], rim_list[3]
        counts = [0 if getattr(rim_list[i], "size", 0) == 0 else rim_list[i].shape[0] for i in range(4)]
        offsets = np.cumsum([0] + counts)  # length 5

        new_rim_list = []
        for i in range(4):
            if counts[i] == 0:
                new_rim_list.append(np.array([]))
                continue

            mask = (ind >= offsets[i]) & (ind < offsets[i + 1])
            local_ind = ind[mask] - offsets[i]
            new_rim_list.append(rim_list[i][local_ind])

        # Side-effect: keep internal rim_list aligned to filtered arrays
        self.__rim_list = new_rim_list

        self.__E_RIMs = np.array(self.__E_RIMs)[ind]
        self.__delta_q = self.__delta_q[ind]

        E_RIMs_total = float(np.sum(self.__E_RIMs))
        self.__proc_E_RIMs = np.array(self.__E_RIMs) / E_RIMs_total * 100

        if cbonds_flag:
            self.__custom_bonds = custom_bonds  # restore the user input

    def add_custom_bonds(self, bonds: NDArray) -> None:
        """Add custom bonds after creating the object.

        Args:
            bonds:
                1D or 2Darray with atom indices, [[i,j]...]
        """
        self.__custom_bonds = np.atleast_2d(bonds)  # additional bonds for analysis of non-covalent interactions

    def set_bond_params(self, covf=constants.COVALENCY_FACTOR, vdwf=constants.VAN_DER_WAALS_FACTOR):
        """
        Args:
            covf:
                float factor for covalent radii to determine covalent bonds
            vdwf:
                float factor for vdw radii to get the upper limit of the custom bond lengths
        """
        self.__covf = covf
        self.__vdwf = vdwf


def jedi_analysis(
    atoms: ase.atoms.Atoms,
    rim_list: list,
    B: np.ndarray,
    H_cart: np.ndarray,
    delta_q: np.ndarray,
    deltaE: float,
    printout: bool | None = None,
    use_ase_units: bool = False,
):
    """
    Analysis of strain energy stored in redundant internal coordinates.

    Parameters
    ----------
    atoms: class
        An ASE Atoms object to determine the atomic species of the indices.
    rim_list: list
        A list of 4 numpy 2D arrays the first array containing bonds, second custom bonds, third bond angles, fourth dihedrals.
    B: np array
        B matrix.
    H_cart: np array
        Hessian in cartesian coordinates.
    delta_q: np array
        Array of deformations along the RICs.
    self._deltaE: float
        Energy difference between the geometries.
    printout: bool
        Flag to print the output.
    ase_units: bool
        Flag to get eV for energies å fo lengths and degree for angles otherwise it is kcal/mol, Bohr and radians.
    Returns:
        Analysis of RIMs.
    """
    # Matrix Calculations via bmatrix module
    H_q = bmatrix.hessian_to_ric(B, H_cart)

    # Energy calculations
    E_RIMs_total = 0.5 * delta_q.T.dot(H_q).dot(delta_q)
    if B.ndim == 1:
        E_RIMs = np.array([0.5 * delta_q[0] * H_q * delta_q[0]])
    else:
        E_RIMs = np.sum(0.5 * (delta_q * H_q).T * delta_q, axis=1)
    proc_E_RIMs = 100 * E_RIMs / E_RIMs_total

    proc_geom_RIMs = 100 * (E_RIMs_total - deltaE) / deltaE

    if printout:
        reporting.jedi_printout(
            atoms,
            rim_list,
            delta_q,
            deltaE,
            E_RIMs_total,
            proc_geom_RIMs,
            proc_E_RIMs,
            E_RIMs,
            ase_units=use_ase_units,
        )

    return proc_E_RIMs, E_RIMs, E_RIMs_total, proc_geom_RIMs, delta_q


def get_hbonds(mol, covf=constants.COVALENCY_FACTOR, vdwf=constants.VAN_DER_WAALS_FACTOR):
    """
    Get all hbonds in a structure.
    Hbonds are defined as the HY bond inside X-H···Y where X and Y can be O, N, F and the angle XHY is larger than 90°
    and the distance between HY is shorter than 0.9 times the sum of the vdw radii of H and Y.

    Parameters
    ----------
    mol: class
        Structure of which the hbonds should be determined.
    Returns:
        2D array of indices.

    """
    cutoff = ase.neighborlist.natural_cutoffs(mol, mult=covf)  ## cutoff for covalent bonds see Bakken et al.
    bl = np.vstack(ase.neighborlist.neighbor_list("ij", a=mol, cutoff=cutoff)).T  # determine covalent bonds

    bl = bl[bl[:, 0] < bl[:, 1]]  # remove double mentioned
    bl = np.unique(bl, axis=0)

    hpartner = ["N", "O", "F"]
    hpartner_ls = []
    hcutoff = {
        ("H", "N"): vdwf * (vdw_radii[1] + vdw_radii[7]),
        ("H", "O"): vdwf * (vdw_radii[1] + vdw_radii[8]),
        ("H", "F"): vdwf * (vdw_radii[1] + vdw_radii[9]),
    }  # save the maximum distances for given pairs to be taken account as interactions
    hbond_ls = []  # create a list to store all the bonds
    for i in range(len(mol)):
        if mol.symbols[i] in hpartner:  # check atoms indices of N F O elements
            hpartner_ls.append(i)
    for i in bl:
        if mol.symbols[i[0]] == "H" and mol.symbols[i[1]] in hpartner:
            for j in hpartner_ls:
                if j != i[1] and (
                    mol.get_distance(i[0], j, mic=True) < hcutoff[(mol.symbols[i[0]], mol.symbols[j])]
                    and mol.get_angle(i[1], i[0], j, mic=True) > 90
                ):
                    hbond_ls.append([i[0], j])
        elif mol.symbols[i[0]] in hpartner and mol.symbols[i[1]] == "H":
            for j in hpartner_ls:
                if j != i[0] and (
                    mol.get_distance(i[1], j, mic=True) < hcutoff[(mol.symbols[i[1]], mol.symbols[j])]
                    and mol.get_angle(i[0], i[1], j, mic=True) > 90
                ):
                    hbond_ls.append([i[1], j])
    if len(hbond_ls) > 0:
        hbond_ls = np.array(hbond_ls)
        hbond_ls = np.sort(hbond_ls, axis=1)
        hbond_ls = np.atleast_2d(hbond_ls)
    return hbond_ls
