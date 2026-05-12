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
from typing_extensions import Any, Dict, List, Optional, Union, deprecated

from strainjedi import bmatrix, constants, reporting, rics
from strainjedi.visualization import ColorMapper, MatplotlibVisualizer, VMDVisualizer


@jsonable("jedi")
class Jedi:
    def __init__(
        self,
        atoms0: ase.atoms.Atoms,
        atomsF: ase.atoms.Atoms,
        modes: ase.vibrations.data.VibrationsData,
        epot: Optional[np.ndarray] = None,
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
        self._atoms0 = atoms0  # ref state
        self._atomsF = atomsF  # strained state
        self._covf = constants.COVALENCY_FACTOR
        self._vdwf = constants.VAN_DER_WAALS_FACTOR
        self._indices = np.arange(0, len(self._atoms0))
        self._custom_bonds = None  # list of custom added bonds
        self._rim_list = rics.intersect(
            rics.calculate(self._atoms0, self._indices, self._custom_bonds),
            rics.calculate(self._atomsF, self._indices, self._custom_bonds),
        )
        self._B = bmatrix.calculate(self._atoms0, self._rim_list)
        self.get_delta_q()
        self._H = modes._hessian2d / (ase.units.Hartree / ase.units.Bohr**2)
        self._energies = epot  # energies of the geometries
        self._proc_E_RIMs = None  # list of procentual energy stored in single RIMs
        self._part_rim_list = None  # rim list for election of atoms
        self._E_RIMs = None  # list of energies stored in the rims
        self._E_RIMs_total = None  # sum of E_rims
        self._ase_units = False
        self._qF = None  # bond lengths and angles in Bohr and degree in distorted molecule
        self._q0 = None  # bond lengths and angles in Bohr and degree in relaxed molecule

    def todict(self) -> Dict[str, Any]:
        """make it saveable with .write()"""
        return {
            "atoms0": self._atoms0,
            "atomsF": self._atomsF,
            "hessian": self._H,
            "bmatrix": self._B,
            "delta_q": self._delta_q,
            "rim_list": self._rim_list,
            "energies": self._energies,
            "indices": self._indices,
            "E_RIMS": self._E_RIMs,
            "proc_E_RIMS": self._proc_E_RIMs,
            "custom_bonds": self._custom_bonds,
        }

    @classmethod
    def fromdict(cls, data: Dict[str, Any]) -> "Jedi":
        """
        Reconstruct a Jedi object from a dictionary.
        Raises ValueError if required fields are missing or have incorrect types.
        """

        # Required fields; This will implicitly raise a KeyError if either is not found.
        # However, as these are absolutely necessary, this is okay to do.
        atoms0 = data["atoms0"]
        atomsF = data["atomsF"]

        # Try to reconstruct VibrationsData if possible
        if "modes" in data and isinstance(data["modes"], ase.vibrations.VibrationsData):
            modes = data["modes"]
        elif "hessian" in data:
            # Support legacy dicts with hessian/indices
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
            ("_H", "hessian"),
            ("_B", "bmatrix"),
            ("_delta_q", "delta_q"),
            ("_rim_list", "rim_list"),
            ("_indices", "indices"),
            ("_E_RIMs", "E_RIMS"),
            ("_proc_E_RIMs", "proc_E_RIMS"),
            ("_custom_bonds", "custom_bonds"),
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
        self.ase_units = ase_units
        # get necessary data
        if len(self._atoms0) != self._H.shape[0] / 3:
            raise ValueError(
                "Hessian has not the fitting shape, possibly a partial hessian. Please try partial_analysis"
            )

        E_geometries = self.get_energies()[0]

        # run the analysis
        (
            self._proc_E_RIMs,
            self._E_RIMs,
            self._E_RIMs_total,
            proc_geom_RIMs,
            self._delta_q,
        ) = jedi_analysis(
            self._atoms0,
            self._rim_list,
            self._B,
            self._H,
            self._delta_q,
            E_geometries,
            ase_units=ase_units,
        )

        if indices:  # get only rims of interest
            self.post_process(indices)
            self._E_RIMs_total = sum(self._E_RIMs)
            proc_geom_RIMs = 100 * (sum(self._E_RIMs) - E_geometries) / E_geometries

        if printout:
            reporting.jedi_printout(
                self._atoms0,
                self._rim_list,
                self._delta_q,
                E_geometries,
                self._E_RIMs_total,
                proc_geom_RIMs,
                self._proc_E_RIMs,
                self._E_RIMs,
                ase_units=ase_units,
            )

    def get_energies(self) -> List[float]:
        """Calls the energies of the Atoms objects.

        Returns:
            [energy difference, energy of atomsF, energy of atoms0]
        """
        e0 = self._atoms0.calc.get_potential_energy()
        eF = self._atomsF.calc.get_potential_energy()
        if not self._ase_units:
            e0 *= ase.units.mol / ase.units.kcal
            eF *= ase.units.mol / ase.units.kcal
        deltaE = eF - e0
        self._energies = [deltaE, eF, e0]
        return [deltaE, eF, e0]  # WHY??

    def get_delta_q(self):
        """Compute and store q0, qF, delta_q for the molecule RICs."""
        self._q0, self._qF, self._delta_q = rics.subtract(self._atoms0, self._atomsF, self._rim_list)

    def visualize(
        self,
        visualizer="mpl",
        colormap="green_red",
        output_dir: Union[Path, str] = "visualization",
        single_mode: Optional[str] = None,
        man_strain: Optional[float] = None,
        show: Optional[bool] = False,
        show_indices: Optional[bool] = False,
        box: Optional[bool] = False,
        split_bonds: Optional[bool] = True,
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
        """

        if self._proc_E_RIMs is None or len(self._proc_E_RIMs) == 0:
            raise ValueError("Analysis has not been run. Jedi.run() must be called before Jedi.visualize()")

        if show and not visualizer == "mpl":
            warnings.warn(
                "'show=True' only works for the Matplotlib Visualizer and will be ignored.",
                UserWarning,
            )

        if show_indices and not visualizer == "mpl":
            warnings.warn(
                "'show_indices=True' only works for the Matplotlib Visualizer and will be ignored.",
                UserWarning,
            )

        energy_unit = "eV" if self.ase_units else "kcal/mol"

        valid_modes = ["bl", "ba", "da", "all"]
        if not single_mode:
            mode_list = valid_modes
        elif single_mode in valid_modes:
            mode_list = [single_mode]
        else:
            raise ValueError(f"Unknown mode '{single_mode}'. single_mode must be in: {valid_modes} or None")

        mapper = ColorMapper(self)
        self.visualization_data = mapper.get_visualization_data(mode_list, colormap, man_strain, split_bonds)

        if visualizer == "mpl":
            vis = MatplotlibVisualizer(self.visualization_data, mapper, output_dir, energy_unit)
            vis.run(show, show_indices, box)

        elif visualizer == "vmd":
            vis = VMDVisualizer(self.visualization_data, mapper, output_dir, energy_unit)
            vis.write_inputs(box)

        else:
            raise ValueError("Unknown visualizer. Visualizer must be 'mpl' or 'vmd'.")

    @deprecated("Use Jedi.visualize(visualizer='vmd') instead.")
    def vmd_gen(
        self,
        _des_colors: Optional[Dict] = None,
        _box: bool = False,
        man_strain: Optional[float] = None,
        modus: Optional[str] = None,
        _colorbar: bool = True,
        label: Union[Path, str] = "vmd",
        incl_coloring: Optional[str] = None,
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
        self.ase_units = ase_units
        if 3 * len(indices) < len(self._H):
            raise ValueError("to little indices for the given hessian")

        has_custom_bonds = False
        if self._custom_bonds is not None:
            old_custom_bonds = self._custom_bonds.copy()
            has_custom_bonds = True
            self._custom_bonds = self._custom_bonds[np.isin(self._custom_bonds, indices).all(axis=1)]

        rim_list = rics.calculate(self._atoms0, self._indices, self._custom_bonds)

        if len(rim_list) == 0:
            raise ValueError("Chosen indexlist has no rims")

        self._B = bmatrix.calculate(self._atoms0, rim_list, indices=self._indices)
        # set B matrix values of not considered atoms to 0
        self._B = bmatrix.restrict(self._B, indices)

        self.get_delta_q()

        E_geometries = self.get_energies()[0]

        (
            self._proc_E_RIMs,
            self._E_RIMs,
            self._E_RIMs_total,
            _proc_geom_RIMs,
            self._delta_q,
        ) = jedi_analysis(
            self._atomsF,
            rim_list,
            self._B,
            self._H,
            self._delta_q,
            E_geometries,
            ase_units=ase_units,
        )
        # get values of rims inside the substructure
        self.post_process(indices)
        self._E_RIMs_total = sum(self._E_RIMs)
        proc_geom_RIMs = 100 * (sum(self._E_RIMs) - E_geometries) / E_geometries
        reporting.jedi_printout(
            self._atoms0,
            self._rim_list,
            self._delta_q,
            E_geometries,
            self._E_RIMs_total,
            proc_geom_RIMs,
            self._proc_E_RIMs,
            self._E_RIMs,
            ase_units=ase_units,
        )

        if has_custom_bonds:
            self._custom_bonds = old_custom_bonds  # restore the user input

    def post_process(
        self, indices
    ):  # a function to get segments of all full analysis for better understanding of local strain
        """
        get only the values of RICs inside a defined substructure

        Args:
            indices:
                list of indices of atoms in desired substructure
        Returns:
            Values for analyzed RIMs in the defined substructure
        """
        # get rims with only the considered atoms
        self._indices = indices
        rim_list = self._rim_list
        cbonds_flag = False
        if self._custom_bonds is not None:
            custom_bonds = self._custom_bonds.copy()
            cbonds_flag = True
            self._custom_bonds = self._custom_bonds[np.isin(self._custom_bonds, indices).all(axis=1)]
        rim_p = rics.intersect(
            rics.calculate(self._atoms0, indices, self._custom_bonds),
            rics.calculate(self._atomsF, indices, self._custom_bonds),
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
        ind = np.hstack(ind)
        ind = ind.astype(int)

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

        # FIXME: This is a side-effect that should be avoided.
        self._rim_list = new_rim_list

        self._E_RIMs = np.array(self._E_RIMs)[ind]
        self._delta_q = self._delta_q[ind]
        E_RIMs_total = sum(self._E_RIMs)
        self._proc_E_RIMs = np.array(self._E_RIMs) / E_RIMs_total * 100
        if cbonds_flag:
            self._custom_bonds = custom_bonds  # restore the user input
        pass

    def add_custom_bonds(self, bonds: NDArray) -> None:
        """Add custom bonds after creating the object.

        Args:
            bonds:
                1D or 2Darray with atom indices, [[i,j]...]
        """

        self._custom_bonds = np.atleast_2d(bonds)  # additional bonds for analysis of non-covalent interactions

    def set_bond_params(self, covf=constants.COVALENCY_FACTOR, vdwf=constants.VAN_DER_WAALS_FACTOR):
        """
        Args:
            covf:
                float factor for  covalent radii to determine covalent bonds
            vdwf:
                float factor for vdw radii to get the upper limit of the custom bond lengths
        """
        self._covf = covf
        self._vdwf = vdwf


def jedi_analysis(
    atoms: ase.atoms.Atoms,
    rim_list: List,
    B: np.ndarray,
    H_cart: np.ndarray,
    delta_q: np.ndarray,
    E_geometries: float,
    printout: bool | None = None,
    ase_units: bool = False,
):
    """
    Analysis of strain energy stored in redundant internal coordinates.

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
    E_geometries: float
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

    # Unit handling
    if ase_units:
        b = rim_list[0].shape[0] + rim_list[1].shape[0]
        delta_q[0:b] *= ase.units.Bohr
        delta_q[b:] = np.degrees(delta_q[b:])
        E_RIMs = E_RIMs * ase.units.Hartree
        E_RIMs_total *= ase.units.Hartree
    else:
        E_RIMs = E_RIMs / ase.units.kcal * ase.units.mol * ase.units.Hartree
        E_RIMs_total *= ase.units.mol / ase.units.kcal * ase.units.Hartree

    proc_geom_RIMs = 100 * (E_RIMs_total - E_geometries) / E_geometries

    if printout:
        reporting.jedi_printout(
            atoms,
            rim_list,
            delta_q,
            E_geometries,
            E_RIMs_total,
            proc_geom_RIMs,
            proc_E_RIMs,
            E_RIMs,
            ase_units=ase_units,
        )

    return proc_E_RIMs, E_RIMs, E_RIMs_total, proc_geom_RIMs, delta_q


def get_hbonds(mol, covf=constants.COVALENCY_FACTOR, vdwf=constants.VAN_DER_WAALS_FACTOR):
    """
    Get all hbonds in a structure.
    Hbonds are defined as the HY bond inside X-H···Y where X and Y can be O, N, F and the angle XHY is larger than 90°
    and the distance between HY is shorter than 0.9 times the sum of the vdw radii of H and Y.

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
                if j != i[1]:
                    if (
                        mol.get_distance(i[0], j, mic=True) < hcutoff[(mol.symbols[i[0]], mol.symbols[j])]
                        and mol.get_angle(i[1], i[0], j, mic=True) > 90
                    ):
                        hbond_ls.append([i[0], j])
        elif mol.symbols[i[0]] in hpartner and mol.symbols[i[1]] == "H":
            for j in hpartner_ls:
                if j != i[0]:
                    if (
                        mol.get_distance(i[1], j, mic=True) < hcutoff[(mol.symbols[i[1]], mol.symbols[j])]
                        and mol.get_angle(i[0], i[1], j, mic=True) > 90
                    ):
                        hbond_ls.append([i[1], j])
    if len(hbond_ls) > 0:
        hbond_ls = np.array(hbond_ls)
        hbond_ls = np.sort(hbond_ls, axis=1)
        hbond_ls = np.atleast_2d(hbond_ls)
    return hbond_ls
