import collections
import itertools
import warnings
from pathlib import Path

import ase.geometry
import ase.neighborlist
import ase.vibrations
import numpy as np
from ase.atoms import Atoms
from ase.data.vdw import vdw_radii
from ase.units import Bohr, Hartree, kcal, mol
from ase.utils import jsonable
from numpy.typing import NDArray
from typing_extensions import Any, Dict, List, Optional, Union, deprecated

from strainjedi import reporting
from strainjedi.visualization import ColorMapper, MatplotlibVisualizer, VMDVisualizer


@jsonable("jedi")
class Jedi:
    def __init__(
        self,
        atoms0: ase.atoms.Atoms,
        atomsF: ase.atoms.Atoms,
        modes: ase.vibrations.data.VibrationsData,
        epot: Union[np.ndarray, None] = None,
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
        self.atoms0 = atoms0  # ref state
        self.atomsF = atomsF  # strained state
        self.modes = modes  # VibrationsData object
        self.vdwf = 0.9
        self.covf = 1.3  # cutoff for covalent bonds see Bakken et al.
        self.indices = np.arange(0, len(self.atoms0))
        self.custom_bonds = None  # list of custom added bonds
        self.get_common_rims()
        self.get_b_matrix()
        self.get_delta_q()
        self.get_hessian()
        self.energies = epot  # energies of the geometries
        self.proc_E_RIMs = None  # list of procentual energy stored in single RIMs
        self.part_rim_list = None  # rim list for election of atoms
        self.E_RIMs = None  # list of energies stored in the rims
        self.E_RIMs_total = None  # sum of E_rims
        self.ase_units = False
        self.qF = None  # bond lengths and angles in Bohr and degree in distorted molecule
        self.q0 = None  # bond lengths and angles in Bohr and degree in relaxed molecule

    def todict(self) -> Dict[str, Any]:
        """make it saveable with .write()"""
        return {
            "atoms0": self.atoms0,
            "atomsF": self.atomsF,
            #'modes': self.modes,
            "hessian": self.H,
            "bmatrix": self.B,
            "delta_q": self.delta_q,
            "rim_list": self.rim_list,
            "energies": self.energies,
            "indices": self.indices,
            "E_RIMS": self.E_RIMs,
            "proc_E_RIMS": self.proc_E_RIMs,
            "custom_bonds": self.custom_bonds,
        }

    @classmethod
    def fromdict(cls, data: Dict[str, Any]) -> "Jedi":
        """make it readable with .read()"""
        # mypy is understandably suspicious of data coming from a dict that
        # holds mixed types, but it can see if we sanity-check with 'assert'
        assert isinstance(data["atoms0"], Atoms)
        assert isinstance(data["atomsF"], Atoms)
        try:
            assert isinstance(data["modes"], ase.vibrations.VibrationsData)
            cl = cls(data["atoms0"], data["atomsF"], data["modes"])
        # FIXME: what exception might occur here?
        except Exception:
            pass

        if data["hessian"] is not None:
            assert isinstance(data["hessian"], (collections.abc.Sequence, np.ndarray))

            if data["indices"] is not None:
                assert isinstance(data["indices"], (collections.abc.Sequence, np.ndarray))
                modes = ase.vibrations.VibrationsData.from_2d(data["atoms0"], data["hessian"], data["indices"])
                cl = cls(data["atoms0"], data["atomsF"], modes)
                cl.indices = data["indices"]
            else:
                modes = ase.vibrations.VibrationsData.from_2d(data["atoms0"], data["hessian"])
                cl = cls(data["atoms0"], data["atomsF"], modes)
            cl.H = data["hessian"]
        if data["bmatrix"] is not None:
            assert isinstance(data["bmatrix"], (collections.abc.Sequence, np.ndarray))
            cl.B = data["bmatrix"]
        if data["delta_q"] is not None:
            assert isinstance(data["delta_q"], (collections.abc.Sequence, np.ndarray))
            cl.delta_q = data["delta_q"]
        if data["rim_list"] is not None:
            assert isinstance(data["rim_list"], (collections.abc.Sequence, np.ndarray))
            cl.rim_list = data["rim_list"]
        if data["energies"] is not None:
            assert isinstance(data["energies"], (collections.abc.Sequence, list))
            cl.energies = data["energies"]
        if data["E_RIMS"] is not None:
            assert isinstance(data["proc_E_RIMS"], (collections.abc.Sequence, np.ndarray))
            cl.E_RIMs = data["E_RIMS"]
        if data["proc_E_RIMS"] is not None:
            assert isinstance(data["proc_E_RIMS"], (collections.abc.Sequence, np.ndarray))
            cl.proc_E_RIMs = data["proc_E_RIMS"]
        if data["custom_bonds"] is not None:
            assert isinstance(data["custom_bonds"], (collections.abc.Sequence, list))
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
        if len(self.atoms0) != self.H.shape[0] / 3:
            raise ValueError(
                "Hessian has not the fitting shape, possibly a partial hessian. Please try partial_analysis"
            )

        try:  # Get energies from calculator
            all_E_geometries = self.get_energies()
        # FIXME: which exception might happen here?
        except Exception:  # Fallback to custom energies
            all_E_geometries = self.energies

        E_geometries = all_E_geometries[0]

        # run the analysis
        (
            self.proc_E_RIMs,
            self.E_RIMs,
            self.E_RIMs_total,
            proc_geom_RIMs,
            self.delta_q,
        ) = jedi_analysis(
            self.atoms0,
            self.rim_list,
            self.B,
            self.H,
            self.delta_q,
            E_geometries,
            ase_units=ase_units,
        )

        if indices:  # get only rims of interest
            self.post_process(indices)
            self.E_RIMs_total = sum(self.E_RIMs)
            proc_geom_RIMs = 100 * (sum(self.E_RIMs) - E_geometries) / E_geometries

        if printout:
            reporting.jedi_printout(
                self.atoms0,
                self.rim_list,
                self.delta_q,
                E_geometries,
                self.E_RIMs_total,
                proc_geom_RIMs,
                self.proc_E_RIMs,
                self.E_RIMs,
                ase_units=ase_units,
            )

    def get_rims(self, mol):
        """Gets the redundant internal coordinates"""

        cutoff = ase.neighborlist.natural_cutoffs(mol, mult=self.covf)  ## cutoff for covalent bonds see Bakken et al.
        bl = np.vstack(ase.neighborlist.neighbor_list("ij", a=mol, cutoff=cutoff)).T  # determine covalent bonds

        bl = bl[bl[:, 0] < bl[:, 1]]  # remove double metioned
        bl, counts = np.unique(bl, return_counts=True, axis=0)
        if ~np.all(counts == 1):
            print(
                "unit cell too small hessian not calculated for self interaction \
                   jedi analysis for a finite system consisting of the cell will be conducted"
            )
        bl = np.atleast_2d(bl)

        if len(self.indices) != len(mol):
            bl = bl[np.all([np.isin(bl[:, 0], self.indices), np.isin(bl[:, 1], self.indices)], axis=0)]

        rim_list = [bl]

        # possibility of adding custom bonds like hbonds, long range interactions
        if self.custom_bonds is not None:
            bl = np.vstack((bl, self.custom_bonds))
            rim_list.append(self.custom_bonds)
        if self.custom_bonds is None:
            rim_list.append(np.array([]))

        # compute adjacency
        neighbors = [[] for _ in range(len(mol))]
        for a, b in bl:
            a = int(a)
            b = int(b)
            neighbors[a].append(b)
            neighbors[b].append(a)

        ########find angles
        # create array containing all angles (ba)
        ba_rows = []
        for x, nbrs in enumerate(neighbors):
            for o1, o2 in itertools.combinations(nbrs, 2):
                ba_rows.append([o1, x, o2])

        ba = np.asarray(ba_rows)
        ba_flag = ba.size > 0

        if ba_flag:
            ba = np.atleast_2d(ba)
            ba = ba[ba[:, 1].argsort(kind="stable")]  # sort by atom2
            ba = ba[ba[:, 0].argsort(kind="stable")]  # sort by atom1

            nan = np.full((len(ba), 1), -1)
            nan = np.hstack((nan, ba))
            rim_list.append(ba)
        else:
            rim_list.append(np.array([]))

        # degree of each node = how often it appears anywhere in bond list.
        # This is represented as the length of each neighbor[x] = [a, b, ...]
        deg = np.fromiter((len(n) for n in neighbors), dtype=np.int64)

        # A bond is torsionable if both endpoints have degree > 1
        mask = (deg[bl[:, 0]] > 1) & (deg[bl[:, 1]] > 1)
        torsionable_bonds = bl[mask]

        # torsion angles
        da_rows = []
        LINEAR = (0.0, 180.0, 360.0)
        for torsionable_row in torsionable_bonds:
            j, k = map(int, torsionable_row)

            left_atoms = [i for i in neighbors[j] if i != k]
            right_atoms = [l for l in neighbors[k] if l != j]

            for i, l in itertools.product(left_atoms, right_atoms):
                da_pre = np.array([i, j, k, l], dtype=int)

                if len(set(da_pre)) != 4:
                    print(
                        "bonds for dihedral angle span over more than one unit cell\n %s will not be taken into account in the further analysis"
                        % (np.atleast_2d(da_pre))
                    )
                    continue

                try:
                    if round(mol.get_angle(i, j, k, mic=True)) in LINEAR:
                        continue
                    if round(mol.get_angle(j, k, l, mic=True)) in LINEAR:
                        continue
                except Exception:
                    continue

                da_rows.append(da_pre)

        da = np.asarray(da_rows, dtype=int)
        rim_list.append(np.atleast_2d(da) if da.size > 0 else np.array([]))
        rim_list_sorted = [arr if arr.size == 0 else np.sort(arr, axis=1, kind="mergesort") for arr in rim_list]

        return rim_list_sorted

    def get_common_rims(self):
        """Get only the RICs in both structures, bond breaks cannot be analysed logically"""
        rim_atoms0 = self.get_rims(self.atoms0)
        rim_atomsF = self.get_rims(self.atomsF)
        if len(rim_atoms0[0]) != len(rim_atomsF[0]):
            (
                warnings.warn_explicit(
                    f"The distorted structure has a different number of bonds ({len(rim_atomsF[0])})\n"
                    f"compared to the relaxed structure ({len(rim_atoms0[0])}). "
                    f"In this case the JEDI strain analysis can not be applied correctly.",
                    UserWarning,
                    "",
                    0,
                )
            )
        if len(rim_atoms0[2]) != len(rim_atomsF[2]):
            (
                warnings.warn_explicit(
                    f"The distorted structure has a different number of angles ({len(rim_atomsF[2])})"
                    f" compared to the relaxed structure ({len(rim_atoms0[2])}). ",
                    UserWarning,
                    "",
                    0,
                )
            )
        if len(rim_atoms0[3]) != len(rim_atomsF[3]):
            (
                warnings.warn_explicit(
                    f"The distorted structure has a different number of dihedral angles ({len(rim_atomsF[3])})"
                    f" compared to the relaxed structure ({len(rim_atoms0[3])}).",
                    UserWarning,
                    "",
                    0,
                )
            )
        common_rims = [np.empty(0) for _ in range(4)]
        for i in range(len(rim_atoms0)):
            if rim_atoms0[i].shape[0] == 0:
                continue
            elif rim_atomsF[i].shape[0] == 0:
                common_rims[i] = np.empty(0)
            else:
                rim_atoms0v = rim_atoms0[i].view([("", rim_atoms0[i].dtype)] * rim_atoms0[i].shape[1]).ravel()
                rim_atomsFv = (
                    rim_atomsF[i].view([("", rim_atomsF[i].dtype)] * rim_atomsF[i].shape[1]).ravel()
                )  # get a viable input for np.intersect1d()

                rim_l, ind, _ = np.intersect1d(
                    rim_atoms0v, rim_atomsFv, return_indices=True
                )  # get the rims that exist in both structures
                rim_l = rim_l[ind.argsort(kind="stable")]

                common_rims[i] = rim_l.view(rim_atoms0[i].dtype).reshape(-1, rim_atoms0[i].shape[1])
        common_rims_sorted = [arr if arr.size == 0 else np.sort(arr, axis=1, kind="mergesort") for arr in common_rims]
        # FIXME: This should not be mutated any further. WTF?
        self.rim_list = common_rims_sorted

        return rim_atoms0

    def get_hessian(self):
        """Calls the hessian from the VibrationsData object"""
        hessian = self.modes._hessian2d
        self.H = hessian / (Hartree / Bohr**2)

    def get_b_matrix(self, indices=None):
        """Calculates the derivatives of the RICs with respect to all cartesian coordinates using ase functions"""
        mol = self.atoms0
        if indices is None:
            indices = np.arange(0, len(mol))
        if len(self.rim_list) == 0:
            self.get_common_rims()

        rim_size = sum([np.shape(length)[0] for length in self.rim_list])
        b = np.zeros([int(len(indices) * 3), int(rim_size)], dtype=float)  # shape of B-matrix (NCarts,NRIMs)

        # get all derivatives
        column = 0  # Initilization of columns to specifiy position in B-Matrix
        for q in self.rim_list[0]:
            row = 0  # Initilization of rows to specifiy position in B-Matrix

            ########  Section for stretches  #########

            BL = [int(q[0]), int(q[1])]  # create list of involved atoms
            q_i, q_j = BL

            u = mol.get_distance(q_i, q_j, mic=True, vector=True)
            for NAtom in indices:  # for-loop of Number of Atoms
                for q in BL:
                    if (
                        NAtom == q
                    ):  # derivative of redundnat internal coordinate w/ respect to cartesian coordinates is not equal zero
                        # if redundant internal coordinate (q) contains the Atomnumber (NAtoms) of the cartesian coordinate (x0_coords) from which is derived from.

                        # if-/elif-statement for the right sign-factor (see [1])
                        if q == q_i:
                            b_i = ase.geometry.get_distances_derivatives(np.atleast_2d(u))[0][0]
                            b[row : row + 3, column] = b_i  # change value of zero array at specified position to b_i
                        elif q == q_j:
                            b_i = ase.geometry.get_distances_derivatives(np.atleast_2d(u))[0][1]
                            b[row : row + 3, column] = b_i  # change value of zero array at specified position to b_i
                row += 3
            column += 1

        for q in self.rim_list[1]:
            row = 0  # Initilization of rows to specifiy position in B-Matrix

            ########  Section for custom stretches  #########

            CL = [int(q[0]), int(q[1])]  # create list of involved atoms
            q_i, q_j = CL

            u = mol.get_distance(q_i, q_j, mic=True, vector=True)
            for NAtom in indices:  # for-loop of Number of Atoms
                for q in CL:
                    if NAtom == q:
                        # if-/elif-statement for the right sign-factor
                        if q == q_i:
                            b_i = ase.geometry.get_distances_derivatives(np.atleast_2d(u))[0][0]
                            b[row : row + 3, column] = b_i  # change value of zero array at specified position to b_i
                        elif q == q_j:
                            b_i = ase.geometry.get_distances_derivatives(np.atleast_2d(u))[0][1]
                            b[row : row + 3, column] = b_i  # change value of zero array at specified position to b_i

                row += 3
            column += 1

        #################ba###############################

        for q in self.rim_list[2]:
            row = 0  # Initilization of rows to specifiy position in B-Matrix

            BA = [int(q[0]), int(q[1]), int(q[2])]  # create list of involved atoms
            q_i, q_j, q_k = BA
            u = mol.get_distance(q_i, q_j, mic=True, vector=True)
            v = mol.get_distance(q_k, q_j, mic=True, vector=True)

            def get_B_matrix_angles_derivatives(u, v):
                angle = ase.geometry.get_angles(u, v)  # angle between v and u

                if angle == 180 or angle == 0:  # an auxilliary vector is used if linear angles are existing
                    (u, v), (lu, lv) = ase.geometry.conditional_find_mic([u, v], cell=None, pbc=None)
                    nu = u / lu
                    nv = v / lv
                    if (np.arccos(np.dot(nu, (np.array([1, -1, 1]))))) == np.pi:
                        w = np.cross(nu, ([-1, 1, 1]))
                    else:
                        w = np.cross(nu, ([1, -1, 1]))

                    nw = w / np.linalg.norm(w)
                    d_ba1 = (np.cross(nu, nw)) / np.linalg.norm(u)
                    d_ba2 = (-1 * ((np.cross(nu, nw)) / np.linalg.norm(u))) + (
                        -1 * ((np.cross(nw, nv)) / np.linalg.norm(v))
                    )  # equation to calculate dBA/dx [1]
                    d_ba3 = (np.cross(nw, nv)) / np.linalg.norm(v)
                    d_ba = np.array([[d_ba1[0], d_ba2[0], d_ba3[0]]])

                else:
                    d_ba = np.radians(ase.geometry.get_angles_derivatives(u, v))
                return d_ba * Bohr

            for NAtom in indices:  # for-loop of Number of Atoms
                for q in BA:
                    if NAtom == q:
                        if q == q_j:  # if-Statements for sign-factors
                            b_j = get_B_matrix_angles_derivatives(np.atleast_2d(u), np.atleast_2d(v))[0][1]
                            b[row : row + 3, column] = -b_j
                        elif q == q_i:
                            b_j = get_B_matrix_angles_derivatives(np.atleast_2d(u), np.atleast_2d(v))[0][0]
                            b[row : row + 3, column] = -b_j
                        elif q == q_k:
                            b_j = get_B_matrix_angles_derivatives(np.atleast_2d(u), np.atleast_2d(v))[0][2]
                            b[row : row + 3, column] = -b_j
                row += 3
            column += 1

        for q in self.rim_list[3]:
            row = 0  # Initilization of rows to specifiy position in B-Matrix

            DA = [
                int(q[0]),
                int(q[1]),
                int(q[2]),
                int(q[3]),
            ]  # create list of involved atoms
            q_i, q_j, q_k, q_l = DA

            u = np.copy(
                np.atleast_2d(mol.get_distance(q_i, q_j, mic=True, vector=True))
            )  #####copy needed because derivative function rewrites vector variable as normed vector
            w = np.copy(np.atleast_2d(mol.get_distance(q_k, q_l, mic=True, vector=True)))
            v = np.copy(np.atleast_2d(mol.get_distance(q_j, q_k, mic=True, vector=True)))

            for NAtom in indices:  # for-loop of Number of Atoms
                for q in DA:
                    if NAtom == q:
                        if q == q_i:  # if-Statements for sign-factors
                            b_k = np.radians(ase.geometry.get_dihedrals_derivatives(u, v, w)[0][0]) * Bohr
                            b[row : row + 3, column] = b_k
                            u = np.copy(
                                np.atleast_2d(mol.get_distance(q_i, q_j, mic=True, vector=True))
                            )  #####copy needed because derivative function rewrites vector variable as normed vector
                            w = np.copy(np.atleast_2d(mol.get_distance(q_k, q_l, mic=True, vector=True)))
                            v = np.copy(np.atleast_2d(mol.get_distance(q_j, q_k, mic=True, vector=True)))

                        elif q == q_j:
                            b_k = np.radians(ase.geometry.get_dihedrals_derivatives(u, v, w)[0][1]) * Bohr
                            b[row : row + 3, column] = b_k
                            u = np.copy(
                                np.atleast_2d(mol.get_distance(q_i, q_j, mic=True, vector=True))
                            )  #####copy needed because derivative function rewrites vector variable as normed vector
                            w = np.copy(np.atleast_2d(mol.get_distance(q_k, q_l, mic=True, vector=True)))
                            v = np.copy(np.atleast_2d(mol.get_distance(q_j, q_k, mic=True, vector=True)))

                        elif q == q_k:
                            b_k = np.radians(ase.geometry.get_dihedrals_derivatives(u, v, w)[0][2]) * Bohr
                            b[row : row + 3, column] = b_k
                            u = np.copy(
                                np.atleast_2d(mol.get_distance(q_i, q_j, mic=True, vector=True))
                            )  #####copy needed because derivative function rewrites vector variable as normed vector
                            w = np.copy(np.atleast_2d(mol.get_distance(q_k, q_l, mic=True, vector=True)))
                            v = np.copy(np.atleast_2d(mol.get_distance(q_j, q_k, mic=True, vector=True)))

                        elif q == q_l:
                            b_k = np.radians(ase.geometry.get_dihedrals_derivatives(u, v, w)[0][3]) * Bohr
                            b[row : row + 3, column] = b_k
                            u = np.copy(
                                np.atleast_2d(mol.get_distance(q_i, q_j, mic=True, vector=True))
                            )  #####copy needed because derivative function rewrites vector variable as normed vector
                            w = np.copy(np.atleast_2d(mol.get_distance(q_k, q_l, mic=True, vector=True)))
                            v = np.copy(np.atleast_2d(mol.get_distance(q_j, q_k, mic=True, vector=True)))
                row += 3
            column += 1

        self.B = np.transpose(b)

    def get_energies(self) -> List[float]:
        """Calls the energies of the Atoms objects.

        Returns:
            [energy difference, energy of atoms0, energy of atomsF]

        """
        e0 = self.atoms0.calc.get_potential_energy()
        eF = self.atomsF.calc.get_potential_energy()
        if not self.ase_units:
            e0 *= mol / kcal
            eF *= mol / kcal
        deltaE = eF - e0
        self.energies = [deltaE, eF, e0]
        return [deltaE, eF, e0]

    def get_delta_q(self):
        """get the strain in RICs substracts the values of the relaxed structure from the strained structure

        Returns:
            2D array of the values.
        """

        try:
            len(self.rim_list)
        # FIXME: what happens here?
        except Exception:
            self.get_common_rims()

        if len(self.B) == 0:
            self.get_b_matrix()
        q0 = []
        qF = []
        dq_da = []

        # for loops for all redunant internal coordinates

        # bonds
        for q in self.rim_list[0]:
            q0.append(self.atoms0.get_distance(int(q[0]), int(q[1]), mic=True) / Bohr)
            qF.append(self.atomsF.get_distance(int(q[0]), int(q[1]), mic=True) / Bohr)
        # custom bonds
        for q in self.rim_list[1]:
            q0.append(self.atoms0.get_distance(int(q[0]), int(q[1]), mic=True) / Bohr)
            qF.append(self.atomsF.get_distance(int(q[0]), int(q[1]), mic=True) / Bohr)
        # angles
        for q in self.rim_list[2]:
            q0.append(np.radians(self.atoms0.get_angle(int(q[0]), int(q[1]), int(q[2]), mic=True)))
            qF.append(np.radians(self.atomsF.get_angle(int(q[0]), int(q[1]), int(q[2]), mic=True)))
        # dihedral angles
        for q in self.rim_list[3]:
            q0_preliminary = np.radians(self.atoms0.get_dihedral(int(q[0]), int(q[1]), int(q[2]), int(q[3]), mic=True))
            qF_preliminary = np.radians(self.atomsF.get_dihedral(int(q[0]), int(q[1]), int(q[2]), int(q[3]), mic=True))

            # get the smallest absolute value of the two possible rotational directions
            dda = qF_preliminary - q0_preliminary
            if 2 * np.pi - abs(dda) < abs(dda):
                dda = (2 * np.pi - abs(dda)) * -np.sign(dda)
            dq_da.append(dda)

        delta_q = np.subtract(qF, q0)

        delta_q = np.append(delta_q, dq_da)

        self.delta_q = delta_q

        self.qF = qF
        self.q0 = q0

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

        if len(self.proc_E_RIMs) == 0:
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
        des_colors: Optional[Dict] = None,
        box: bool = False,
        man_strain: Optional[float] = None,
        modus: Optional[str] = None,
        colorbar: bool = True,
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
        self.indices = np.arange(0, len(self.atoms0)).tolist()
        self.get_hessian()
        if 3 * len(indices) < len(self.H):
            raise ValueError("to little indices for the given hessian")

        cbonds_flag = False
        if self.custom_bonds is not None:
            custom_bonds = self.custom_bonds.copy()
            cbonds_flag = True
            self.custom_bonds = self.custom_bonds[np.isin(self.custom_bonds, indices).all(axis=1)]

        self.rim_list = self.get_common_rims()

        rim_list = self.rim_list
        if len(rim_list) == 0:
            raise ValueError("Chosen indexlist has no rims")

        self.get_b_matrix(indices=self.indices)
        # set B matrix values of not considered atoms to 0
        for i in range(len(self.H)):
            if i not in indices:
                self.B[:, i * 3 : i * 3 + 3] = 0
        ind = np.array([[i * 3, i * 3 + 1, i * 3 + 2] for i in indices]).ravel()
        self.B = np.take(self.B, ind, axis=1)

        self.get_delta_q()

        try:
            all_E_geometries = self.get_energies()
        except Exception:
            all_E_geometries = self.energies
        E_geometries = all_E_geometries[0]

        (
            self.proc_E_RIMs,
            self.E_RIMs,
            self.E_RIMs_total,
            proc_geom_RIMs,
            self.delta_q,
        ) = jedi_analysis(
            self.atomsF,
            rim_list,
            self.B,
            self.H,
            self.delta_q,
            E_geometries,
            ase_units=ase_units,
        )
        # get values of rims inside the substructure
        self.post_process(indices)
        self.E_RIMs_total = sum(self.E_RIMs)
        proc_geom_RIMs = 100 * (sum(self.E_RIMs) - E_geometries) / E_geometries
        reporting.jedi_printout(
            self.atoms0,
            self.rim_list,
            self.delta_q,
            E_geometries,
            self.E_RIMs_total,
            proc_geom_RIMs,
            self.proc_E_RIMs,
            self.E_RIMs,
            ase_units=ase_units,
        )

        if cbonds_flag:
            self.custom_bonds = custom_bonds  # restore the user input

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
        self.indices = indices
        rim_list = self.rim_list
        cbonds_flag = False
        if self.custom_bonds is not None:
            custom_bonds = self.custom_bonds.copy()
            cbonds_flag = True
            self.custom_bonds = self.custom_bonds[np.isin(self.custom_bonds, indices).all(axis=1)]
        rim_p = self.get_common_rims()  # get rimlist of substructure

        ind = []
        rim_list_c = []  # preparing for stacking rim_list to be able to use np.unique

        for i in range(4):  # rim_list is always of length 4
            if rim_list[i].shape == (0,):
                rim_list_c.append([])
            else:
                if rim_p[i].shape[0] > 0:
                    rim_list_c.append(np.vstack((rim_list[i], rim_p[i])))
                else:
                    rim_list_c.append(np.vstack(rim_list[i]))
            _, z = np.unique(rim_list_c[-1], return_counts=True, axis=0)

            ind.append(np.where(z > 1)[0])  # get indices where ric is in both sets
        for i in range(4):
            ind[i] = ind[i] + np.sum([p.shape[0] for p in rim_list[0:i]])  # get correct indices for the stacked array
        ind = np.hstack(ind)
        ind = ind.astype(int)

        self.E_RIMs = np.array(self.E_RIMs)[ind]
        self.delta_q = self.delta_q[ind]
        E_RIMs_total = sum(self.E_RIMs)
        self.proc_E_RIMs = np.array(self.E_RIMs) / E_RIMs_total * 100
        if cbonds_flag:
            self.custom_bonds = custom_bonds  # restore the user input
        pass

    def add_custom_bonds(self, bonds: NDArray) -> None:
        """Add custom bonds after creating the object.

        Args:
            bonds:
                1D or 2Darray with atom indices, [[i,j]...]
        """

        self.custom_bonds = np.atleast_2d(bonds)  # additional bonds for analysis of non-covalent interactions

    def set_bond_params(self, covf=1.3, vdwf=0.9):
        """
        Args:
            covf:
                float factor for  covalent radii to determine covalent bonds
            vdwf:
                float factor for vdw radii to get the upper limit of the custom bond lengths
        """
        self.covf = covf
        self.vdwf = vdwf


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
    # jedi analysis function
    ###########################
    ##  Matrix Calculations  ##
    ###########################
    B_transp = np.transpose(B)
    # Calculate the number of RIMs (= number of rows in the B-Matrix), equivalent to number of redundant internal coordinates

    # Calculate the pseudoinverse of the B-Matrix and its transposed (take care of diatomic molecules specifically)
    if B.ndim == 1:
        B_plus = B_transp / 2
        B_transp_plus = B / 2
    else:
        B_plus = np.linalg.pinv(B, 0.0001)
        B_transp_plus = np.linalg.pinv(np.transpose(B), 0.0001)

    # Calculate the P-Matrix (eq. 4 in Helgaker's paper)
    P = np.dot(B, B_plus)

    #############################################
    # JEDI analysis	        	#
    #############################################

    # Calculate the Hessian in RIMs (take care to get the correct multiplication for a diatomic molecule
    if B.ndim == 1:
        H_q = B_transp_plus.dot(H_cart).dot(B_plus)
    else:
        H_q = P.dot(B_transp_plus).dot(H_cart).dot(B_plus).dot(P)

    # Calculate the total energies in RIMs and its deviation from E_geometries
    E_RIMs_total = 0.5 * np.transpose(delta_q).dot(H_q).dot(delta_q)

    # Get the energy stored in every RIM (take care to get the right multiplication for a diatomic molecule)

    if B.ndim == 1:
        E_RIMs = np.array([0.5 * delta_q[0] * H_q * delta_q[0]])

    else:
        E_RIMs = np.sum(0.5 * (delta_q * H_q).T * delta_q, axis=1)
    # Get the percentage of the energy stored in every RIM

    proc_E_RIMs = 100 * E_RIMs / E_RIMs_total

    if ase_units:
        b = np.shape(rim_list[0])[0] + np.shape(rim_list[1])[0]  # border between lengths and angles
        delta_q[0:b] *= Bohr
        delta_q[b::] = np.degrees(delta_q[b::])
        E_RIMs = np.array(E_RIMs) * Hartree
        E_RIMs_total *= Hartree
    else:
        E_RIMs = np.array(E_RIMs) / kcal * mol * Hartree
        E_RIMs_total *= mol / kcal * Hartree

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


def get_hbonds(mol, covf=1.3, vdwf=0.9):
    """
    Get all hbonds in a structure.
    Hbonds are defined as the HY bond inside X-H···Y where X and Y can be O, N, F and the angle XHY is larger than 90° and the distance between HY is shorter than 0.9 times the sum of the vdw radii of H and Y.

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
