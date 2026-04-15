from dataclasses import dataclass

import numpy as np
from ase import atoms
from typing_extensions import Dict, List

from strainjedi import __version__, quotes


@dataclass(frozen=True, slots=True)
class ReportLayout:
    """
    Formatting/layout settings for JEDI text reports.
    Usage:
        layout = ReportLayout()
    """

    total_width: int = 120
    columns: tuple[str, ...] = (
        "no",
        "type",
        "indices",
        "delta_q",
        "percentage",
        "energy",
    )

    @property
    def header(self) -> Dict[str, int]:
        return {"header": self.total_width}

    @property
    def energy_comparison(self) -> Dict[str, int]:
        return {
            "label": int(self.total_width / 6),
            "energy": int(self.total_width / 5),
            "deviation": int(self.total_width / 5),
        }

    def column_titles(self, use_ase_units: bool) -> Dict[str, str]:
        return {
            "no": "RIC No.",
            "type": "RIC type",
            "indices": "indices",
            "delta_q": "delta_q (Å,°)" if use_ase_units else "delta_q (a.u.)",
            "percentage": "Percentage",
            "energy": "Energy (eV)" if use_ase_units else "Energy (kcal/mol)",
        }

    def column_widths(self) -> Dict[str, int]:
        width = int(self.total_width / max(1, len(self.columns)))
        return {key: width for key in self.columns}

    def format_value(self, key: str, value) -> str:
        if key == "delta_q":
            return f"{value:.4f}"
        if key == "percentage":
            return f"{value:.2f}"
        if key == "energy":
            return f"{value:.4f}"
        return str(value)

    def format_row(self, row: Dict[str, object]) -> str:
        widths = self.column_widths()
        parts: list[str] = []
        for key in self.columns:
            value = self.format_value(key, row[key])
            width = widths[key]
            parts.append(f"{value:^{width}}")
        return "".join(parts)

    def format_header(self, use_ase_units: bool) -> str:
        widths = self.column_widths()
        titles = self.column_titles(use_ase_units)
        parts: list[str] = []
        for key in self.columns:
            title = titles[key]
            width = widths[key]
            parts.append(f"{title:^{width}}")
        return "".join(parts)


def _render_banner(output: list[str], layout: ReportLayout) -> None:
    width = layout.total_width
    border = "*" * width
    output.append(border)
    output.append(f"*{'JEDI ANALYSIS':^{width - 2}}*")
    output.append(f"*{'Judgement of Energy DIstribution':^{width - 2}}*")
    output.append(border)
    output.append(f"{f'version {__version__}':^{width}}")


def _render_energy_comparison(
    output: list[str],
    layout: ReportLayout,
    ase_units: bool,
    E_geometries: float,
    E_RIMs_total: float,
    proc_geom_RIMs: float,
) -> None:
    title = "Strain Energy (eV)" if ase_units else "Strain Energy (kcal/mol)"
    output.append(
        "{0:>{label}}{1:^{energy}}{2:^{deviation}}".format(
            " ", title, "Deviation (%)", **layout.energy_comparison
        )
    )
    output.append(
        "{0:<{label}}{1:^{energy}.4f}{2:^{deviation}}".format(
            "Ab Initio", E_geometries, "-", **layout.energy_comparison
        )
    )
    output.append(
        "{0:<{label}}{1:^{energy}.4f}{2:^{deviation}.2f}".format(
            "JEDI", E_RIMs_total, proc_geom_RIMs, **layout.energy_comparison
        )
    )


def _format_indices(atoms_obj: atoms.Atoms, rim: str, k: np.ndarray) -> str:
    if rim in ("bond", "custom"):
        return f"{atoms_obj.symbols[k[0]]}{k[0]}  {atoms_obj.symbols[k[1]]}{k[1]}"
    if rim == "angle":
        return (
            f"{atoms_obj.symbols[k[0]]}{k[0]} "
            f"{atoms_obj.symbols[k[1]]}{k[1]} "
            f"{atoms_obj.symbols[k[2]]}{k[2]}"
        )
    return (
        f"{atoms_obj.symbols[k[0]]}{k[0]} "
        f"{atoms_obj.symbols[k[1]]}{k[1]} "
        f"{atoms_obj.symbols[k[2]]}{k[2]} "
        f"{atoms_obj.symbols[k[3]]}{k[3]}"
    )


def _append_ric_rows(
    output: list[str],
    layout: ReportLayout,
    atoms_obj: atoms.Atoms,
    rim_list: List,
    rics_dict: Dict[int, str],
    delta_q: np.ndarray | None,
    proc_E_RIMs: np.ndarray,
    E_RIMs: np.ndarray,
) -> None:
    ric_counter = 0
    for ric_type, rim in rics_dict.items():
        for k in rim_list[ric_type]:
            row = {
                "no": ric_counter + 1,
                "type": rim,
                "indices": _format_indices(atoms_obj, rim, k),
                "percentage": proc_E_RIMs[ric_counter],
                "energy": E_RIMs[ric_counter],
            }
            if "delta_q" in layout.columns and delta_q is not None:
                row["delta_q"] = delta_q[ric_counter]
            output.append(layout.format_row(row))
            ric_counter += 1


def jedi_printout(
    atoms_obj: atoms.Atoms,
    rim_list: List,
    delta_q: np.ndarray,
    E_geometries: float,
    E_RIMs_total: float,
    proc_geom_RIMs: float,
    proc_E_RIMs: List,
    E_RIMs: np.ndarray,
    ase_units: bool = False,
    layout: ReportLayout | None = None,
):
    """
    Printout of analysis of stored strain energy in redundant internal coordinates.

    atoms_obj: ASE Atoms
        Used to determine the atomic species of the indices.
    rim_list: list
        A list of 4 numpy 2D arrays: bonds, custom bonds, angles, dihedrals.
    """
    layout = layout or ReportLayout()

    output: list[str] = []
    output.append("\n")
    _render_banner(output, layout)
    _render_energy_comparison(
        output, layout, ase_units, E_geometries, E_RIMs_total, proc_geom_RIMs
    )

    output.append(layout.format_header(ase_units))
    _append_ric_rows(
        output,
        layout,
        atoms_obj,
        rim_list,
        {0: "bond", 1: "custom", 2: "angle", 3: "dihedral"},
        delta_q,
        proc_E_RIMs,
        E_RIMs,
    )

    print("\n".join(output))
    print(quotes.random_quote())


def jedi_printout_bonds(
    atoms_obj: atoms.Atoms,
    rim_list: np.ndarray,
    E_geometries: float,
    E_RIMs_total: float,
    proc_geom_RIMs: float,
    proc_E_RIMs: np.ndarray,
    E_RIMs: np.ndarray,
    ase_units: bool = False,
    file: str = "total",
    layout: ReportLayout | None = None,
):
    """
    Printout of analysis of stored strain energy in the bonds.

    atoms_obj: ASE Atoms
        Used to determine the atomic species of the indices.
    rim_list: list
        A list/array where entry 0 contains bonds and entry 1 contains custom bonds.
    """
    layout = layout or ReportLayout(
        columns=("no", "type", "indices", "percentage", "energy")
    )

    output: list[str] = []
    output.append("\n")
    _render_banner(output, layout)
    _render_energy_comparison(
        output, layout, ase_units, E_geometries, E_RIMs_total, proc_geom_RIMs
    )

    output.append(layout.format_header(ase_units))
    _append_ric_rows(
        output,
        layout,
        atoms_obj,
        rim_list,
        {0: "bond", 1: "custom"},
        None,
        proc_E_RIMs,
        E_RIMs,
    )

    with open(file, "w") as f:
        f.writelines("\n".join(output))
