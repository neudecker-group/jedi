from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import ase.units

if TYPE_CHECKING:
    # imported only by type checkers, never at runtime to avoid cyclic imports
    from strainjedi.jedi import Jedi

import numpy as np
from ase import atoms

from strainjedi import __version__, quotes


@dataclass(frozen=True, slots=True)
class ReportLayout:
    """
    Formatting/layout settings for JEDI text reports.

    This dataclass controls the *presentation* of the JEDI report only:
    overall line width, which columns are shown, and how headers/rows are
    formatted for printing or writing to a file.

    Parameters
    ----------
    total_width:
        Target character width of the generated report (banner, headers, rows).
        Column widths are derived from this value.
        Default 120.

    columns:
        Ordered column keys to render for the RIC table. The order here is the
        order in the output.

        - ``"no"``: 1-based row number
        - ``"type"``: RIC type label (e.g. bond/custom/angle/dihedral)
        - ``"indices"``: atom indices rendered with element symbols
        - ``"delta_q"``: change in internal coordinate
        - ``"percentage"``: percent contribution of the RIC to the total
        - ``"energy"``: energy contribution of the RIC

        Default is all of the above.

    use_ase_units:
        If True, headers use ASE-friendly units (Å/° and eV). If False, headers
        use the alternative labels used by the legacy output (a.u. and kcal/mol).
        Default True.

    Examples
    --------
    Default layout (full table including ``delta_q``)::

        layout = ReportLayout()

    Bonds-only layout (like ``strainjedi.reporting.jedi_printout_bonds``)::

        layout = ReportLayout(columns=("no", "type", "indices", "percentage", "energy"))

    Compact layout, using a.u. and kcal/mol::

        layout = ReportLayout(total_width=90, columns=("no", "indices", "energy"), use_ase_units=False)
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
    use_ase_units: bool = True

    @property
    def header(self) -> dict[str, int]:
        return {"header": self.total_width}

    @property
    def energy_comparison(self) -> dict[str, int]:
        return {
            "label": int(self.total_width / 6),
            "energy": int(self.total_width / 5),
            "deviation": int(self.total_width / 5),
        }

    @property
    def column_titles(self) -> dict[str, str]:
        return {
            "no": "RIC No.",
            "type": "RIC type",
            "indices": "indices",
            "delta_q": "delta_q (Å,°)" if self.use_ase_units else "delta_q (a.u.)",
            "percentage": "Percentage",
            "energy": "Energy (eV)" if self.use_ase_units else "Energy (kcal/mol)",
        }

    @property
    def column_widths(self) -> dict[str, int]:
        width = int(self.total_width / max(1, len(self.columns)))
        return {key: width for key in self.columns}

    def format_value(self, key: str, value: Any) -> str:
        if key in ("delta_q", "energy"):
            return f"{value:.4f}"
        if key == "percentage":
            return f"{value:.2f}"
        return str(value)

    def format_row(self, row: Mapping[str, Any]) -> str:
        widths = self.column_widths
        return "".join(f"{self.format_value(k, row[k]):^{widths[k]}}" for k in self.columns)

    def format_header(self) -> str:
        widths = self.column_widths
        titles = self.column_titles
        return "".join(f"{titles[k]:^{widths[k]}}" for k in self.columns)

    def render_banner(self) -> list[str]:
        width = self.total_width
        border = "*" * width
        return [
            border,
            f"*{'JEDI ANALYSIS':^{width - 2}}*",
            f"*{'Judgement of Energy DIstribution':^{width - 2}}*",
            border,
            f"{f'version {__version__}':^{width}}",
        ]


@dataclass(frozen=True, slots=True)
class _EnergySummary:
    ab_initio: float
    jedi_total: float
    deviation_percent: float


@dataclass(frozen=True, slots=True)
class _RicTable:
    """
    Holds the flattened RIC rows input data.

    `types` and `indices` must be aligned and define the row ordering.
    Other arrays (delta_q/percentage/energy) must match that same ordering.
    """

    types: list[str]
    indices: list[np.ndarray]
    percentage: np.ndarray
    energy: np.ndarray
    delta_q: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = len(self.types)
        if len(self.indices) != n:
            raise ValueError("types and indices must have the same length")
        if self.percentage.shape[0] != n:
            raise ValueError("percentage length must match row count")
        if self.energy.shape[0] != n:
            raise ValueError("energy length must match row count")
        if self.delta_q is not None and self.delta_q.shape[0] != n:
            raise ValueError("delta_q length must match row count")


def _energy_comparison_lines(layout: ReportLayout, ase_units: bool, summary: _EnergySummary) -> list[str]:
    title = "Strain Energy (eV)" if ase_units else "Strain Energy (kcal/mol)"
    fmt = layout.energy_comparison
    return [
        "{0:>{label}}{1:^{energy}}{2:^{deviation}}".format(" ", title, "Deviation (%)", **fmt),
        "{0:<{label}}{1:^{energy}.4f}{2:^{deviation}}".format("Ab Initio", summary.ab_initio, "-", **fmt),
        "{0:<{label}}{1:^{energy}.4f}{2:^{deviation}.2f}".format(
            "JEDI", summary.jedi_total, summary.deviation_percent, **fmt
        ),
    ]


def _format_indices(atoms_obj: atoms.Atoms, ric_type: str, idx: np.ndarray) -> str:
    # Note: idx is expected to be a 1D array of atom indices (len 2/3/4).
    if ric_type in ("bond", "custom"):
        return f"{atoms_obj.symbols[idx[0]]}{idx[0]}  {atoms_obj.symbols[idx[1]]}{idx[1]}"
    if ric_type == "angle":
        return (
            f"{atoms_obj.symbols[idx[0]]}{idx[0]} "
            f"{atoms_obj.symbols[idx[1]]}{idx[1]} "
            f"{atoms_obj.symbols[idx[2]]}{idx[2]}"
        )
    return (
        f"{atoms_obj.symbols[idx[0]]}{idx[0]} "
        f"{atoms_obj.symbols[idx[1]]}{idx[1]} "
        f"{atoms_obj.symbols[idx[2]]}{idx[2]} "
        f"{atoms_obj.symbols[idx[3]]}{idx[3]}"
    )


def _flatten_rics(rim_list: list, ric_type_map: Mapping[int, str]) -> tuple[list[str], list[np.ndarray]]:
    types: list[str] = []
    indices: list[np.ndarray] = []
    for ric_type, name in ric_type_map.items():
        for idx in rim_list[ric_type]:
            types.append(name)
            indices.append(idx)
    return types, indices


def _table_lines(
    layout: ReportLayout,
    atoms_obj: atoms.Atoms,
    table: _RicTable,
) -> list[str]:
    lines: list[str] = [layout.format_header()]

    for i, (ric_type, idx) in enumerate(zip(table.types, table.indices)):
        row: dict[str, Any] = {
            "no": i + 1,
            "type": ric_type,
            "indices": _format_indices(atoms_obj, ric_type, idx),
            "percentage": table.percentage[i],
            "energy": table.energy[i],
        }
        if "delta_q" in layout.columns and table.delta_q is not None:
            row["delta_q"] = table.delta_q[i]
        lines.append(layout.format_row(row))

    return lines


def _build_report(
    *,
    atoms_obj: atoms.Atoms,
    ase_units: bool,
    layout: ReportLayout,
    energy: _EnergySummary,
    table: _RicTable,
) -> str:
    lines: list[str] = ["\n"]
    lines.extend(layout.render_banner())
    lines.extend(_energy_comparison_lines(layout, ase_units, energy))
    lines.extend(_table_lines(layout, atoms_obj, table))
    return "\n".join(lines)


def jedi_printout(
    atoms_obj: atoms.Atoms,
    rim_list: list,
    delta_q: np.ndarray,
    E_geometries: float,
    E_RIMs_total: float,
    proc_geom_RIMs: float,
    proc_E_RIMs: np.ndarray,
    E_RIMs: np.ndarray,
    ase_units: bool = False,
    layout: ReportLayout | None = None,
) -> None:
    """
    Printout of analysis of stored strain energy in redundant internal coordinates.

    atoms_obj: ASE Atoms
        Used to determine the atomic species of the indices.
    rim_list: list
        A list of 4 numpy 2D arrays: bonds, custom bonds, angles, dihedrals.
    """
    layout = layout or ReportLayout(use_ase_units=ase_units)

    if ase_units:
        # Convert units to Angstrom and degrees if necessary (ASE units)
        b = rim_list[0].shape[0] + rim_list[1].shape[0]
        delta_q[0:b] *= ase.units.Bohr
        delta_q[b:] = np.degrees(delta_q[b:])
        E_RIMs = E_RIMs * ase.units.Hartree
        E_RIMs_total *= ase.units.Hartree
    else:
        E_RIMs = E_RIMs / ase.units.kcal * ase.units.mol * ase.units.Hartree
        E_RIMs_total *= ase.units.mol / ase.units.kcal * ase.units.Hartree
        E_geometries *= ase.units.mol / ase.units.kcal

    types, indices = _flatten_rics(rim_list, {0: "bond", 1: "custom", 2: "angle", 3: "dihedral"})
    report = _build_report(
        atoms_obj=atoms_obj,
        ase_units=ase_units,
        layout=layout,
        energy=_EnergySummary(E_geometries, E_RIMs_total, proc_geom_RIMs),
        table=_RicTable(
            types=types,
            indices=indices,
            delta_q=delta_q,
            percentage=proc_E_RIMs,
            energy=E_RIMs,
        ),
    )

    print(report)
    print(quotes.random_quote())


def jedi_printout_bonds(
    atoms_obj: atoms.Atoms,
    rim_list: list,
    E_geometries: float,
    E_RIMs_total: float,
    proc_geom_RIMs: float,
    proc_E_RIMs: np.ndarray,
    E_RIMs: np.ndarray,
    ase_units: bool = False,
    file: str = "total",
    layout: ReportLayout | None = None,
) -> None:
    """
    Printout of analysis of stored strain energy in the bonds.

    atoms_obj: ASE Atoms
        Used to determine the atomic species of the indices.
    rim_list: list
        A list/array where entry 0 contains bonds and entry 1 contains custom bonds.
    """
    layout = layout or ReportLayout(columns=("no", "type", "indices", "percentage", "energy"))

    types, indices = _flatten_rics(rim_list, {0: "bond", 1: "custom"})
    report = _build_report(
        atoms_obj=atoms_obj,
        ase_units=ase_units,
        layout=layout,
        energy=_EnergySummary(E_geometries, E_RIMs_total, proc_geom_RIMs),
        table=_RicTable(
            types=types,
            indices=indices,
            percentage=proc_E_RIMs,
            energy=E_RIMs,
            delta_q=None,
        ),
    )

    with open(file, "w") as f:
        f.write(report)


def report(jedi: Jedi, *, layout: ReportLayout | None = None, file: str = "jedi_analysis.txt"):
    """
    Create a report and save it to `file`.
    """
    from ase.io.jsonio import encode

    layout = layout or ReportLayout()
    with open(file, "w") as f:
        f.write("WIP. Meanwhile, here's a JSON dump of Jedi:\n")
        f.write(json.dumps(encode(jedi), indent=2))
