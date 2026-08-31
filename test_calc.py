#!/usr/bin/env python3
"""Era-0 benchmark (new ASE-JEDI): time 'arrays -> colorable bond data' per structure."""

import json, time
import numpy as np
from pathlib import Path
from ase import Atoms
from ase.io import read
from ase.vibrations import VibrationsData
from ase.calculators.singlepoint import SinglePointCalculator
from strainjedi import Jedi
from strainjedi.visualization import ColorMapper

DATADIR = Path("/mnt/programs/rawsita/dev-jedi/strainjedi/tests/data/input_files")
OUT = Path("/mnt/programs/rawsita/dev-jedi/strainjedi/tools/bench/results/era0_new_3.json")
ALKANES = [
    (1, "methane"),
    (2, "ethane"),
    (3, "propane"),
    (4, "butane"),
    (5, "pentane"),
    (6, "hexane"),
    (7, "heptane"),
    (8, "octane"),
    (9, "nonane"),
    (10, "decane"),
    (11, "undecane"),
    (12, "dodecane"),
    (13, "tridecane"),
    (14, "tetradecane"),
    (15, "pentadecane"),
    (16, "hexadecane"),
    (17, "heptadecane"),
    (18, "octadecane"),
    (19, "nonadecane"),
    (20, "icosane"),
    (21, "henicosane"),
    (22, "docosane"),
    (23, "tricosane"),
    (24, "tetracosane"),
    (25, "pentacosane"),
    (26, "hexacosane"),
    (27, "heptacosane"),
    (28, "octacosane"),
    (29, "nonacosane"),
    (30, "triacontane"),
]
MODE_LIST = ["all"]  # match C++ colour_bonds' single all-RIC coloring
REPS = 5


def load_raw(stem):
    """One-time I/O, kept OUT of the timed region."""
    m1 = read(DATADIR / f"{stem}.xyz")
    m2 = read(DATADIR / f"{stem}_distorted.xyz")
    H = np.loadtxt(DATADIR / f"{stem}_full_hessian.txt")
    return m1.get_chemical_symbols(), m1.get_positions(), m2.get_positions(), H


def timed_once(symbols, pos1, pos2, H):
    """Everything from in-memory arrays to colorable data — the measured pipeline."""
    t0 = time.perf_counter()
    mol1 = Atoms(symbols=symbols, positions=pos1)
    mol1.calc = SinglePointCalculator(mol1, energy=-2.0)  # static dummy energy, no calc
    mol2 = Atoms(symbols=symbols, positions=pos2)
    mol2.calc = SinglePointCalculator(mol2, energy=-1.0)
    hessian = VibrationsData.from_2d(mol1, H)
    j = Jedi(mol1, mol2, hessian)
    j.run(printout=False)
    ColorMapper(j, False).get_visualization_data(MODE_LIST)
    return time.perf_counter() - t0


OUT.parent.mkdir(parents=True, exist_ok=True)
results = []
for c, name in ALKANES:
    stem = f"{c:02d}-{name}"
    raw = load_raw(stem)
    n_atoms = len(raw[0])
    best = min(timed_once(*raw) for _ in range(REPS))
    results.append({"version": "python-new", "molecule": name, "carbons": c, "n_atoms": n_atoms, "seconds": best})
    print(f"{stem:18s} {n_atoms:3d} atoms  {best * 1e3:9.2f} ms")

OUT.write_text(json.dumps(results, indent=2))
print(f"\nwrote {OUT}")
