"""ASE glue for driving ORCA: geometry-optimisation dynamics and result reading.

This is calculator plumbing, not parsing -- see :mod:`strainjedi.io` for the latter. It lives
here so that :mod:`strainjedi.io` can stay a pure, ASE-free parser layer, and it is expected
to move upstream into ASE eventually.

The ``ORCA`` calculator class that used to live here was deleted: it subclassed the old
``FileIOCalculator``, whereas ASE 3.28's ORCA calculator is a ``GenericFileIOCalculator``, so
it could not be constructed at all. Use ``ase.calculators.orca.ORCA`` instead --
:func:`strainjedi.calculators.build.build_calc` does.
"""

from collections.abc import Iterable
from typing import Dict, Optional

from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.units import Hartree


def read(filename):
    """Read orca outputs.

    filename: str"""
    energy = None
    with open(filename, "r") as fileobj:
        lineiter = iter(fileobj)
        for line in lineiter:
            if "CARTESIAN COORDINATES (ANGSTROEM)" in line:
                positions = []
                symbols = ""
                next(lineiter)

                while True:
                    line = next(lineiter).split()

                    if line == []:
                        break
                    symbols += line[0]
                    positions.append([float(word) for word in line[1:]])

            elif "FINAL SINGLE POINT ENERGY" in line:
                convert = Hartree
                energy = float(line.split()[4]) * convert

        atoms = Atoms(symbols, positions=positions)
        atoms.set_positions(positions)

        atoms.calc = SinglePointCalculator(atoms, energy=energy)

    return atoms


class OrcaDynamics:
    calctype = "optimizer"
    delete = ["force"]
    keyword: Optional[str] = None
    special_keywords: Dict[str, str] = dict()

    def __init__(self, atoms, calc=None):
        """
        atoms: class
            Structure with orca calculator"""
        self.atoms = atoms
        if calc is not None:
            self.calc = calc
        else:
            if self.atoms.calc is None:
                raise ValueError("{} requires a valid ORCA calculator object!".format(self.__class__.__name__))

            self.calc = self.atoms.calc

    def todict(self):
        return {"type": self.calctype, "optimizer": self.__class__.__name__}

    def delete_keywords(self, kwargs):
        """removes list of keywords (delete) from kwargs"""
        for d in self.delete:
            kwargs.pop(d, None)

    def set_keywords(self, kwargs):
        args = kwargs.pop(self.keyword, [])

        if isinstance(args, str):
            args = [args]
        elif isinstance(args, Iterable):
            args = list(args)

        for key, template in self.special_keywords.items():
            if key in kwargs:
                val = kwargs.pop(key)
                args.append(template.format(val))

        kwargs["jobtype"] = "opt"

    def run(self, **kwargs):

        self.delete_keywords(kwargs)
        self.delete_keywords(self.calc.parameters)
        self.set_keywords(kwargs)

        self.calc.set(**self.keywords)

        self.atoms.calc = self.calc

        try:
            self.atoms.get_potential_energy()
        except OSError:
            converged = False
        else:
            converged = True

        atoms = read(self.calc.label + ".out")

        self.atoms.positions = atoms.positions

        self.atoms.calc = atoms.calc

        return converged


class OrcaOptimizer(OrcaDynamics):
    """
    Allowing ase to use Orca geometry optimizations

    """

    keywords = {"task": "opt"}
