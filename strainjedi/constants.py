COVALENCY_FACTOR = 1.3
"""
Bonds are assigned where the atoms distance is less than or equal to 
COVALENCY_FACTOR times the sum of their covalent radii. Set to 1.3,
See Bakken et al: https://doi.org/10.1063/1.1515483
"""

VAN_DER_WAALS_FACTOR = 0.9
"""
Similar to COVALENCY_FACTOR, only consider candiates where the distance is
less than or equal to this factor times the sum of their van der Waals radii.
Set to 0.9 (< 1) to ensure we are inside the respective bond partners'
sphere of influences, without considering potential partners that are realistically
too far apart.
See https://doi.org/10.1063/5.0199247
"""

HARTREE_EV = 27.211386024367243
"""
One Hartree in eV. Deliberately copied from ``ase.units.Hartree`` rather than taken from a
CODATA table: ASE derives its value from ``_hbar/_e/_me/_eps0`` with
``__codata_version__ = '2014'``, which differs from published CODATA 2018
(27.211386245988) in the eighth significant digit. Using ASE's number keeps results
identical while the parsers are being made ASE-free. If this is ever updated, the stored
test data must be regenerated in the same commit.
"""

BOHR_ANG = 0.5291772105638411
"""
One Bohr radius in Angstrom. Copied from ``ase.units.Bohr`` for the same reason as
:data:`HARTREE_EV`; published CODATA 2018 gives 0.529177210903.
"""

HESSIAN_AU_TO_ASE = HARTREE_EV / BOHR_ANG**2
"""
Converts a Hessian in Hartree/Bohr^2 (what quantum chemistry programs print, and what the
parsers return) to eV/Angstrom^2 (what ``ase.vibrations.VibrationsData`` is defined in).
Equals 97.1736236254206.
"""
