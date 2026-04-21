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
