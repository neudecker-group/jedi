"""
strainjedi.

The JEDI analysis python library.
"""

__author__ = "Tim Neudecker, Sanna Benter, Henry Wang, Wilke Dononelli"
__credits__ = "Neudecker Group"
__version__ = "1.1.0"

# This needs to be here to avoid circular imports/partially initialised module errors.

from strainjedi.jedi import Jedi

# support the following statements:
#  from strainjedi import *
#  from strainjedi import Jedi
#
# Of course, users can still use the old way:
#  from strainjedi.jedi import Jedi
#
# But this __all__ directive is pretty neat.
__all__ = ["Jedi"]
