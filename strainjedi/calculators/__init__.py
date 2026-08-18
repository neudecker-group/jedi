"""ASE calculator construction and job driving for the supported programs.

Separate from :mod:`strainjedi.io` on purpose. Reading an output file and running a
calculation are different jobs with different dependencies: the parsers are plain NumPy and
know nothing about ASE, while everything here is ASE glue. Setting up and driving
calculations is really ASE's responsibility, so this package is expected to shrink over time.
"""
