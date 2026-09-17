"""External benchmark conversion and evaluation boundaries.

The package root intentionally imports nothing.  Legacy evaluator processes
must be able to import ``clearvla.benchmarks.bridge`` without installing HDF5,
dataset conversion or split-manifest dependencies.  Conversion helpers remain
available from their explicit modules.
"""

__all__: list[str] = []


