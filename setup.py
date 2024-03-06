import os
from setuptools import setup, find_packages

setup(
    name = "jedi",
    version = "0.0.1",
    description = ("Judgement of Energy Analysis"),
    license = "MIT",
    packages=find_packages(),
    package_dir={"": "jedi"},
    install_requires=[
        "pytest>=6.1.0",
        "ase>=3.22.1"
        "numpy"
    ]
)