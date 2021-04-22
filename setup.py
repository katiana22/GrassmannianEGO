import os
import sys
from setuptools import setup, find_packages
PACKAGE_NAME = 'GrassmannianEGO'
MINIMUM_PYTHON_VERSION = 3, 5

setup(
    name='GrassmannianEGO',
    version=read_package_variable('__version__'),
    description='Codes for implementing atomistic-informed calibration of partial differential equations (PDEs) with manifold learning and Bayesian optimization',
    author='Katiana Kontolati, Darius Alix-Williams, Nicholas M. Boffi, Michael L. Falk, Chris H. Rycroft, Michael D. Shields',
    author_email='kontolati@jhu.edu',
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'sklearn', 'matplotlib', 'itertools', 'UQpy', 'chaospy'],
    url='https://github.com/katiana22/GrassmannianEGO',
    classifiers=['Programming Language :: Python :: 3.5'],
    keywords='Grassmannian manifold surrogates Bayesian optimization',
)
