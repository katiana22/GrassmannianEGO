import os
import sys
from setuptools import setup, find_packages
PACKAGE_NAME = 'GrassmannianEGO'
MINIMUM_PYTHON_VERSION = 3, 5


def check_python_version():
    """Exit when the Python version is too low."""
    if sys.version_info < MINIMUM_PYTHON_VERSION:
        sys.exit("Python {}.{}+ is required.".format(*MINIMUM_PYTHON_VERSION))


def read_package_variable(key):
    """Read the value of a variable from the package without importing."""
    module_path = os.path.join(PACKAGE_NAME, '__init__.py')
    with open(module_path) as module:
        for line in module:
            parts = line.strip().split(' ')
            if parts and parts[0] == key:
                return parts[-1].strip("'")
    assert 0, "'{0}' not found in '{1}'".format(key, module_path)


check_python_version()
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
