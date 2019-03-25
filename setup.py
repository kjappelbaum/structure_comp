from setuptools import setup

setup(
    name='structure_comp',
    version='0.1',
    packages=['structure_comp'],
    url='',
    license='MIT License',
    install_requires=[
        'numpy',
        'pymatgen',
        'ase',
        'tqdm',
        'pandas',
        'scipy'
    ],
    extras_require={
        'testing': ['pytest'],
        'pre-commit': [
            'pre-commit==1.11.0', 'yapf==0.24.0', 'prospector==1.1.5',
            'pylint==1.9.3'
        ]
    },
    author='Kevin M. Jablonka',
    author_email='kevin.jablonka@epfl.ch',
    description=
    'small, efficient package to remove duplicates from structural databases')
