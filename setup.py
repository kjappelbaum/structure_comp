from setuptools import setup

setup(
    name='structure_comp',
    version='0.1',
    packages=['structure_comp'],
    url='',
    license='MIT',
    install_requires=[
        'numpy>=1.14.3', 'pymatgen',  'ase', 'tqdm', 'pandas', 'scipy', 'scikit-learn'
    ],
    extras_require={
        'testing': ['pytest'],
        'docs': ['sphinx-rtd-theme', 'sphinxcontrib-bibtex'],
        'pre-commit': [
            'pre-commit==1.11.0', 'yapf==0.24.0', 'prospector==1.1.5',
            'pylint==1.9.3'
        ]
    },
    author='Kevin M. Jablonka',
    author_email='kevin.jablonka@epfl.ch',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    description=
    'small, efficient package to remove duplicates from structural databases')
