import numpy as np

## Creates an empty list of Cartesians
def create_geometry(NAtoms):
    geometry = []
    for ii in range(NAtoms):
        geometry.append( [0]* 3)
    return geometry

## Modifies input data by exploiting Python referencing
def read_geometry(coordinates):
    coordinates[0] = [0, 0, 0]
    coordinates[1] = [0, 1, 0]

def read_atypes(atypes):
    atypes[0] = "H"
    atypes[1] = "H"

def setup_free_atom_data(dipoles, quadrupoles, c6_coeffs, atom_types):
    for ii in range(len(dipoles)):
        dipoles[ii] = free_dipoles[ atom_types[ii] ]
        quadrupoles[ii] = free_quadrupoles[ atom_types[ii] ]
        c6_coeffs[ii] = free_c6s[ atom_types[ii] ]
    return

### Constants and physical parameters
free_dipoles = {"H": 4.5}
free_quadrupoles = {"H": 15}
free_c6s = {"H": 6.5}

### Initializing empty variables
NAtoms = 2
coords = create_geometry(NAtoms)
atom_types = [0] * NAtoms
dipole_alphas, quadrupole_alphas, c6s = [0] * NAtoms, [0] * NAtoms, [0] * NAtoms
r2_free, r2_aim, r4_free, r4_aim = [0] * NAtoms, [0] * NAtoms, [0] * NAtoms, [0] * NAtoms

### Reading data
read_geometry(coords)
read_atypes(atom_types)
setup_free_atom_data(dipole_alphas, quadrupole_alphas, c6s, atom_types)
