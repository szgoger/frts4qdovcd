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

def read_volumes(r2free, r2, r4free, r4): #TODO: these are just some random values below...
    for ii in range(len(r2free)):
        r2free[ii] = 3.0
        r4free[ii] = 45.0/2.0
        r2[ii] = 2.7
        r4[ii] = 14.0
        r2[ii], r4[ii] = r2free[ii], r4free[ii] #TODO uncomment this line!!!!!!

## Reascaling of free atomic parameters
def rescale_dipole(dipoles, r2free, r2aim):
    scaled_dipoles = np.multiply( dipoles, np.divide( np.power(r2_aim,2.0) , np.power(r2_free,2.0) ))
    return scaled_dipoles

def rescale_c6s(c6, r2free, r2aim):
    scaled_c6s = np.multiply( c6, np.divide( np.power(r2_aim,4.0) , np.power(r2_free,4.0) ))
    return scaled_c6s

def rescale_quadrupole(quadrupoles, r2free, r2aim, r4free, r4aim):
    free_factor = np.power(r4free, 2.0)/r2free
    aim_factor = np.power(r4aim, 2.0)/r2aim
    scaled_quadrupoles = np.multiply(quadrupoles, np.divide(aim_factor, free_factor))
    return scaled_quadrupoles

## Fully parametrized non-empirical QDO
def qdo(alpha1, c6, alpha2):
    model_params = []
    for ii in range(len(alpha1)):
        frequency = (4 * c6[ii])/(3 * alpha1[ii]**2)
        mass = (alpha1[ii]/alpha2[ii]) * (  3 / (4 * frequency) )
        charge = np.sqrt( mass * alpha1[ii] * frequency**2 )
        model_params.append( [frequency, mass, charge] )
    return model_params


###
### The program starts here
###


### Constants and physical parameters
free_dipoles = {"H": 4.5}
free_quadrupoles = {"H": 15.0}
free_c6s = {"H": 6.5}

### Initializing empty variables
NAtoms = 2
coords = create_geometry(NAtoms)
atom_types = [0] * NAtoms
dipole_alphas, quadrupole_alphas, c6s = [0] * NAtoms, [0] * NAtoms, [0] * NAtoms
r2_free, r2_aim, r4_free, r4_aim = [0] * NAtoms, [0] * NAtoms, [0] * NAtoms, [0] * NAtoms

### Reading data from actual DFT calculation
read_geometry(coords)
read_atypes(atom_types)
read_volumes(r2_free, r2_aim, r4_free, r4_aim)

### Free atom data
setup_free_atom_data(dipole_alphas, quadrupole_alphas, c6s, atom_types)

### Rescaling to atom in a molecule
aim_dipoles = rescale_dipole(dipole_alphas, r2_free, r2_aim)
aim_c6s = rescale_c6s(c6s, r2_free, r2_aim)
aim_quadrupoles = rescale_quadrupole(quadrupole_alphas, r2_free, r2_aim, r4_free, r4_aim)

### Parametrizing the QDO model
qdo_parameters = qdo(aim_dipoles, aim_c6s, aim_quadrupoles) #returns a vector of NAtoms vector [frequency, mass, charge]

print(qdo_parameters)
