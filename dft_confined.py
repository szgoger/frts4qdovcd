#!/usr/bin/env python
 
'''
Evaluating multipole polarizabilities with DFT in confinement
'''
import numpy as np
import pyscf
from pyscf import gto, scf, tools, dft
from scipy.special import sph_harm
import math as m


# Simple coordinate conversion
def cart2sph(coord):
    x,y,z = coord
    r = np.sqrt(x**2+y**2+z**2)
    theta=np.arctan2(np.sqrt(x**2+y**2),z)
    phi=np.arctan2(y,x)
    return r, theta, phi


###
# Initialize the molecule and the calculation
###
def setup_system():
    mol = gto.M( # Change atom here
        atom = 'H 0 0 0',
        basis = {'H': 'cc-pv5z'},
        spin=1,
        charge=0,
        verbose=0
    )
    mol.build()

    # Unperturbed calculators
    mf0 = dft.UKS(mol)
    mf0.xc = 'R2SCAN,R2SCAN'
    e0 = mf0.kernel()

    # Grid for spatial integration
    grid = pyscf.dft.gen_grid.Grids(mol)
    grid.level = 2
    grid.build()

    return mol, mf0, grid, e0

###
# The main function, handles the external field and the confinement
###
def apply_confinement_and_field(molecule,E,confinement,l):
    #molecule.set_common_orig([0, 0, 0])  # The gauge origin for dipole integral

    # We apply the confinement
    h=(molecule.intor('cint1e_kin_sph') + molecule.intor('cint1e_nuc_sph')+ confinement*molecule.intor('cint1e_r2_sph')) # Adding field to the atom
    mf = dft.UKS(molecule)
    mf.xc = 'R2SCAN,R2SCAN'
    mf.get_hcore = lambda *args: h
    mf.scf()
    e1_conf = mf.kernel()

    # Calculating the matrix elements
    r2_conf = gen_matrix_element(rn, molecule, mf, 2)
    r3_conf = gen_matrix_element(rn, molecule, mf, 2)
    new_pol = np.real(polarizability_lm_lpmp(molecule, mf, l, 0, l, 0))

    # Now we apply an external electric field
    if l==1: # dipole
        field=[E,0,0]
        h=(h+ np.einsum('x,xij->ij', field, molecule.intor('cint1e_r_sph', comp=3))) # Adding field to the atom
        mf = dft.UKS(molecule)
        mf.xc = 'R2SCAN,R2SCAN'
        mf.get_hcore = lambda *args: h
        mf.scf()
        e1_field = mf.kernel()

    elif l==2: # quadrupole
        field=[E/2.0,0,0]
        h=(h+3.0*np.einsum('x,xij->ij', field, molecule.intor('int1e_zz', comp=3))-np.einsum('x,xij->ij', field, molecule.intor('cint1e_r2_sph', comp=3)) ) # Adding field to the atom
        mf = dft.UKS(molecule)
        mf.xc = 'R2SCAN,R2SCAN'
        mf.get_hcore = lambda *args: h
        mf.scf()
        e1_field = mf.kernel()

    return e1_conf, e1_field, r2_conf, r3_conf, new_pol

###
# New polarizability predictor
###
def polarizability_lm_lpmp(molecule, coupledcluster, l, m, lp, mp): #returns \alpha_lml'm'
    #nr_of_electrons = gen_matrix_element(rn, molecule, coupledcluster,0)
    nr_of_electrons = 1
    prefactor = 4*nr_of_electrons*4*np.pi/(2*l+1)

    c_expt_value = gen_matrix_element(c_operator, molecule, coupledcluster, (lp, mp))
    field_squared = gen_matrix_element(solid_harmonic_squared, molecule, coupledcluster, (lp, mp))
    response_squared = gen_matrix_element(solid_harmonic_squared, molecule, coupledcluster, (l, m))
    return prefactor*field_squared*response_squared/c_expt_value


################################################################################################
# "Library" of functions for general matrix elements
################################################################################################

###
# General matrix element evaluation
# returns <Psi_0|f(r)|Psi_0>###
###
def gen_matrix_element(function_to_eval, molecule, myhf, args):
    matelements = np.asarray([])

    for position in grid.coords:
        r, theta, phi = cart2sph( position)
        real_space_value = function_to_eval(r, theta, phi, args) # General fcn evaluated on grid
        matelements = np.append(matelements, real_space_value)

    dm1 = myhf.make_rdm1(ao_repr=True)[0] + myhf.make_rdm1(ao_repr=True)[1]
    ao_value =  dft.numint.eval_ao(molecule, grid.coords, deriv=0)
    rho = dft.numint.eval_rho(molecule, ao_value, dm1) # Density on the same grid

    combined = grid.weights * matelements
    integral = np.dot(rho.T, combined.T)
    return integral 

def rn(r, theta, phi, n):
    return r**n

def c_operator(r, theta, phi, args):
    l, m = args
    spharm_m = sph_harm(m, l, phi, theta)
    spharm_mpone = sph_harm(m+1, l,  phi, theta)
    return np.power(r,2*l-2) * ((l**2) * spharm_m**2.0 + l*(l+1)*(np.abs(spharm_mpone)**2)) 

def solid_harmonic_squared(r, theta, phi, args):
    l, m = args
    one_harmonic = (r**l) * sph_harm(m, l, phi, theta)
    return one_harmonic * one_harmonic


################################################################################################
################################################################################################
molecule, calculator, grid, e0 = setup_system() # setting up the calculation

field = 0.005
confinements = np.arange(0,1,0.05)
l = 1 #1 for dipole, 2 for quadrupolole
for conf in confinements:
    curr_conf = - 0.05 * e0*conf # Up to 5% total energy as confinement
    e1_conf, e1_field, r2, r3, pol = apply_confinement_and_field(molecule,field,curr_conf,l)

    if l == 1:
        alpha_from_energies = -2*(e1_field-e1_conf)/field/field
    elif l == 2:
        alpha_from_energies = -3*(e1_field-e1_conf)/field/field

    print(conf,alpha_from_energies, r3, r2**2, pol)
