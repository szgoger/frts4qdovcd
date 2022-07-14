#!/usr/bin/env python
 
'''
Based on example at https://github.com/pyscf/pyscf.github.io/blob/master/examples/scf/40-apply_electric_field.py
'''
 
import numpy as np
import pyscf
from pyscf import gto, scf, tools, dft
from scipy.special import sph_harm
import math as m

# Setting up the molecule
mol = gto.M(
    atom = 'O 0 0 0; ghost 0 0 1; ghost 0 2 0',
    basis = {'O': 'cc-pvqz', 'ghost': gto.basis.load('aug-cc-pvtz', 'H')}
)
mol.spin=2
mol.charge=0
mol.build()
mol.verbose=0

# Simple coordinate conversion
def cart2sph(coord):
    x,y,z = coord
    r = np.sqrt(x**2+y**2+z**2)
    theta=np.arctan2(np.sqrt(x**2+y**2),z)
    phi=np.arctan2(y,x)
    return r, theta, phi

# General matrix element evaluation
# returns <Psi_0|f(r)|Psi_0>
def gen_matrix_element(function_to_eval, molecule, myhf, args):
    matelements = np.asarray([])

    for position in grid.coords:
        r, theta, phi = cart2sph( position)
        real_space_value = function_to_eval(r, theta, phi, args)
        matelements = np.append(matelements, real_space_value)

    ao_value =  dft.numint.eval_ao(molecule, grid.coords, deriv=0)

    combined = grid.weights * matelements
    integral = np.dot(ao_value.T, combined.T)
    return integral 


# The main function, handles the external field and the confinement
def apply_confinement_and_field(molecule,E,l):
    field=[E,0,0] # X field
#    molecule.set_common_orig([0, 0, 0])  # Not sure if we need it...?

    relative_confinement = np.abs(molecule.intor('cint1e_kin_sph').trace() + molecule.intor('cint1e_nuc_sph').trace())*0.01*l

    h=(molecule.intor('cint1e_kin_sph') + molecule.intor('cint1e_nuc_sph') + relative_confinement* molecule.intor('cint1e_r2_sph')) #Adding a confinement l*r^2
    mf = dft.UKS(molecule)
    mf.xc = "PBE0"
    mf.get_hcore = lambda *args: h
    mf.scf()
    confined_energy = mf.kernel()
    r2_conf =  gen_matrix_element(rn, mol, mf,2)
    r3_conf =  gen_matrix_element(rn, mol, mf,3)
    alpha_conf = polarizability_lm_lpmp(mol, mf, 1, 0, 1, 0) #dipole alpha

    h=(h+ np.einsum('x,xij->ij', field, molecule.intor('cint1e_r_sph', comp=3))) # Adding field to the confined atom
    mf = scf.UKS(molecule)
    mf.get_hcore = lambda *args: h
    mf.scf()
    confined_field_energy = mf.kernel()
    return confined_energy, confined_field_energy, alpha_conf, r2_conf, r3_conf

# The main function, handles the external field and the confinement
def apply_confinement_and_quadrupole_field(molecule,E,l):
    field=[E/2.0,0,0] # X field
    molecule.set_common_orig([0, 0, 0])  # Not sure if we need it...?

    h=(molecule.intor('cint1e_kin_sph') + molecule.intor('cint1e_nuc_sph') + l* molecule.intor('cint1e_r2_sph')) #Adding a confinement l*r^2
    mf = scf.UKS(molecule)
    mf.xc = "PBE0"
    mf.get_hcore = lambda *args: h
    mf.scf()
    mf.kernel()
    r2_conf =  gen_matrix_element(rn, molecule, mf ,2) # Calculating r2 in confimenent (note: no field here)
    r4_conf =  gen_matrix_element(rn, molecule, mf ,4) # Calculating r2 in confimenent (note: no field here)
    confined_energy = conf.e_tot

    h=(h+3.0*np.einsum('x,xij->ij', field, mol.intor('int1e_zz', comp=3))-np.einsum('x,xij->ij', field, mol.intor('cint1e_r2_sph', comp=3))) # Adding field to the confined atom
    mf = scf.UKS(molecule)
    mf.get_hcore = lambda *args: h
    mf.scf()
    confined_field_energy = mf.kernel()
    return confined_energy, confined_field_energy, r2_conf, r4_conf


# Unperturbed calculators
mf0 = dft.UKS(mol)
mf0.xc = 'PBE0'
mf0.kernel()

# Grid for spatial integration
grid = pyscf.dft.gen_grid.Grids(mol)
grid.level = 5
grid.build()

# Input functions for general matrix elements
def rn(r, theta, phi, n):
    return r**n

def c_operator(r, theta, phi, args):
    l, m = args
    spharm_m = sph_harm(m, l, phi, theta) # documentation is wrong for l and m???
    spharm_mpone = sph_harm(m+1, l,  phi, theta)
    return np.power(r,2*l-2) * ((l**2) * spharm_m**2.0 + l*(l+1)*(np.abs(spharm_mpone)**2)) 

def solid_harmonic_squared(r, theta, phi, args):
    l, m = args
    one_harmonic = (r**l) * sph_harm(m, l, phi, theta)
    return one_harmonic * one_harmonic

def polarizability_lm_lpmp(molecule, coupledcluster, l, m, lp, mp): #returns \alpha_lml'm'
    #nr_of_electrons = gen_matrix_element(rn, molecule, coupledcluster,0)
    nr_of_electrons = 1
    prefactor = 4*nr_of_electrons*4*np.pi/(2*l+1)

    c_expt_value = gen_matrix_element(c_operator, molecule, coupledcluster, (lp, mp))
    field_squared = gen_matrix_element(solid_harmonic_squared, molecule, coupledcluster, (lp, mp))
    response_squared = gen_matrix_element(solid_harmonic_squared, molecule, coupledcluster, (l, m))
    return prefactor*field_squared*response_squared/c_expt_value

#print(polarizability_lm_lpmp(mol, cc0, 1, 0, 1, 0))

#results = open("conf_alpha2_r2_r4.txt", "w")

field = 0.005
confinements = np.arange(0,1,0.05)
r2_0, r3_0, al4_0, a0 = "k", "k", "k", "k"
for conf in confinements:
    ene0, ene1, alpha_l4, r2_conf, r3_conf = apply_confinement_and_field(mol,field,conf)
    alpha = -2*(ene1-ene0)/field/field
    if r2_0 == "k":
        r2_0 = r2_conf
        r3_0 = r3_conf
        al4_0 = alpha_l4
        a0 = alpha
    #alpha = -3*(ene1-ene0)/0.0001/0.0001
    #results.write(str(conf) + "  "+ str(alpha) + "   "+ str(r2) +"  "+str(r4)+ "\n")
    print(conf, "     ", str(a0*np.real(alpha_l4)/np.real(al4_0)/alpha), "     ",  a0*r2_conf**2/r2_0**2/alpha,  "     ", a0*r3_conf /r3_0/alpha)


