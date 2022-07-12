#!/usr/bin/env python
 
'''
Based on example at https://github.com/pyscf/pyscf.github.io/blob/master/examples/scf/40-apply_electric_field.py
'''
 
import numpy as np
import pyscf
from pyscf import gto, scf, tools, cc, dft
from scipy.special import sph_harm
import math as m

# Simple coordinate conversion
def cart2sph(coord):
    x,y,z = coord
    r = np.sqrt(x**2+y**2+z**2)
    theta=np.arctan2(np.sqrt(x**2+y**2),z)
    phi=np.arctan2(y,x)
    return r, theta, phi

# General matrix element evaluation
# returns <Psi_0|f(r)|Psi_0>
def gen_matrix_element(function_to_eval, molecule, mycc, args):
    matelements = np.asarray([])

    for position in grid.coords:
        r, theta, phi = cart2sph( position)
        real_space_value = function_to_eval(r, theta, phi, args)
        matelements = np.append(matelements, real_space_value)

    dm1 = mycc.make_rdm1(ao_repr=True)[0] + mycc.make_rdm1(ao_repr=True)[1]
    ao_value =  dft.numint.eval_ao(mol, grid.coords, deriv=0)
    rho = dft.numint.eval_rho(mol, ao_value, dm1)

    combined = grid.weights * matelements
    integral = np.dot(rho.T, combined.T)
    return integral 


# The main function, handles the external field and the confinement
def apply_confinement_and_field(molecule,E,l):
    field=[E,0,0] # X field
    molecule.set_common_orig([0, 0, 0])  # Not sure if we need it...?

    h=(molecule.intor('cint1e_kin_sph') + molecule.intor('cint1e_nuc_sph') + l* molecule.intor('cint1e_r2_sph')) #Adding a confinement l*r^2
    mf = scf.UHF(molecule)
    mf.get_hcore = lambda *args: h
    mf.scf()
    mf.kernel()
    conf = cc.UCCSD(mf)
    conf.kernel()
    r2_conf = get_rn_expt(molecule,conf,grid,3) # Calculating r2 in confimenent (note: no field here)
    confined_energy = conf.e_tot

    h=(h+ np.einsum('x,xij->ij', field, molecule.intor('cint1e_r_sph', comp=3))) # Adding field to the confined atom
    mf = scf.UHF(molecule)
    mf.get_hcore = lambda *args: h
    mf.scf()
    mf.kernel()
    conf_field = cc.UCCSD(mf).run()
    confined_field_energy = conf_field.e_tot
    return confined_energy, confined_field_energy, r2_conf

# The main function, handles the external field and the confinement
def apply_confinement_and_quadrupole_field(molecule,E,l):
    field=[E/2.0,0,0] # X field
    molecule.set_common_orig([0, 0, 0])  # Not sure if we need it...?

    h=(molecule.intor('cint1e_kin_sph') + molecule.intor('cint1e_nuc_sph') + l* molecule.intor('cint1e_r2_sph')) #Adding a confinement l*r^2
    mf = scf.UHF(molecule)
    mf.get_hcore = lambda *args: h
    mf.scf()
    mf.kernel()
    conf = cc.UCCSD(mf)
    conf.kernel()
    r2_conf = get_rn_expt(molecule,conf,grid,2) # Calculating r2 in confimenent (note: no field here)
    r4_conf = get_rn_expt(molecule,conf,grid,4) # Calculating r2 in confimenent (note: no field here)
    confined_energy = conf.e_tot

    h=(h+3.0*np.einsum('x,xij->ij', field, mol.intor('int1e_zz', comp=3))-np.einsum('x,xij->ij', field, mol.intor('cint1e_r2_sph', comp=3))) # Adding field to the confined atom
    mf = scf.UHF(molecule)
    mf.get_hcore = lambda *args: h
    mf.scf()
    mf.kernel()
    conf_field = cc.UCCSD(mf).run()
    confined_field_energy = conf_field.e_tot
    return confined_energy, confined_field_energy, r2_conf, r4_conf

# Setting up the molecule
mol = gto.Mole()
mol.atom = '''
     Ne 0.0000 0.0000     0.000000000000
  '''
mol.basis = 'aug-cc-pvQz'
mol.spin=0
mol.build()

# Unperturbed calculators
mf0 = scf.UHF(mol)
mf0.kernel()
cc0 = cc.UCCSD(mf0)
cc0.kernel()

# Grid for spatial integration
grid = pyscf.dft.gen_grid.Grids(mol)
grid.level = 9
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
    prefactor = 16*np.pi/(2*l+1)
    c_expt_value = gen_matrix_element(c_operator, molecule, coupledcluster, (lp, mp))
    field_squared = gen_matrix_element(solid_harmonic_squared, molecule, coupledcluster, (lp, mp))
    response_squared = gen_matrix_element(solid_harmonic_squared, molecule, coupledcluster, (l, m))
    return prefactor*field_squared*response_squared/c_expt_value

print(polarizability_lm_lpmp(mol, cc0, 1, 0, 1, 0))

#results = open("conf_alpha2_r2_r4.txt", "w")

#confinements = np.arange(0,0.2,0.001)
#for conf in confinements:
#    ene0, ene1, r2, r4 = apply_confinement_and_quadrupole_field(mol,0.0001,conf)
#    alpha = -3*(ene1-ene0)/0.0001/0.0001
#    results.write(str(conf) + "  "+ str(alpha) + "   "+ str(r2) +"  "+str(r4)+ "\n")


