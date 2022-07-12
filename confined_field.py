#!/usr/bin/env python
 
'''
SG
'''
 
import numpy as np
import pyscf
from pyscf import gto, scf, tools, cc, dft
from scipy.special import sph_harm
import math as m

def cart2sph(coord):
    x,y,z = coord
    r = np.sqrt(x**2+y**2+z**2)
    theta=np.arctan2(np.sqrt(x**2+y**2),z)
    phi=np.arctan2(y,x)
    return r, theta, phi

def gen_matrix_element(function_to_eval, molecule, mycc, m, l):
    matelements = np.asarray([])

    for position in grid.coords:
        r, theta, phi = cart2sph( position)
        real_space_value = function_to_eval(r, theta, phi, m, l)
        matelements = np.append(matelements, real_space_value)

    dm1 = mycc.make_rdm1(ao_repr=True)[0] + mycc.make_rdm1(ao_repr=True)[1]
    ao_value =  dft.numint.eval_ao(mol, grid.coords, deriv=0)
    rho = dft.numint.eval_rho(mol, ao_value, dm1)

    combined = grid.weights * matelements
    integral = np.dot(rho.T, combined.T)
    return integral 


def spharm_matrix(mycc,l,m,nspher,nr):
    matelements = np.asarray([])

    for position in grid.coords:
        r, theta, phi = cart2sph( position)
        spharm = sph_harm(m, l, phi, theta) # documentation is wrong for l and m???
        matelements = np.append(matelements, spharm**nspher * r**nr)

    dm1 = mycc.make_rdm1(ao_repr=True)[0] + mycc.make_rdm1(ao_repr=True)[1]
    ao_value =  dft.numint.eval_ao(mol, grid.coords, deriv=0)
    rho = dft.numint.eval_rho(mol, ao_value, dm1)

    combined = grid.weights * matelements
    integral = np.dot(rho.T, combined.T)
    return integral 


def denom(molecule,mycc, l,m=0):
# Note that while the function can be called with any m, it only makes sense for m=0
    matelements = np.asarray([])
    for position in grid.coords:
        r, theta, phi = cart2sph( position)
        spharm_m = sph_harm(m, l, phi, theta) # documentation is wrong for l and m???
        spharm_mpone = sph_harm(m+1, l,  phi, theta)

        matelements = np.append(matelements, np.power(r,2*l-2) * ((l**2) * spharm_m**2.0 + l*(l+1)*(np.abs(spharm_mpone)**2))  )
    combined = grid.weights * matelements

    dm1 = mycc.make_rdm1(ao_repr=True)[0] + mycc.make_rdm1(ao_repr=True)[1]
    ao_value =  dft.numint.eval_ao(mol, grid.coords, deriv=0)
    rho = dft.numint.eval_rho(mol, ao_value, dm1)

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

# Real space evaluation of r^n matrix element

# Real space evaluation of r^n matrix element
def get_rn_expt(molecule,mycc,grids,n):
    r = np.linalg.norm(grid.coords, axis=1)
    rn = np.power(r,n)
    combined = grid.weights * rn

    dm1 = mycc.make_rdm1(ao_repr=True)[0] + mycc.make_rdm1(ao_repr=True)[1]
    ao_value =  dft.numint.eval_ao(mol, grid.coords, deriv=0)
    rho = dft.numint.eval_rho(mol, ao_value, dm1)

    integral = np.dot(rho.T, combined.T)
    return integral

# Setting up the molecule
mol = gto.Mole()
mol.atom = '''
     He 0.0000 0.0000     0.000000000000
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
grid.level = 3
grid.build()

# Polarizability predictors from the paper
#denominator = denom(mol,cc0,1)
#spharm = spharm_matrix(cc0,1,0,2,2)
#alpha=(spharm**2 )/denominator
#print("predicted dipole from L4 formula: ",alpha)
#        spharm = sph_harm(m, l, phi, theta) # documentation is wrong for l and m???

# Testing general matrix elements
#def r2(r, theta, phi, m, n): #need to define m and n because of bad code
#    return r**2

def c_operator(r, theta, phi, m, l):
    spharm_m = sph_harm(m, l, phi, theta) # documentation is wrong for l and m???
    spharm_mpone = sph_harm(m+1, l,  phi, theta)
    return np.power(r,2*l-2) * ((l**2) * spharm_m**2.0 + l*(l+1)*(np.abs(spharm_mpone)**2)) 

def solid_harmonic_squared(r, theta, phi, m, l):
    one_harmonic = (r**l) * sph_harm(m, l, phi, theta)
    return one_harmonic * one_harmonic

l_denom = gen_matrix_element(c_operator,mol, cc0, 0, 1)
l_numer = gen_matrix_element(solid_harmonic_squared,mol, cc0, 0, 1)

print("lambda = ",l_numer/l_denom)

print("predicted pol = ",4*l_numer*l_numer/l_denom)

#print("expt value ",denom(mol,cc0,1))
#print("Y l=1 m=0 expt value ",spharm_matrix(cc0,1,0))

#results = open("conf_alpha2_r2_r4.txt", "w")

#confinements = np.arange(0,0.2,0.001)
#for conf in confinements:
#    ene0, ene1, r2, r4 = apply_confinement_and_quadrupole_field(mol,0.0001,conf)
#    alpha = -3*(ene1-ene0)/0.0001/0.0001
#    results.write(str(conf) + "  "+ str(alpha) + "   "+ str(r2) +"  "+str(r4)+ "\n")


