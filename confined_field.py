#!/usr/bin/env python
 
'''
Based on example at https://github.com/pyscf/pyscf.github.io/blob/master/examples/scf/40-apply_electric_field.py
'''
 
import numpy as np
import pyscf
from pyscf import gto, scf, tools, cc, dft
from scipy.special import sph_harm

def cart2sph(cartesians):
    x, y, z = cartesians
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r #TODO verify consistency of angles

def spharm_matrix(coord,l,m):
    for position in grid.coords:
        print(cart2sph( position))
    #phi, theta, r = cart2sph(coord)
    #r = np.linalg.norm(grid.coords, axis=1)
    #rn = np.power(r,n)
    #combined = grid.weights * rn

    #dm1 = mycc.make_rdm1(ao_repr=True)[0] + mycc.make_rdm1(ao_repr=True)[1]
    #ao_value =  dft.numint.eval_ao(mol, grid.coords, deriv=0)
    #rho = dft.numint.eval_rho(mol, ao_value, dm1)

    #integral = np.dot(rho.T, combined.T)
    return np.power(r,l) * sph_harm(l, m, phi, theta) # documentation is wrong for l and m???


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
     Na 0.0000 0.0000     0.000000000000
  '''
mol.basis = 'aug-cc-pvQz'
mol.spin=1
mol.build()

# Unperturbed calculators
mf0 = scf.UHF(mol)
mf0.kernel()
cc0 = cc.UCCSD(mf0)
cc0.kernel()

# Grid for spatial integration
grid = pyscf.dft.gen_grid.Grids(mol)
grid.level = 8
grid.build()

results = open("conf_alpha2_r2_r4.txt", "w")

confinements = np.arange(0,0.2,0.001)
for conf in confinements:
    ene0, ene1, r2, r4 = apply_confinement_and_quadrupole_field(mol,0.0001,conf)
    alpha = -3*(ene1-ene0)/0.0001/0.0001
    results.write(str(conf) + "  "+ str(alpha) + "   "+ str(r2) +"  "+str(r4)+ "\n")

