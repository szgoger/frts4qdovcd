#!/usr/bin/env python
 
'''
Based on example at https://github.com/pyscf/pyscf.github.io/blob/master/examples/scf/40-apply_electric_field.py
'''
 
import numpy as np
import pyscf
from pyscf import gto, scf, tools, cc, dft

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
    r2_conf = get_rn_expt(molecule,conf,grid,2) # Calculating r2 in confimenent (note: no field here)
    confined_energy = conf.e_tot

    h=(h+ np.einsum('x,xij->ij', field, molecule.intor('cint1e_r_sph', comp=3))) # Adding field to the confined atom
    mf = scf.UHF(molecule)
    mf.get_hcore = lambda *args: h
    mf.scf()
    mf.kernel()
    conf_field = cc.UCCSD(mf).run()
    confined_field_energy = conf_field.e_tot
    return confined_energy, confined_field_energy, r2_conf

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
grid.level = 8
grid.build()

confinements = np.arange(0,5,0.1)
for conf in confinements:
    ene0, ene1, r2 = apply_confinement_and_field(mol,0.001,conf)
    alpha = -2*(ene1-ene0)/0.001/0.001
    print(conf,alpha, r2)

