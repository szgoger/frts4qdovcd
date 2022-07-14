#!/usr/bin/env python
 
'''
Based on example at https://github.com/pyscf/pyscf.github.io/blob/master/examples/scf/40-apply_electric_field.py
'''
 
import numpy
import pyscf
from pyscf import gto, scf, tools, cc
 
mol = gto.Mole() # Benzene
mol.atom = '''
     He 0.0000 0.0000     0.000000000000
  '''
mol.basis = 'aug-cc-pvQz'
mol.spin=0
mol.build()
mf0 = scf.UHF(mol)
mf0.kernel()
free = cc.UCCSD(mf0).run()

def apply_matrix(E):
    mol.set_common_orig([0, 0, 0])  # The gauge origin for dipole integral

    h=( mol.intor('cint1e_kin_sph') + mol.intor('cint1e_nuc_sph') + numpy.einsum('x,xij->ij', E, mol.intor('cint1e_r4_sph', comp=3)))
    print("calculation with confinement and field")
    mf = scf.UHF(mol)
    mf.get_hcore = lambda *args: h
    mf.scf()
    mf.kernel()
    conf_field = cc.UCCSD(mf).run()
    confined_field_energy = conf_field.e_tot
    return  confined_field_energy
 
field = 0.0001

e0 = free.e_tot

e1 = apply_matrix((field,0,0))
alpha = -2 * (e1-e0)/field/field
 
eigen
