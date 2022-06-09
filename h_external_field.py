#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#
 
'''
This example has two parts.  The first part applies oscillated electric field
by modifying the 1-electron Hamiltonian.  The second part generate input
script for Jmol to plot the hemos.  Running jmol yyyx.spt can output 50
image files of hemos under different electric field.
'''
 
import numpy
from pyscf import gto, scf, tools
 
mol = gto.Mole() # Benzene
mol.atom = '''
     H   0.0000 0.0000     0.000000000000
  '''
mol.basis = 'aug-cc-pVQZ'
mol.spin=1
mol.build()
mf0 = scf.UHF(mol)
mf0.kernel()
 
#
# Past 1, generate all hemos with external field
#
N = 50 # 50 samples in one period of the oscillated field
mo_id = 20  # hemo
dm_init_guess = [None]
 
def apply_field(E):
    mol.set_common_orig([0, 0, 0])  # The gauge origin for dipole integral
    h =(mol.intor('cint1e_kin_sph') + mol.intor('cint1e_nuc_sph') + numpy.einsum('x,xij->ij', E, mol.intor('cint1e_r_sph', comp=3)))
    mf = scf.UHF(mol)
    mf.get_hcore = lambda *args: h
    mf.scf(dm_init_guess[0])
    dm_init_guess[0] = mf.make_rdm1()
    mf.kernel()
    #mo = mf.mo_coeff[:,mo_id]
    #if mo[23] < -1e-5:  # To ensure that all MOs have same phase
    #    mo *= -1
    return 1
 
fields = numpy.asarray([0.0001])
mos = [apply_field((i,0,0)) for i in fields]
print(mos)
 
