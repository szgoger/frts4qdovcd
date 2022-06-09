#!/usr/bin/env python
 
'''
Based on example at https://github.com/pyscf/pyscf.github.io/blob/master/examples/scf/40-apply_electric_field.py
'''
 
import numpy
from pyscf import gto, scf, tools, cc
 
mol = gto.Mole() # Benzene
mol.atom = '''
     He 0.0000 0.0000     0.000000000000
  '''
mol.basis = 'aug-cc-pVQZ'
mol.spin=0
mol.build()
mf0 = scf.UHF(mol)
mf0.kernel()
cc.UCCSD(mf0).run()
 
 
def apply_field(E):
    mol.set_common_orig([0, 0, 0])  # The gauge origin for dipole integral
    h =(mol.intor('cint1e_kin_sph') + mol.intor('cint1e_nuc_sph') + numpy.einsum('x,xij->ij', E, mol.intor('cint1e_r_sph', comp=3)))
    mf = scf.UHF(mol)
    mf.get_hcore = lambda *args: h
    mf.scf()
    mf.kernel()
    cc.UCCSD(mf).run()
 
fields = numpy.asarray([0.0001])
for i in fields:
    apply_field((i,0,0))
 
