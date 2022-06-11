from pyscf import gto, scf, cc, dft, ao2mo
import numpy as np

mol = gto.M(
            atom = 'H 0 0 0',  # in Angstrom
            basis = 'aug-cc-pvqz',
            spin = 1
                    )
mf = scf.UHF(mol).run()
mycc = cc.UCCSD(mf)
mycc.kernel()
print('CCSD total energy', mycc.e_tot)

grid = dft.gen_grid.Grids(mol)
grid.level = 8
grid.build()

r = np.linalg.norm(grid.coords, axis=1)
rn = np.power(r,2)

combined = grid.weights * rn

dm1 = mycc.make_rdm1(ao_repr=True)[0] + mycc.make_rdm1(ao_repr=True)[1]
ao_value =  dft.numint.eval_ao(mol, grid.coords, deriv=0)
#orb = mycc.mo_coeff
#mo_value = ao2mo.full(ao_value, orb)

rho = dft.numint.eval_rho(mol, ao_value, dm1)
#ni = dft.numint.NumInt()
#rho = dft.numint.get_rho(ni, mol, dm1, grid)

integral = np.dot(rho.T, combined.T)
print(integral)
