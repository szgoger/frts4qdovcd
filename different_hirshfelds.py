#This code runs through AIMS output files, mines Hirshfeld volumes and polarizabilities

#With these values, TS and TS+SCS polarizabilities are calculated

#It is straightforward to compare different volume scalings

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent/'../..'))
import numpy as np


#Atomic polarizabilities from Schwerdtfeger et al, 2018, DOI: 10.1080/00268976.2018.1535143
def atomic_polarizabilities(list_of_elements):
	polarizabilities,uncertanities = [],[]
	for element in list_of_elements:
		if element == 'H':
			polarizabilities.append(4.50456)
			uncertanities.append(0.0)
		elif element == 'C':
			polarizabilities.append(11.3)
			uncertanities.append(0.2)
		elif element == 'O':
			polarizabilities.append(5.3)
			uncertanities.append(0.2)
		elif element == 'F':
			polarizabilities.append(3.74)
			uncertanities.append(0.08)
		elif element == 'Cl':
			polarizabilities.append(14.6)
			uncertanities.append(0.1)
		elif element == 'Br':
			polarizabilities.append(21)
			uncertanities.append(1.0)
		elif element == 'S':
			polarizabilities.append(19.4)
			uncertanities.append(0.1)
		elif element == 'N':
			polarizabilities.append(7.4)
			uncertanities.append(0.2)
	return polarizabilities, uncertanities

#The function to return geometries and atom types from an AIMS output file
#Input: absolute path to AIMS output file
def fetch_geometry(filename):
	f = open(filename,'r')
	coordinates, atom_types = [], [] #Results will be in a list
	for line in f:
		if '  atom   ' in str(line):
			getgeom = str(line).split()
			for i in range(1,4):  #This way, we get a list of 3N
				coordinates.append(getgeom[i])
			atom_types.append(getgeom[-1]) #This just fetches the last element
	f.close()
	return coordinates, atom_types

#Function that returns free and Hirshfeld volumes
#Input: absolute path to AIMS output file
def fetch_hirshfeld(filename): 
	f = open(filename, 'r')
	elements = []  #Needed for free atom polarizabilities
	freevolumes = []  #Will contain volumes of free atoms
	hirshfeld1, hirshfeld2, hirshfeld3, hirshfeld4 = [], [], [], []  #Will contain Hirshfeld volumes
	for line in f:
		if '| Number of atoms            ' in str(line):
			getatoms = str(line).split()
			nr_of_atoms = int(getatoms[-1])
		elif '| Atom   ' in str(line):
			getelements = str(line).split()
			elements.append(getelements[-1])
		elif '|   Free atom volume1       :' in str(line):
			getfree = str(line).split()
			freevolumes.append(float(getfree[-1]))
		elif 'Hirshfeld volume1 ' in str(line):
			gethirshfeld= str(line).split()
			hirshfeld1.append(float(gethirshfeld[-1]))
		elif 'Hirshfeld volume2 ' in str(line):
			gethirshfeld= str(line).split()
			hirshfeld2.append(float(gethirshfeld[-1]))
		elif 'Hirshfeld volume3 ' in str(line):
			gethirshfeld= str(line).split()
			hirshfeld3.append(float(gethirshfeld[-1]))
		elif 'Hirshfeld volume4 ' in str(line):
			gethirshfeld= str(line).split()
			hirshfeld4.append(float(gethirshfeld[-1]))
	f.close()
	return elements, freevolumes, hirshfeld1, hirshfeld2, hirshfeld3, hirshfeld4

#From the free atomic alphas, hirshfeld and free volumes, returns the alpha with the given scaling factor
def ts_alpha(freeatoms, hirshfeld, freevolumes, scaling):	
	alpha = 0.0
	list_alpha = []
	for ii in range(len(hirshfeld)): #Here is the formula to approximate alpha
		alpha = alpha + freeatoms[ii] * (hirshfeld[ii]/freevolumes[ii])**(scaling)
		list_alpha.append(freeatoms[ii] * (hirshfeld[ii]/freevolumes[ii])**(scaling))
	return alpha, list_alpha

def ts_scs(coords, atypes, list_alphas):
    mbd.my_task = myid
    mbd.n_tasks = ntasks
    mode = 'M' if ntasks > 1 else ''

    bohr = mbd.bohr
    mbd.param_dipole_matrix_accuracy = 1e-10
    mbd.init_grid(10)	
	
    #These following ugly 3 lines are needed for getting the correct data type and shape of the coordinates
    float_coords_bohr = [float(i)/0.532 for i in coords]
    xyz = np.reshape(float_coords_bohr, (len(atypes),3))
    tupxyz = [tuple(xyz[i]) for i in range(len(atypes))] #We need a list of tuples....

    #omega = mbd.omega_eff(C6, list_alphas)
    alpha_scs=mbd.do_scs(mode, 'dip,gg', tupxyz, list_alphas)
    sum_alpha_scs = sum(mbd.contract_polarizability(alpha_scs))
    return sum_alpha_scs

def fetch_reference(filename):
    fpol = open(filename,'r')
    for line in fpol:
        if 'Properties' in str(line):
            axx = float(line.split()[1][10:])
            ayy = float(line.split()[2])
            azz = float(line.split()[3])
    polarizability = (axx+ayy+azz)/(3.0)
    fpol.close()
    return polarizability

directory_hirshfeld = 'different_hirshfelds' 

#directory_hirshfeld = '/home/petsza/research/MBD/azag0-mbd-c417667/tests/argon-dimer/home/users/sgoger/v43_hirshfeld_aims/output_files'
#directory_alpha = '/home/petsza/research/MBD/azag0-mbd-c417667/tests/argon-dimer/home/users/sgoger/v43_hirshfeld_aims/polarizability_reference/output_files'

pol_resfile = open('pol_results','w')  #Output containing the polarizabilities
pol_resfile.write("Filename		TS_1		TS_43		SCS_1		SCS_43		Reference \n")

hydrogen_hirshfelds, carbon_hirshfelds , oxygen_hirshfelds, nitrogen_hirshfelds= [], [], [], []

for filename in os.listdir(directory_hirshfeld): #Files in this folder are called QM7b.960.xyz, while files in the other directories are aims.out.QM7b.983
    elements, freevolumes, hirshfeld1, hirshfeld2, hirshfeld3, hirshfeld4 = fetch_hirshfeld(os.path.join(directory_hirshfeld, filename)) #Getting Hirshfeld volumes
    for elem in range(len(elements)):
        if elements[elem] == "C":
            carbon_hirshfelds.append(hirshfeld3[elem])
        elif elements[elem] == "N":
            nitrogen_hirshfelds.append(hirshfeld3[elem])
        elif elements[elem] == "O":
            oxygen_hirshfelds.append(hirshfeld3[elem])
        elif elements[elem] == "H":
            hydrogen_hirshfelds.append(hirshfeld3[elem])

for hirsh in nitrogen_hirshfelds:
    print(str(hirsh))

#    freeatoms, errors = atomic_polarizabilities(elements) #Get free atomic polarizabilities

	#Calculating the TS polarizabilities with different scalings
 #   polarizability_with_1, list_with_1 = ts_alpha(freeatoms, hirshfeld, freevolumes, 1.0)
  #  polarizability_with_43, list_with_43 = ts_alpha(freeatoms, hirshfeld, freevolumes, 4.0/3.0)

    #Now comes the TS+SCS part
   # coords, atypes = fetch_geometry(os.path.join(directory_hirshfeld, hirshfeldname))

    #We do the SCS screening of the TS polarizabilities
   # ts_scs_1 = ts_scs(coords, atypes, list_with_1)
    #ts_scs_43 = ts_scs(coords, atypes, list_with_43)
    
    #Getting the reference from the file
  #  pol_ref = fetch_reference(os.path.join(directory_alpha, filename))

    #We write the output file from here
 #   pol_resfile.write(str(filename)+'		'+str(polarizability_with_1)+'		'+str(polarizability_with_43)+'          '+str(ts_scs_1)+'          '+str(ts_scs_43)+'		'+str(polarizability)+'\n')

#pol_resfile.close()
