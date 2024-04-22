"""A class for Jedi analysis"""

import os
import collections
import warnings
import numpy as np
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Union
import ase.neighborlist
import ase.geometry
from ase.atoms import Atoms
from ase.vibrations import VibrationsData
from ase.atoms import Atom
from ase.utils import jsonable
import ase.io
from ase.units import Hartree, Bohr, mol, kcal
from .colors import colors


def jedi_analysis(atoms,rim_list,B,H_cart,delta_q,E_geometries,printout=None,ase_units=False):
    '''
    Analysis of strain energy stored in redundant internal coordinates.

    atoms: class 
        An ASE Atoms object to determine the atomic species of the indices.
    rim_list: list
        A list of 4 numpy 2D arrays the first array containing bonds, second custom bonds, third bond angles, fourth dihedrals.
    B: np array
        B matrix.
    H_cart: np array
        Hessian in cartesian coordinates.
    delta_q: np array
        Array of deformations along the RICs.
    E_geometries: float
        Energy difference between the geometries.
    printout: bool
        Flag to print the output.
    ase_units: bool
        Flag to get eV for energies å fo lengths and degree for angles otherwise it is kcal/mol, Bohr and radians.
    Returns:
        Analysis of RIMs.
    '''
    #jedi analysis function
    ###########################
    ##  Matrix Calculations  ##
    ###########################
    B_transp = np.transpose(B)
    # Calculate the number of RIMs (= number of rows in the B-Matrix), equivalent to number of redundant internal coordinates
    NRIMs = int(len(rim_list))

    # Calculate the pseudoinverse of the B-Matrix and its transposed (take care of diatomic molecules specifically)
    if B.ndim == 1:
        B_plus = B_transp/2
        B_transp_plus = B/2
    else:
        B_plus = np.linalg.pinv(B, 0.0001)
        B_transp_plus = np.linalg.pinv( np.transpose(B),0.0001 )


    # Calculate the P-Matrix (eq. 4 in Helgaker's paper)
    P = np.dot(B, B_plus)


    #############################################
    #	    	   JEDI analysis	        	#
    #############################################

    # Calculate the Hessian in RIMs (take care to get the correct multiplication for a diatomic molecule
    if B.ndim == 1:
        H_q = B_transp_plus.dot( H_cart ).dot( B_plus )
    else:
        H_q = P.dot( B_transp_plus ).dot( H_cart ).dot( B_plus ).dot( P )

    # Calculate the total energies in RIMs and its deviation from E_geometries
    E_RIMs_total = 0.5 * np.transpose( delta_q ).dot( H_q ).dot( delta_q )



    # Get the energy stored in every RIM (take care to get the right multiplication for a diatomic molecule)

    if B.ndim == 1:
        E_RIMs = np.array([0.5 * delta_q[0] * H_q * delta_q[0]])

    else:
        E_RIMs = np.sum(0.5*(delta_q*H_q).T*delta_q,axis=1)
    # Get the percentage of the energy stored in every RIM
    proc_E_RIMs = []

    proc_E_RIMs = 100 * E_RIMs / E_RIMs_total

    if ase_units==True:
        b=np.shape(rim_list[0])[0]+np.shape(rim_list[1])[0] #border between lengths and angles
        delta_q[0:b] *= Bohr
        delta_q[b::] = np.degrees(delta_q[b::])
        E_RIMs=np.array(E_RIMs)*Hartree
        E_RIMs_total *= Hartree
    elif ase_units == False:
        E_RIMs=np.array(E_RIMs)/kcal*mol*Hartree
        E_RIMs_total *= mol/kcal*Hartree

    proc_geom_RIMs = 100 * ( E_RIMs_total - E_geometries ) / E_geometries

    if printout:
        jedi_printout(atoms,rim_list,delta_q,E_geometries, E_RIMs_total, proc_geom_RIMs,proc_E_RIMs, E_RIMs,ase_units=ase_units)

    return proc_E_RIMs,E_RIMs, E_RIMs_total, proc_geom_RIMs,delta_q

def jedi_printout(atoms,rim_list,delta_q,E_geometries, E_RIMs_total, proc_geom_RIMs,proc_E_RIMs, E_RIMs,ase_units=False):
    '''
    Printout of analysis of stored strain energy in redundant internal coordinates.

    atoms: class
        An ASE Atoms object to determine the atomic species of the indices.
    rim_list: list
        A list of 4 numpy 2D arrays the first array containing bonds, second custom bonds, third bond angles, fourth dihedrals.
    delta_q: np array
        Array of deformations along the RICs.
    E_geometries: float
        Energy difference between the geometries.
    E_RIMs_total: float
        Calculated total strain energy by jedi.
    proc_geom_RIMs: float
        Percentage deviation between calculated total energies.
    proc_E_RIMs: array
        Array of energy stored in each RIC.
    ase_units: bool
        Flag to get eV for energies å fo lengths and degree for angles otherwise it is kcal/mol, Bohr and radians. 
    '''
    #############################################
    #	    	   Output section	        	#
    #############################################

    output = []
    # Header
    output.append("\n \n")
    output.append("************************************************")
    output.append("\n *                 JEDI ANALYSIS                *")
    output.append("\n *       Judgement of Energy DIstribution       *")
    output.append("\n ************************************************\n")

    # Comparison of total energies
    if ase_units == False:
        output.append("\n                   Strain Energy (kcal/mol)  Deviation (%)")
    elif ase_units == True:
        output.append("\n                   Strain Energy (eV)        Deviation (%)")
    output.append("\n      Ab initio     " + "%.8f" % E_geometries + "                  -" )
    output.append('\n%5s%16.8f%21.2f' % (" JEDI           ", E_RIMs_total, proc_geom_RIMs))


    # JEDI analysis

    if ase_units == False:
        output.append("\n RIC No.       RIC type                       indices        delta_q (au) Percentage    Energy (kcal/mol)")
    elif ase_units == True:
        output.append("\n RIC No.       RIC type                       indices        delta_q (Å,°) Percentage    Energy (eV)")
    i = 0

    for k in rim_list[0]:
        rim = "bond"
        ind = "%s%d %s%d"%(atoms.symbols[k[0]], k[0], atoms.symbols[k[1]], k[1])
        output.append('\n%6i%7s%-11s%30s%17.7f%9.1f%17.7f' % (i+1, " ", rim, ind, delta_q[i], proc_E_RIMs[i], E_RIMs[i]))
        i += 1
    for k in rim_list[1]:
        rim = "custom"
        ind = "%s%d %s%d"%(atoms.symbols[k[0]], k[0], atoms.symbols[k[1]], k[1])
        output.append('\n%6i%7s%-11s%30s%17.7f%9.1f%17.7f' % (i+1, " ", rim, ind, delta_q[i], proc_E_RIMs[i], E_RIMs[i]))
        i += 1
    for k in rim_list[2]:
        rim = "bond angle"
        ind = "%s%d %s%d %s%d"%(atoms.symbols[k[0]], k[0], atoms.symbols[k[1]], k[1], atoms.symbols[k[2]], k[2])
        output.append('\n%6i%7s%-11s%30s%17.7f%9.1f%17.7f' % (i+1, " ", rim, ind, delta_q[i], proc_E_RIMs[i], E_RIMs[i]))
        i += 1
    for k in rim_list[3]:
        rim = "dihedral"
        ind = "%s%d %s%d %s%d %s%d"%(atoms.symbols[k[0]], k[0], atoms.symbols[k[1]], k[1], atoms.symbols[k[2]], k[2], atoms.symbols[k[3]], k[3])
        output.append('\n%6i%7s%-11s%30s%17.7f%9.1f%17.7f' % (i+1, " ", rim, ind, delta_q[i], proc_E_RIMs[i], E_RIMs[i]))
        i += 1
    print(*output)
    from . import quotes
    print(quotes.quotes())

def jedi_printout_bonds(atoms,rim_list,E_geometries, E_RIMs_total, proc_geom_RIMs,proc_E_RIMs, E_RIMs,ase_units=False,file='total'): #total strain in bonds after adding contributions of stretching angles and dihedral angles
    '''
    Printout of analysis of stored strain energy in the bonds.

    atoms: class
        An ASE Atoms object to determine the atomic species of the indices.
    rim_list: list
        A list of 4 numpy 2D arrays the first array containing bonds, second custom bonds, third bond angles, fourth dihedrals.
    delta_q: np array
        Array of deformations along the RICs.
    E_geometries: float
        Energy difference between the geometries.
    E_RIMs_total: float
        Calculated total strain energy by jedi.
    proc_geom_RIMs: float
        Percentage deviation between calculated total energies.
    proc_E_RIMs: np array
        Array of energy stored in each RIC.
    ase_units: bool
        Flag to get eV for energies å fo lengths and degree for angles otherwise it is kcal/mol, Bohr and radians. 
    file: string
        File to store the output.

    '''
    #############################################
    #	    	   Output section	        	#
    #############################################


    output = []
    # Header

    output.append("\n ************************************************")
    output.append("\n *                 JEDI ANALYSIS                *")
    output.append("\n *       Judgement of Energy DIstribution       *")
    output.append("\n ************************************************\n")

    # Comparison of total energies
    if ase_units==False:
        output.append("\n                   Strain Energy (kcal/mol)  Deviation (%)")
    elif ase_units==True:
        output.append("\n                   Strain Energy (eV)        Deviation (%)")
    output.append("\n      Ab initio     " + "%.8f" % E_geometries + "                  -")
    output.append('\n%5s%16.8f%21.2f' % (" JEDI           ", E_RIMs_total, proc_geom_RIMs))


    # strain in the bonds

    if ase_units == False:
        output.append("\n RIC No.       RIC type                       indices       Percentage    Energy (kcal/mol)")
    elif ase_units == True:
        output.append("\n RIC No.       RIC type                       indices       Percentage    Energy (eV)")
    i = 0

    for k in rim_list[0]:
        rim = "bond"
        ind = "%s%d %s%d"%(atoms.symbols[k[0]], k[0], atoms.symbols[k[1]], k[1])
        output.append('\n%6i%7s%-11s%30s%9.1f%17.7f' % (i+1, " ", rim, ind, proc_E_RIMs[i], E_RIMs[i]))
        i += 1
    for k in rim_list[1]:
        rim = "custom"
        ind = "%s%d %s%d"%(atoms.symbols[k[0]], k[0], atoms.symbols[k[1]], k[1])
        output.append('\n%6i%7s%-11s%30s%9.1f%17.7f' % (i+1, " ", rim, ind, proc_E_RIMs[i], E_RIMs[i]))
        i += 1
    f = open(file, 'w')
    f.writelines(output)
    f.close()

def get_hbonds(mol,covf=1.3,vdwf=0.9):
    '''
    Get all hbonds in a structure.
    Hbonds are defined as the HY bond inside X-H···Y where X and Y can be O, N, F and the angle XHY is larger than 90° and the distance between HY is shorter than 0.9 times the sum of the vdw radii of H and Y.

    mol: class
        Structure of which the hbonds should be determined.
    Returns:
        2D array of indices.
    '''
    cutoff=ase.neighborlist.natural_cutoffs(mol,mult=covf)   ## cutoff for covalent bonds see Bakken et al.
    bl=np.vstack(ase.neighborlist.neighbor_list('ij',a=mol,cutoff=cutoff)).T   #determine covalent bonds

    bl=bl[bl[:,0]<bl[:,1]]      #remove double mentioned
    bl = np.unique(bl,axis=0)
    from ase.data.vdw import vdw_radii
    hpartner = ['N','O','F']
    hpartner_ls = []
    hcutoff = {('H','N'):vdwf*(vdw_radii[1]+vdw_radii[7]),
    ('H','O'):vdwf*(vdw_radii[1]+vdw_radii[8]),
    ('H','F'):vdwf*(vdw_radii[1]+vdw_radii[9])}  #save the maximum distances for given pairs to be taken account as interactions
    hbond_ls = []                                    #create a list to store all the bonds
    for i in range(len(mol)):
        if mol.symbols[i] in hpartner:              #check atoms indices of N F O elements
            hpartner_ls.append(i)
    for i in bl:
        if mol.symbols[i[0]] == 'H' and mol.symbols[i[1]] in hpartner:
            for j in hpartner_ls:
                if j != i[1]:
                    if mol.get_distance(i[0],j,mic=True)<  hcutoff[(mol.symbols[i[0]], mol.symbols[j])] \
                        and mol.get_angle(i[1],i[0],j,mic=True)>90:
                        hbond_ls.append([i[0], j])
        elif mol.symbols[i[0]] in hpartner and mol.symbols[i[1]]=='H':
            for j in hpartner_ls:
                if j != i[0]:
                    if mol.get_distance(i[1],j,mic=True) < hcutoff[(mol.symbols[i[1]], mol.symbols[j])] and mol.get_angle(i[0],i[1],j,mic=True) >90:
                        hbond_ls.append([i[1], j])
    if len(hbond_ls) > 0:
        hbond_ls = np.array(hbond_ls)
        hbond_ls = np.sort(hbond_ls, axis=1)
        hbond_ls = np.atleast_2d(hbond_ls)
    return hbond_ls

@jsonable('jedi')
class Jedi:
    def __init__(self, atoms0, atomsF, modes): #indices=None
        '''

        atoms0: class
            Atoms object of relaxed structure with calculated energy.
        atomsF: class
            Atoms object of strained structure with calculated energy.
        modes: class
            VibrationsData object with hessian of relaxed structure.
        '''
        self.atoms0 = atoms0        #ref state
        self.atomsF = atomsF        #strained state
        self.modes = modes          #VibrationsData object
        self.B = None               #Wilson#s B
        self.delta_q = None         #strain in internal coordinates
        self.rim_list = None        #list of Redundant internal modes
        self.H = None               #cartesian Hessian of ref state
        self.energies = None        #energies of the geometries
        self.proc_E_RIMs = None     #list of procentual energy stored in single RIMs
        self.part_rim_list = None     #rim list for election of atoms
        self.indices = None           #indices to chose special atoms
        self.E_RIMs = None            #list of energies stored in the rims
        self.custom_bonds = None        #list of custom added bonds
        self.ase_units = False
        self.vdwf=0.9
        self.covf=1.3

 #       if np.any(np.round(atoms0.cell, 4) != np.round(atomsF.cell, 4)): #jedi does not work for pbc systems that change their cell shape
#            raise GeneratorExit

    def todict(self) -> Dict[str, Any]:
        '''make it saveable with .write()

        '''
        return {'atoms0': self.atoms0,
                'atomsF': self.atomsF,
                #'modes': self.modes,
                'hessian': self.H,
                'bmatrix': self.B,
                'delta_q': self.delta_q,
                'rim_list': self.rim_list,
                'energies': self.energies,
                'indices': self.indices,
                'E_RIMS': self.E_RIMs,
                'proc_E_RIMS': self.proc_E_RIMs,
                'custom_bonds': self.custom_bonds}
    @classmethod
    def fromdict(cls, data: Dict[str, Any]) -> 'Jedi':
        '''make it readable with .read()

        '''
        # mypy is understandably suspicious of data coming from a dict that
        # holds mixed types, but it can see if we sanity-check with 'assert'
        assert isinstance(data['atoms0'], Atoms)
        assert isinstance(data['atomsF'], Atoms)
        try:
            assert isinstance(data['modes'], VibrationsData)
            cl=cls(data['atoms0'], data['atomsF'], data['modes'])
        except:
            pass


        if data['hessian'] is not None:
            assert isinstance(data['hessian'], (collections.abc.Sequence,
                                                np.ndarray))

            if data['indices'] is not None:
                assert isinstance(data['indices'], (collections.abc.Sequence,
                                    np.ndarray))
                modes = VibrationsData.from_2d(data['atoms0'],data['hessian'],data['indices'])
                cl=cls(data['atoms0'], data['atomsF'],modes)
                cl.indices=data['indices']
            else:
                modes = VibrationsData.from_2d(data['atoms0'],data['hessian'])
                cl=cls(data['atoms0'], data['atomsF'],modes)
            cl.H = data['hessian']
        if data['bmatrix'] is not None:
            assert isinstance(data['bmatrix'], (collections.abc.Sequence,
                                                np.ndarray))
            cl.B = data['bmatrix']
        if data['delta_q'] is not None:
            assert isinstance(data['delta_q'], (collections.abc.Sequence,
                                                np.ndarray))
            cl.delta_q = data['delta_q']
        if data['rim_list'] is not None:
            assert isinstance(data['rim_list'], (collections.abc.Sequence,
                                                np.ndarray))
            cl.rim_list = data['rim_list']
        if data['energies'] is not None:
            assert isinstance(data['energies'], (collections.abc.Sequence,
                                                list))
            cl.energies = data['energies']
        if data['E_RIMS'] is not None:
            assert isinstance(data['proc_E_RIMS'], (collections.abc.Sequence,
                                                np.ndarray))
            cl.E_RIMs = data['E_RIMS']
        if data['proc_E_RIMS'] is not None:
            assert isinstance(data['proc_E_RIMS'], (collections.abc.Sequence,
                                                np.ndarray))
            cl.proc_E_RIMs = data['proc_E_RIMS']
        if data['custom_bonds'] is not None:
            assert isinstance(data['custom_bonds'], (collections.abc.Sequence,
                                                list))
        return cl

    def run(self,indices=None,ase_units=False):
        """Runs the analysis. Calls all necessary functions to get the needed values.

        Args:
            indices: 
                list of indices of a substructure if desired
            ase_units: boolean
                flag to get eV for energies å fo lengths and degree for angles otherwise it is kcal/mol, Bohr and radians 
        Returns:
            Indices, strain, energy in every RIM
        """
        self.ase_units = ase_units
        # get necessary data
        self.indices=np.arange(0,len(self.atoms0))
        self.get_common_rims()
        rim_list = self.rim_list
        self.get_b_matrix()
        B = self.B
        self.get_delta_q()
        delta_q = self.delta_q
        self.get_hessian()
        H_cart = self.H         #Hessian of optimized (ground state) structure
        if len(self.atoms0) != H_cart.shape[0]/3:

            raise ValueError('Hessian has not the fitting shape, possibly a partial hessian. Please try partial_analysis')
        try:
            all_E_geometries = self.get_energies()
        except:
            all_E_geometries = self.energies
        E_geometries=all_E_geometries[0]


        #run the analysis
        self.proc_E_RIMs,self.E_RIMs,E_RIMs_total,proc_geom_RIMs,self.delta_q = jedi_analysis(self.atoms0,rim_list,B,H_cart,delta_q,E_geometries,ase_units=ase_units)

        if indices:          #get only rims of interest
            self.post_process(indices)
            E_RIMs_total = sum(self.E_RIMs)
            proc_geom_RIMs = 100*(sum(self.E_RIMs)-E_geometries)/E_geometries

        jedi_printout(self.atoms0,self.rim_list,self.delta_q,E_geometries, E_RIMs_total, proc_geom_RIMs,self.proc_E_RIMs, self.E_RIMs,ase_units=ase_units)
        pass





    def get_rims(self,mol):
        '''Gets the redundant internal coordinates

        '''
        ###bondlengths####
        mol = mol

        indices = self.indices
        cutoff = ase.neighborlist.natural_cutoffs(mol,mult=self.covf)   ## cutoff for covalent bonds see Bakken et al.
        bl = np.vstack(ase.neighborlist.neighbor_list('ij',a=mol,cutoff=cutoff)).T   #determine covalent bonds

        bl=bl[bl[:,0]<bl[:,1]]      #remove double metioned
        bl, counts = np.unique(bl,return_counts=True,axis=0)
        if ~ np.all(counts == 1):
            print('unit cell too small hessian not calculated for self interaction \
                   jedi analysis for a finite system consisting of the cell will be conducted')
        bl = np.atleast_2d(bl)

        if  len(indices) != len(mol):
            bl = bl[np.all([np.in1d(bl[:,0], indices),  np.in1d(bl[:,1], indices)],axis=0)]

        rim_list = [bl]


        #possibility of adding custom bonds like hbonds, long range interactions
        if self.custom_bonds is not None:
            try:
                bl=np.vstack((bl,self.custom_bonds))
            except:
                ValueError('custom bonds not in the correct format. 2D array needed with shape (x,2)')
            rim_list.append(self.custom_bonds)
        if self.custom_bonds is None:
            rim_list.append(np.array([]))


        ########find angles
        #create array containing all angles (ba)
        ba_flag=False
        row_index=0
        for self_index, self_row in enumerate(bl): # iterates through rows of bonds
            for other_index, other_row in enumerate(bl): # iterates through rows of bonds
                if other_index > self_index:
                    temp_ba_list = [self_row[0], self_row[1], other_row[0], other_row[1]]
                    temp_ba_counter = Counter(temp_ba_list) # counts all entries in temporary bondangle list, counts duplicates
                    connecting_atom = list([item for item in temp_ba_counter if temp_ba_counter[item]>1]) # checks which atom is duplicate
                    other_atoms = [] # list collecting other atoms than connecting atom
                    if connecting_atom: # duplicate atom is connecting atom
                        for atom in temp_ba_list:
                            if atom not in connecting_atom:
                                other_atoms.append(atom)
                        if row_index==0:
                            ba = np.array([other_atoms[0], connecting_atom[0], other_atoms[1]])
                            ba_flag = True
                        else:
                            ba = np.vstack((ba, [other_atoms[0], connecting_atom[0], other_atoms[1]])) # add bondlengths to dataframe
                        row_index += 1


        if ba_flag == True :
            ba = np.atleast_2d(ba)
            ba = ba[ba[:, 1].argsort()]  #sort by atom2
            ba = ba[ba[:, 0].argsort(kind='mergesort')]  # sort by atom1

            nan=np.full((len(ba),1),-1)
            nan=np.hstack((nan,ba))
            rim_list.append(ba)
        else:
            rim_list.append(np.array([]))



        ###torsion angles###########


        tb_flag=False
        row_index = 0
        #create dataframe containing list of all bonds with torsion angles (df_torsionable_bonds)
        for self_index, self_row in enumerate(bl): # iterates through rows of bonds
            bond_partner1 = False # if both bond partners are set to True, no terminal bond. Thus, possible torsion around bond.
            bond_partner2 = False
            for other_index, other_row in enumerate(bl): # iterates through rows of bonds
                if other_index != self_index: # only iterate bonds other than self
                    if other_row[0] == self_row[0] or other_row[1] == self_row[0]: # Check first Atom
                        bond_partner1 = True # Set to True if neighbouring atom

                    if other_row[0] == self_row[1] or other_row[1] == self_row[1]: # Check second Atom#
                        bond_partner2 = True # Set to True if neighbouring atom

                    if bond_partner1 == True and bond_partner2 == True: # if both bond partners are set to True, no terminal bond. Thus, possible torsion around bond.
                        if row_index == 0:
                            torsionable_bonds=np.array([self_row[0],self_row[1]])
                            tb_flag = True
                        else:
                            torsionable_bonds=np.vstack((torsionable_bonds, [self_row[0],self_row[1]]))
                        bond_partner1 = False
                        bond_partner2 = False
                        row_index += 1
                        break
        if tb_flag == True:
            da_flag = False
            torsionable_bonds = np.atleast_2d(torsionable_bonds)
            row_index = 0
            for torsionable_row in torsionable_bonds:
                TA_Atoms_0 = []
                TA_Atoms_3 = []
                TA_Atom_0 = False # atom connected to TA_Atom_1
                TA_Atom_3 = False # atom connected to TA_Atom_2
                for  other_row in bl: # iterates through rows of bonds

                    if other_row[0] == torsionable_row[0] and other_row[1] == torsionable_row[1]:
                        continue

                    ### FIRST ATOM CONNECTION
                    elif other_row[0] == torsionable_row[0] or other_row[1] == torsionable_row[0]:

                        if other_row[0] == torsionable_row[0]:
                            TA_Atom_0 = other_row[1]

                        else:
                            TA_Atom_0 = other_row[0]
                        TA_Atoms_0.append(TA_Atom_0)

                    ### SECOND ATOM CONNECTION
                    if other_row[0] == torsionable_row[1] or other_row[1] == torsionable_row[1]:
                        if other_row[0] == torsionable_row[1]:
                            TA_Atom_3 = other_row[1]

                        else:
                            TA_Atom_3 = other_row[0]
                        TA_Atoms_3.append(TA_Atom_3)

                for single_TA_Atom_0 in TA_Atoms_0:
                    for single_TA_Atom_3 in TA_Atoms_3:
                        da_pre=np.atleast_2d(np.array([single_TA_Atom_0,  torsionable_row[0], torsionable_row[1], single_TA_Atom_3]))

                        if len(np.unique(da_pre[0],axis=0)) != len(da_pre[0]):
                            print('bonds for dihedral angle span over more than one unit cell\n %s will not be taken into account in the further analysis'%(da_pre))


                        else:
                            try:
                                if round(mol.get_angle(int(single_TA_Atom_0),int(torsionable_row[0]),int(torsionable_row[1]),mic=True)) in [0.0,180.0,360.0] or \
                                    round(mol.get_angle(int(torsionable_row[0]),int(torsionable_row[1]),int(single_TA_Atom_3),mic=True)) in [0.0,180.0,360.0] :    #check for linear angles
                                    continue
                                if row_index == 0:
                                    da = np.array([single_TA_Atom_0,  torsionable_row[0], torsionable_row[1], single_TA_Atom_3])
                                    da_flag=True
                                else:
                                    da = np.vstack((da,[single_TA_Atom_0,  torsionable_row[0], torsionable_row[1], single_TA_Atom_3]))

                                row_index += 1
                            except:
                                continue

            if da_flag==True:
                rim_list.append(da)
            else:
                rim_list.append(np.array([]))
        else:
            rim_list.append(np.array([]))


        return rim_list

    def get_common_rims(self):
        '''Get only the RICs in both structures bond breaks cannot be analysed logically

        '''
        rim_atoms0 = self.get_rims(self.atoms0)
        rim_atomsF = self.get_rims(self.atomsF)

        for i in range(len(rim_atoms0)):
            if rim_atoms0[i].shape[0]==0 or rim_atomsF[i].shape[0]==0:
                break
            else:
                rim_atoms0v = rim_atoms0[i].view([('', rim_atoms0[i].dtype)] * rim_atoms0[i].shape[1]).ravel()
                rim_atomsFv = rim_atomsF[i].view([('', rim_atomsF[i].dtype)] * rim_atomsF[i].shape[1]).ravel()    #get a viable input for np.intersect1d()

                rim_l,ind,z = np.intersect1d(rim_atoms0v, rim_atomsFv,return_indices=True)    #get the rims that exist in both structures
                rim_l = rim_l[ind.argsort()]

                rim_atoms0[i] = rim_l.view(rim_atoms0[i].dtype).reshape(-1, rim_atoms0[i].shape[1])
                self.rim_list = rim_atoms0

        return rim_atoms0

    def get_hessian(self):
        '''Calls the hessian from the VibrationsData object
        '''
        hessian = self.modes._hessian2d
        self.H = hessian /(Hartree/Bohr**2)
        return hessian

    def get_b_matrix(self,indices=None):
        '''Calculates the derivatives of the RICs with respect to all cartesian coordinates using ase functions

        '''
        mol = self.atoms0
        if indices == None:
            indices = np.arange(0,len(mol))
        if  len(self.rim_list) == 0:
            self.get_common_rims()

        rim_size = sum([np.shape(l)[0] for l in self.rim_list])
        b = np.zeros([int(len(indices)*3), int(rim_size)], dtype=float)   #shape of B-matrix (NCarts,NRIMs)

        # get all derivatives 
        column = 0 # Initilization of columns to specifiy position in B-Matrix
        for q in self.rim_list[0]:
            row = 0  # Initilization of rows to specifiy position in B-Matrix

            ########  Section for stretches  #########


            BL = []
            BL = [int(q[0]), int(q[1])]  # create list of involved atoms
            q_i, q_j = BL

            u = mol.get_distance(q_i,q_j,mic=True,vector=True)
            for NAtom in indices:  # for-loop of Number of Atoms 

                for q in BL:
                    if NAtom == q:  # derivative of redundnat internal coordinate w/ respect to cartesian coordinates is not equal zero
                                    # if redundant internal coordinate (q) contains the Atomnumber (NAtoms) of the cartesian coordinate (x0_coords) from which is derived from.

                        # if-/elif-statement for the right sign-factor (see [1])
                        if q == q_i:
                            b_i = ase.geometry.get_distances_derivatives(np.atleast_2d(u))[0][0]
                            b[row:row+3, column] = b_i  # change value of zero array at specified position to b_i
                        elif q == q_j:
                            b_i = ase.geometry.get_distances_derivatives(np.atleast_2d(u))[0][1]
                            b[row:row+3, column] = b_i  # change value of zero array at specified position to b_i
                row += 3
            column += 1

        for q in self.rim_list[1]:
            row = 0  # Initilization of rows to specifiy position in B-Matrix

            ########  Section for custom stretches  #########


            CL = []
            CL = [int(q[0]), int(q[1])]  # create list of involved atoms
            q_i, q_j = CL

            u = mol.get_distance(q_i,q_j,mic=True,vector=True)
            for NAtom in indices:  # for-loop of Number of Atoms

                for q in CL:
                    if NAtom == q:
                        # if-/elif-statement for the right sign-factor 
                        if q == q_i:
                            b_i = ase.geometry.get_distances_derivatives(np.atleast_2d(u))[0][0]
                            b[row:row+3, column] = b_i  # change value of zero array at specified position to b_i
                        elif q == q_j:
                            b_i = ase.geometry.get_distances_derivatives(np.atleast_2d(u))[0][1]
                            b[row:row+3, column] = b_i  # change value of zero array at specified position to b_i

                row += 3
            column += 1




    #################ba###############################


        for q in self.rim_list[2]:
            BA = []
            row = 0 # Initilization of rows to specifiy position in B-Matrix

            BA = [int(q[0]), int(q[1]), int(q[2])]  # create list of involved atoms
            q_i, q_j, q_k = BA
            u = mol.get_distance(q_i,q_j,mic=True,vector=True)
            v = mol.get_distance(q_k,q_j,mic=True,vector=True)



            def get_B_matrix_angles_derivatives(u,v):
                angle = ase.geometry.get_angles(u,v) # angle between v and u

                if angle == 180 or angle==0:   #an auxilliary vector is used if linear angles are existing
                    (u, v), (lu, lv) = ase.geometry.conditional_find_mic([u, v],cell=None,pbc=None)
                    nu = u / lu
                    nv = v / lv
                    if (np.arccos(np.dot(nu, (np.array([1, -1, 1]))))) == np.pi:
                        w = np.cross(nu, ([-1, 1, 1]))
                    else:
                        w = np.cross(nu, ([1, -1, 1]))

                    nw = w / np.linalg.norm(w)
                    d_ba1 = (((np.cross(nu, nw))/np.linalg.norm(u)))
                    d_ba2 = (-1 * ((np.cross(nu, nw))/np.linalg.norm(u))) + (-1 * ((np.cross(nw, nv))/np.linalg.norm(v))) # equation to calculate dBA/dx [1]
                    d_ba3 = ((np.cross(nw, nv))/np.linalg.norm(v))
                    d_ba=np.array([[d_ba1[0], d_ba2[0], d_ba3[0]]])

                else:

                    d_ba=np.radians(ase.geometry.get_angles_derivatives(u,v))
                return d_ba*Bohr

            for NAtom in indices:  # for-loop of Number of Atoms 

                for q in BA:
                    if NAtom == q:
                        b_j = 0
                        if q == q_j:  # if-Statements for sign-factors
                            b_j =  get_B_matrix_angles_derivatives(np.atleast_2d(u),np.atleast_2d(v))[0][1]
                            b[row:row+3, column] = -b_j
                        elif q == q_i:
                            b_j = get_B_matrix_angles_derivatives(np.atleast_2d(u),np.atleast_2d(v))[0][0]
                            b[row:row+3, column] = -b_j
                        elif q == q_k:
                            b_j = get_B_matrix_angles_derivatives(np.atleast_2d(u),np.atleast_2d(v))[0][2]
                            b[row:row+3, column] = -b_j
                row += 3
            column += 1





        for q in self.rim_list[3]:
            DA = []
            row = 0 # Initilization of rows to specifiy position in B-Matrix

            DA = [int(q[0]), int(q[1]), int(q[2]), int(q[3])]  # create list of involved atoms
            q_i, q_j, q_k, q_l = DA

            u = np.copy(np.atleast_2d(mol.get_distance(q_i,q_j,mic=True,vector=True))) #####copy needed because derivative function rewrites vector variable as normed vector
            w = np.copy(np.atleast_2d(mol.get_distance(q_k,q_l,mic=True,vector=True)))
            v = np.copy(np.atleast_2d(mol.get_distance(q_j,q_k,mic=True,vector=True)))


            for NAtom in indices:  # for-loop of Number of Atoms

                for q in DA:

                    if NAtom == q:
                        b_k = 0

                        if q == q_i:  # if-Statements for sign-factors
                            b_k = np.radians(ase.geometry.get_dihedrals_derivatives(u, v, w)[0][0])*Bohr
                            b[row:row+3, column] = b_k
                            u = np.copy(np.atleast_2d(mol.get_distance(q_i,q_j,mic=True,vector=True))) #####copy needed because derivative function rewrites vector variable as normed vector
                            w = np.copy(np.atleast_2d(mol.get_distance(q_k,q_l,mic=True,vector=True)))
                            v = np.copy(np.atleast_2d(mol.get_distance(q_j,q_k,mic=True,vector=True)))

                        elif q == q_j:
                            b_k = np.radians(ase.geometry.get_dihedrals_derivatives(u, v, w)[0][1])*Bohr
                            b[row:row+3, column] = b_k
                            u = np.copy(np.atleast_2d(mol.get_distance(q_i,q_j,mic=True,vector=True))) #####copy needed because derivative function rewrites vector variable as normed vector
                            w = np.copy(np.atleast_2d(mol.get_distance(q_k,q_l,mic=True,vector=True)))
                            v = np.copy(np.atleast_2d(mol.get_distance(q_j,q_k,mic=True,vector=True)))

                        elif q == q_k:
                            b_k = np.radians(ase.geometry.get_dihedrals_derivatives(u, v, w)[0][2])*Bohr
                            b[row:row+3, column] = b_k
                            u = np.copy(np.atleast_2d(mol.get_distance(q_i,q_j,mic=True,vector=True))) #####copy needed because derivative function rewrites vector variable as normed vector
                            w = np.copy(np.atleast_2d(mol.get_distance(q_k,q_l,mic=True,vector=True)))
                            v = np.copy(np.atleast_2d(mol.get_distance(q_j,q_k,mic=True,vector=True)))

                        elif q == q_l:
                            b_k = np.radians(ase.geometry.get_dihedrals_derivatives(u, v, w)[0][3])*Bohr
                            b[row:row+3, column] = b_k
                            u = np.copy(np.atleast_2d(mol.get_distance(q_i,q_j,mic=True,vector=True))) #####copy needed because derivative function rewrites vector variable as normed vector
                            w = np.copy(np.atleast_2d(mol.get_distance(q_k,q_l,mic=True,vector=True)))
                            v = np.copy(np.atleast_2d(mol.get_distance(q_j,q_k,mic=True,vector=True)))
                row += 3
            column += 1

        B = np.transpose(b)
        self.B = B

        return B

    def get_energies(self):
        '''Calls the energies of the Atoms objects.

            Returns: 
                [energy difference, energy of atoms0, energy of atomsF]

        '''
        e0 = self.atoms0.calc.get_potential_energy()
        eF = self.atomsF.calc.get_potential_energy()
        if self.ase_units==False:
            e0*=mol/kcal
            eF*=mol/kcal
        deltaE = eF - e0
        self.energies=[deltaE, eF, e0]
        return [deltaE, eF, e0]

    def get_delta_q(self):
        '''get the strain in RICs substracts the values of the relaxed structure from the strained structure
        
            Returns: 
                2D array of the values.
        '''

        try:
            len(self.rim_list)
        except:
            self.get_common_rims()

        if  len(self.B) == 0:
            self.get_b_matrix()
        B = self.B
        q0 = []
        qF = []
        dq_da = []

  # for loops for all redunant internal coordinates

        #bonds
        for q in self.rim_list[0]:
            q0.append(self.atoms0.get_distance(int(q[0]),int(q[1]),mic=True)/Bohr)
            qF.append(self.atomsF.get_distance(int(q[0]),int(q[1]),mic=True)/Bohr)
        #custom bonds
        for q in self.rim_list[1]:
            q0.append(self.atoms0.get_distance(int(q[0]),int(q[1]),mic=True)/Bohr)
            qF.append(self.atomsF.get_distance(int(q[0]),int(q[1]),mic=True)/Bohr)
        #angles
        for q in self.rim_list[2]:
            q0.append(np.radians(self.atoms0.get_angle(int(q[0]),int(q[1]),int(q[2]),mic=True)))
            qF.append(np.radians(self.atomsF.get_angle(int(q[0]),int(q[1]),int(q[2]),mic=True)))
        #dihedral angles
        for q in self.rim_list[3]:
            q0_preliminary=np.radians(self.atoms0.get_dihedral(int(q[0]),int(q[1]),int(q[2]),int(q[3]),mic=True))
            qF_preliminary=np.radians(self.atomsF.get_dihedral(int(q[0]),int(q[1]),int(q[2]),int(q[3]),mic=True))

        # get the smallest absolute value of the two possible rotational directions
            dda=qF_preliminary-q0_preliminary
            if 2*np.pi-abs(dda)<abs(dda):
                dda = (2*np.pi-abs(dda))*-np.sign(dda)
            dq_da.append(dda)

        delta_q = np.subtract(qF, q0)

        try:
            delta_q=np.append(delta_q,dq_da)
        except:
            pass

        self.delta_q = delta_q

        return delta_q

    def vmd_gen(self,
                des_colors: Optional[Dict] = None,
                box: bool = False,
                man_strain: Optional[float] = None,
                modus: Optional[str] = None,
                colorbar: bool = True,
                label: Union[Path, str] = 'vmd'):
        """
        Generates vmd scripts and files to save the values for the color coding

        Args:
            des_colors: (dict)
                key: order number, value: [R,G,B]
            box: boolean
                True: draw box
                False: ignore box
            man_strain: float
                reference value for the strain energy used in the color scale
                default: 'None'
            modus: str
                defines where to use the man_strain
                default: 'None'
            colorbar: boolean
                draw colorbar or not
            label: string or pathlib.Path
                name of folder for the created files
        """
        if isinstance(label, str):
            destination_dir = Path(label)
        elif isinstance(label, Path):
            destination_dir = label
        else:
            raise TypeError("Please specify the directory (label) to write vmd scripts to as Path or string")
        destination_dir.mkdir(parents=True, exist_ok=True)

        #########################
        #       Basic stuff     #
        #########################
        if man_strain is not None and modus is None:
            print('\nPlease set a modus otherwise man_strain will be neglected')
        if not self.ase_units:
            unit = "kcal/mol"
        elif self.ase_units:
            unit = "eV"
        rim_list = self.rim_list
        self.atomsF.write(destination_dir / 'xF.xyz')
        if len(self.proc_E_RIMs) == 0:
            self.run()
        proc_E_RIMs = self.proc_E_RIMs
        pbc_flag = False
        if self.atomsF.get_pbc().any():
            pbc_flag=True
        # Check whether we need to write ba, da and all and read basic stuff
        file_list = []
        bl = []
        ba = []
        da = []
        ba_flag = False
        da_flag = False

        for i in rim_list[0]:
        # Bond lengths (a molecule has at least one bond):
            numbers = [int(i[0]),int(i[1])]
            bl.append(numbers)
            if 'bl' not in file_list:
                file_list.append('bl')
            # All (to sum up the values with angles and dihedrals:
                file_list.append('all')

        # custom bonds
        for i in rim_list[1]:
            numbers = [int(i[0]), int(i[1])]
            bl.append(numbers)

        # Bond angles:
        for i in rim_list[2]:
            ba_flag = True
            numbers = [int(i[0]), int(i[1]), int(i[2])]
            ba.append(numbers)
            if 'ba' not in file_list:
                file_list.append('ba')

        # Dihedral angles:
        for i in rim_list[3]:
            da_flag = True
            numbers = [int(n) for n in i]
            da.append(numbers)
            if 'da' not in file_list:
                file_list.append('da')

        # percental energy of RIMs
        E_RIMs_perc = np.array(proc_E_RIMs)
        E_RIMs = self.E_RIMs

        # Write some basic stuff to the tcl scripts
        output = [[], [], [], []]
        for outindex, filename in enumerate(file_list):
            if filename == "bl" or filename == "ba" or filename == "da" or filename == "all":

                output[outindex].append(f'\n# Load a molecule\nmol new {destination_dir.resolve() / "xF.xyz"}\n\n')
                output[outindex].append('\n# Change bond radii and various resolution parameters\nmol representation '
                                        'cpk 0.8 0.0 30 5\nmol representation bonds 0.2 30\n\n')
                output[outindex].append('\n# Change the drawing method of the first graphical representation to '
                                        'CPK\nmol modstyle 0 top cpk\n')
                output[outindex].append('\n# Color only H atoms white\nmol modselect 0 top {name H}\n')
                output[outindex].append('\n# Change the color of the graphical representation 0 to white\ncolor '
                                        'change rgb 0 1.00 1.00 1.00\nmol modcolor 0 top {colorid 0}\n')
                output[outindex].append('\n# The background should be white ("blue" has the colorID 0, which we have '
                                        'changed to white)\ncolor Display Background blue\n\n')
                output[outindex].append('\n# Define the other colorIDs\n')

        # Define colorcodes for various atomtypes
        if des_colors is not None:
            for i in des_colors:
                colors[i]=des_colors[i]         # desired colors overwrite the standard ones

        symbols = np.unique(self.atomsF.get_chemical_symbols())
        symbols = symbols[symbols!='H']           # get all symbols except H, H is white

        N_colors_atoms = len(symbols)
        N_colors = 32 - N_colors_atoms - 1           # vmd only supports 32 colors for modcolor

        # Generate the color-code and write it to the tcl scripts
        for outindex, filename in enumerate(file_list):
            if filename == "bl" or filename == "ba" or filename == "da" or filename == "all":

                colorbar_colors = []

                # get green to red gradient
                for i in range(N_colors):
                    R_value = float(i)/(N_colors/2)
                    if R_value > 1:
                        R_value = 1
                    if N_colors % 2 == 0:
                        G_value = 2 - float(i+1)/(N_colors/2)
                    if N_colors % 2 != 0:
                        G_value = 2 - float(i)/(N_colors/2)
                    if G_value > 1:
                        G_value = 1

                    B_value = 0

                    output[outindex].append('%1s%5i%10.6f%10.6f%10.6f%1s'
                                            % ("color change rgb", i+1, R_value, G_value, B_value, "\n"))
                    colorbar_colors.append((R_value, G_value, B_value))

                # add color codes of atoms
                for j in range(N_colors_atoms):
                    output[outindex].append('\n%1s%5i%10.6f%10.6f%10.6f%1s'
                                            % ("color change rgb",
                                               N_colors+j+1,
                                               float(colors[symbols[j]][0]),
                                               float(colors[symbols[j]][1]),
                                               float(colors[symbols[j]][2]), "\n"))

                # add color code for axes and box
                output[outindex].append('\n%1s%5i%10.6f%10.6f%10.6f%1s'
                                        % ("color change rgb", 32, float(0), float(0), float(0), "\n"))     #black
                output[outindex].append('\n%1s%5i%10.6f%10.6f%10.6f%1s'
                                        % ("color change rgb", 1039, float(1), float(0), float(0), "\n"))   #red
                output[outindex].append('\n%1s%5i%10.6f%10.6f%10.6f%1s'
                                        % ("color change rgb", 1038, float(0), float(1), float(0), "\n"))   #green
                output[outindex].append('\n%1s%5i%10.6f%10.6f%10.6f%1s'
                                        % ("color change rgb", 1037, float(0), float(0), float(1), "\n"))   #blue
                output[outindex].append('\n%1s%5i%10.6f%10.6f%10.6f%1s'
                                        % ("color change rgb", 1036, float(0.25), float(0.75), float(0.75), "\n"))  #cyan
                output[outindex].append("\ncolor Axes X 1039\ncolor Axes Y 1038\ncolor Axes Z 1037\ncolor Axes Origin "
                                        "1036\ncolor Axes Labels 32")

                # define color of atoms with the color code above
                for j in range(N_colors_atoms):
                    output[outindex].append('\n\nmol representation cpk 0.7 0.0 30 5')
                    output[outindex].append('\nmol addrep top')
                    output[outindex].append('\n%s%i%s' % ("mol modstyle ", j+1, " top cpk"))
                    output[outindex].append('\n%s%i%s%i%s' % ("mol modcolor ", j+1, " top {colorid ", N_colors+j+1, "}"))
                    output[outindex].append('\n%s%i%s%s%s' % ("mol modselect ", j+1, " top {name ", symbols[j], "}"))

        #########################
        #	Binning		#
        #########################
        # Welcome
        print("\n\nCreating tcl scripts for generating color-coded structures in VMD...")
        if len(self.indices) < len(self.atomsF):  # if there are only values for a substructure
            p_rim = self.rim_list.copy()          # store rims that were analyzed
            p_indices = self.indices
            self.indices = range(len(self.atomsF))

            rim = self.get_common_rims().copy()   # get rims of whole structure to show the whole structure

            for i in range(2):
                if rim[i].shape[0] == 0:
                    break

                rim[i] = np.ascontiguousarray(rim[i])
                a = np.array(rim_list[i]).view([('', np.array(rim_list[i]).dtype)] * np.array(rim_list[i]).shape[1]).ravel()
                b = np.array(rim[i]).view([('', np.array(rim_list[i]).dtype)] * np.array(rim_list[i]).shape[1]).ravel()
                rim[i] = np.setxor1d(a, b)
                rim[i] = rim[i].view(np.array(rim_list[i]).dtype).reshape(-1, 2)  # get unconsidered rims
                nan = np.full((len(rim[i]), 1), np.nan)         # nan for special color (black)
                rim[i] = np.hstack((rim[i], nan))              # stack unanalyzed rims for later vmd visualization
            bond_E_array_app = rim

            self.indices = p_indices #TODO ? see line 1068

        # get bonds that reach out of the unit cell
        if pbc_flag:
            bond_E_array_pbc = [np.empty((0, 2)), np.empty((0, 2))]
            bond_E_array_pbc_trans = [np.empty((0, 2)), np.empty((0, 2))] # initialize list

            from ase.data.vdw import vdw_radii # for long range bonds
            cutoff = [vdw_radii[atom.number] * self.vdwf for atom in self.atomsF]
            ex_bl = np.vstack(ase.neighborlist.neighbor_list('ij', a=self.atomsF, cutoff=cutoff)).T
            ex_bl = np.hstack((ex_bl, ase.neighborlist.neighbor_list('S', a=self.atomsF, cutoff=cutoff)))
            ex_bl = np.hstack((ex_bl, ase.neighborlist.neighbor_list('D', a=self.atomsF, cutoff=cutoff)))
            atoms_ex_cell = ex_bl[(ex_bl[:, 2] != 0) | (ex_bl[:, 3] != 0) | (ex_bl[:, 4]!= 0)]  # determines which
            # nearest neighbors are outside the unit cell
            mol = self.atomsF.copy()     # an extended cell is needed for vmd since it does not show intercellular bonds
            mol.wrap()                   # wrap molecule important for atoms close to the boundaries

            # check if bond or custom bond
            bondscheck = self.rim_list[0][:, (0, 1)]
            if self.rim_list[1].shape[0] != 0:
                customcheck = self.rim_list[1][:, (0, 1)]

            for i in range(len(atoms_ex_cell)):
                # get positions of cell external atoms by adding the vector
                pos_ex_atom = mol.get_positions()[int(atoms_ex_cell[i,0])]+atoms_ex_cell[i,5:8]
                #if pos_ex_atom in mol.positions:
                # get the indices of the corresponding atoms inside the cell
                original_rim = [int(atoms_ex_cell[i, 0]), int(atoms_ex_cell[i, 1])]
                original_rim.sort()     # needs to be sorted because rim list only covers one direction
                if len(np.where(np.all(mol.positions == pos_ex_atom, axis=1))[0]) > 0:
                    ex_ind = np.where(np.all(mol.positions == pos_ex_atom, axis=1))[0][0]
                else:
                    ex_ind = len(mol)
                    # TODO how often is len(np.where(np.all())) check exec
                    if len(np.where(np.all(original_rim == bondscheck, axis=1))[0]) > 0 \
                            or (self.rim_list[1].shape[0] != 0
                                and len(np.where(np.all(original_rim == customcheck, axis=1))[0]) > 0):
                        # append to the virtual atoms object
                        mol.append(Atom(symbol=mol.symbols[int(atoms_ex_cell[i, 1])], position=pos_ex_atom))

                if len(np.where(np.all(original_rim == bondscheck, axis=1))[0]) > 0:
                    # add to bond list with auxiliary index
                    bond_E_array_pbc[0] = np.append(bond_E_array_pbc[0], [[atoms_ex_cell[i, 0], ex_ind]], axis=0)
                    bond_E_array_pbc_trans[0] = np.append(bond_E_array_pbc_trans[0], [original_rim], axis=0)

                elif self.rim_list[1].shape[0] != 0 and len(np.where(np.all(original_rim == customcheck, axis=1))[0]) > 0:
                    # add to bond list with auxiliary index
                    bond_E_array_pbc[1] = np.append(bond_E_array_pbc[1], [[atoms_ex_cell[i, 0], ex_ind]], axis=0)
                    bond_E_array_pbc_trans[1] = np.append(bond_E_array_pbc_trans[1], [original_rim], axis=0)

            mol.write(destination_dir / 'xF.xyz')  # save the modified structure with auxilliary atoms for vmd

        if len(self.indices) < len(self.atomsF):
            self.rim_list = p_rim                 # restore the partial rim list # TODO class attribute is changed?

        # Achieve the binning for bl, ba, da an all simultaneously
        for outindex, filename in enumerate(file_list):
            if filename == "bl" or filename == "ba" or filename == "da" or filename == "all":

                # Create an array that stores the bond connectivity as the first two entries.
                # The energy will be added as the third entry.
                E_array = np.full((len(bl), 3), np.nan)
                for i in range(len(bl)):
                    E_array[i][0] = bl[i][0]
                    E_array[i][1] = bl[i][1]

            # Create an array that stores only the energies in the coordinate of interest and print some information
            # Get rid of ridiculously small values and treat diatomic molecules explicitly
            # (in order to create a unified picture, we have to create all these arrays in any case)

            # Bonds
                if filename == "bl" or filename == "all":
                    if len(bl) == 1:
                        E_bl_perc = E_RIMs_perc[0]
                        E_bl = E_RIMs
                    else:
                        E_bl_perc = E_RIMs_perc[0:len(bl)]
                        E_bl = E_RIMs[0:len(bl)]
                        if E_bl_perc.max() <= 0.001:
                            E_bl_perc = np.zeros(len(bl))
                    if filename == "bl":
                        print("\nProcessing bond lengths...")
                        print("%s%6.2f%s" % ("Maximum energy in a bond length:      ", E_bl_perc.max(), '%'))
                        print("%s%6.2f%s" % ("Total energy in the bond lengths:     ", E_bl_perc.sum(),'%'))

            # Bendings
                if (filename == "ba" and ba_flag == True) or (filename == "all" and ba_flag == True):
                    E_ba_perc = E_RIMs_perc[len(bl):len(bl)+len(ba)]
                    E_ba = E_RIMs[len(bl):len(bl)+len(ba)]
                    if E_ba_perc.max() <= 0.001:
                        E_ba_perc = np.zeros(len(ba))
                    if filename == "ba":
                        print("\nProcessing bond angles...")
                        print("%s%6.2f%s" % ("Maximum energy in a bond angle:       ", E_ba_perc.max(), '%'))
                        print("%s%6.2f%s" % ("Total energy in the bond angles:      ", E_ba_perc.sum(),'%'))

             # Torsions (handle stdout separately)
                if (filename == "da" and da_flag == True ) or (filename == "all" and da_flag == True):
                    E_da_perc = E_RIMs_perc[len(bl)+len(ba):len(bl)+len(ba)+len(da)]
                    E_da = E_RIMs[len(bl)+len(ba):len(bl)+len(ba)+len(da)]
                    if E_da_perc.max() <= 0.001:
                        E_da_perc = np.zeros(len(da))
                if filename == "da" and da_flag == True:
                    print("\nProcessing dihedral angles...")
                    print("%s%6.2f%s" % ("Maximum energy in a dihedral angle:   ", E_da_perc.max(), '%'))
                    print("%s%6.2f%s" % ("Total energy in the dihedral angles:  ", E_da_perc.sum(),'%'))

            # Map onto the bonds (create "all" on the fly and treat diatomic molecules explicitly)
            # Bonds (trivial)
                if filename == "bl" or filename == "all":
                    for i in range(len(bl)):
                        if len(bl) == 1:
                            E_array[i][2] = E_bl[i]
                        else:
                            E_array[i][2] = E_bl[i]

            # Bendings
                if (filename == "ba" and ba_flag == True) or (filename == "all" and ba_flag == True):
                    for i in range(len(ba)):
                        for j in range(len(bl)):
                            # look for the right connectivity
                            if ((ba[i][0] == bl[j][0] and ba[i][1] == bl[j][1])
                                    or (ba[i][0] == bl[j][1] and ba[i][1] == bl[j][0])
                                    or (ba[i][1] == bl[j][0] and ba[i][2] == bl[j][1])
                                    or (ba[i][1] == bl[j][1] and ba[i][2] == bl[j][0])):
                                E_array[j][2] += 0.5 * E_ba[i]
                                if np.isnan(E_array[j][2] ):
                                    E_array[j][2] = 0.5 * E_ba[i]

            # Torsions
                if (filename == "da" and da_flag == True) or ( filename == "all" and da_flag == True ):
                    for i in range(len(da)):
                        for j in range(len(bl)):
                            if ((da[i][0] == bl[j][0] and da[i][1] == bl[j][1])
                                    or (da[i][0] == bl[j][1] and da[i][1] == bl[j][0])
                                    or (da[i][1] == bl[j][0] and da[i][2] == bl[j][1])
                                    or (da[i][1] == bl[j][1] and da[i][2] == bl[j][0])
                                    or (da[i][2] == bl[j][0] and da[i][3] == bl[j][1])
                                    or (da[i][2] == bl[j][1] and da[i][3] == bl[j][0])):
                                E_array[j][2] += (float(1)/3) * E_da[i]
                                if np.isnan(E_array[j][2]):
                                    E_array[j][2] = (float(1)/3) * E_da[i]

                if filename == "all" and rim_list[1].shape[0] != 0:
                    custom_E = sum(E_array[:, 2][len(bl)-len(self.custom_bonds):len(bl)])
                elif filename == "all" and rim_list[1].shape[0] == 0:
                    custom_E = np.nan

                custom_E_array = E_array[len(rim_list[0]):len(bl)]
                bond_E_array = E_array[0:len(rim_list[0])]

                if len(self.indices)<len(self.atomsF):
                    # stack bonds that were neglected before to show the whole structure
                    bond_E_array = np.vstack((bond_E_array, bond_E_array_app[0]))
                    try:
                        custom_E_array = np.vstack((custom_E_array, bond_E_array_app[1]))
                    except:
                        pass

            # get energies for bonds that reach out of the unit cell
                if pbc_flag:
                    translate={}       # the new bonds need to get the same values as the original ones inside the cell
                    for i in range(len(bond_E_array)):
                        translate[(np.min([bond_E_array[i, 0], bond_E_array[i, 1]]),
                                   np.max([bond_E_array[i, 0], bond_E_array[i, 1]]))] = bond_E_array[i, 2]
                    ctranslate={}
                    for i in range(len(custom_E_array)):
                        ctranslate[(np.min([custom_E_array[i, 0], custom_E_array[i, 1]]),
                                    np.max([custom_E_array[i, 0], custom_E_array[i, 1]]))] = custom_E_array[i, 2]

                    for i in range(len(bond_E_array_pbc[0])):
                        # get the indices of the corresponding atoms inside the cell
                        original_rim = bond_E_array_pbc_trans[0][i]
                        # add to bond list with auxillary index
                        bond_E_array = np.vstack((bond_E_array,
                                                  [int(bond_E_array_pbc[0][i][0]),
                                                   int(bond_E_array_pbc[0][i][1]),
                                                   translate[tuple(original_rim)]]))
                    for i in range(len(bond_E_array_pbc[1])):
                        # get the indices of the corresponding atoms inside the cell
                        original_rim = [int(bond_E_array_pbc_trans[1][i][0]), int(bond_E_array_pbc_trans[1][i][1])]
                        custom_E_array = np.delete(custom_E_array,
                                                   np.where((custom_E_array[:, 0] == original_rim[0]) &
                                                            (custom_E_array[:, 1] == original_rim[1]))[0], axis=0)
                        original_rim.sort()    # needs to be sorted because rim list only covers one direction
                        custom_E_array = np.vstack((custom_E_array,
                                                    [int(bond_E_array_pbc[1][i][0]),
                                                     int(bond_E_array_pbc[1][i][1]),
                                                     ctranslate[tuple(original_rim)]]))

            # Store the maximum energy in a variable for later call
            if filename == "all":
                max_energy = float(np.nanmax(E_array, axis=0)[2])  # maximum energy in one bond
                for row in E_array:

                    if max_energy in row:
                        atom_1_max_energy = int(row[0])
                        atom_2_max_energy = int(row[1])

            # Generate the binning windows by splitting bond_E_array into N_colors equal windows
            if filename == "all":
                if modus == "all":
                    if man_strain is None:
                        print(f"modus {modus} was called, but no maximum strain is given.")
                        binning_windows = np.linspace(0, np.nanmax(E_array, axis=0)[2], num=N_colors)
                    else:
                        binning_windows = np.linspace(0, float(man_strain), num=N_colors)
                else:
                    binning_windows = np.linspace(0, np.nanmax(E_array, axis=0)[2], num=N_colors)

            elif filename == "bl":
                if modus == "bl":
                    if man_strain is None:
                        print(f"modus {modus} was called, but no maximum strain is given.")
                        binning_windows = np.linspace(0, np.nanmax(E_array, axis=0)[2], num=N_colors)
                    else:
                        binning_windows = np.linspace(0, float(man_strain), num=N_colors)
                else:
                    binning_windows = np.linspace(0, np.nanmax(E_array, axis=0)[2], num=N_colors)

            elif filename == "ba":
                if modus == "ba":
                    if man_strain is None:
                        print(f"modus {modus} was called, but no maximum strain is given.")
                        binning_windows = np.linspace(0, np.nanmax(E_array, axis=0)[2], num=N_colors)
                    else:
                        binning_windows = np.linspace(0, float(man_strain), num=N_colors)
                else:
                    binning_windows = np.linspace(0, np.nanmax(E_array, axis=0)[2], num=N_colors)

            elif filename == "da":
                if modus == "da":
                    if man_strain is None:
                        print(f"modus {modus} was called, but no maximum strain is given.")
                        binning_windows = np.linspace(0, np.nanmax(E_array, axis=0)[2], num=N_colors)
                    else:
                        binning_windows = np.linspace(0, float(man_strain), num=N_colors)
                else:
                    binning_windows = np.linspace(0, np.nanmax(E_array, axis=0)[2], num=N_colors)
            else:
                binning_windows = np.linspace(0, np.nanmax(E_array, axis=0)[2], num=N_colors)

            if pbc_flag & box:
                output[outindex].append("\n\n# Adding a pbc box")
                output[outindex].append('\npbc set {%f %f %f %f %f %f}'
                                        % (self.atomsF.cell.cellpar()[0],
                                           self.atomsF.cell.cellpar()[1],
                                           self.atomsF.cell.cellpar()[2],
                                           self.atomsF.cell.cellpar()[3],
                                           self.atomsF.cell.cellpar()[4],
                                           self.atomsF.cell.cellpar()[5]))
                output[outindex].append("\npbc box -color 32")
                output[outindex].append("\n\n# Adding a representation with the appropriate colorID for each bond")

            # Calculate which binning_windows value is closest to the bond-percentage and do the output
            for i, b in enumerate(bond_E_array):
                if np.isnan(b[2]):
                    colorID = 32                       #black
                else:
                    colorID = np.abs(binning_windows - b[2]).argmin() + 1
                output[outindex].append('\nmol addrep top')
                output[outindex].append('\n%s%i%s' % ("mol modstyle ", N_colors_atoms+i+1, " top bonds"))
                output[outindex].append('\n%s%i%s%i%s'
                                        % ("mol modcolor ", N_colors_atoms+i+1, " top {colorid ", colorID, "}"))
                output[outindex].append('\n%s%i%s%i%s%i%s'
                                        % ("mol modselect ", N_colors_atoms+i+1, " top {index ", b[0], " ", b[1], "}\n"))
            for i in custom_E_array:
                if np.isnan(i[2]):
                    colorID = 32                       #black
                else:
                    colorID = np.abs( binning_windows - i[2]).argmin() + 1
                output[outindex].append('\nset x [[atomselect top "index %d %d"] get {x y z}]'%(i[0],i[1]))
                output[outindex].append('\nset a [lindex $x 0] ')
                output[outindex].append('\nset b [lindex $x 1] ')
                output[outindex].append('\ndraw  color %d' % colorID)
                output[outindex].append('\ndraw line  $a $b width 3 style dashed')

        #colorbar
            if colorbar:
                min = 0.000
                if filename == "all":
                    if modus == "all":
                        if man_strain is None:
                            max = np.nanmax(E_array, axis=0)[2]
                        else:
                            max = man_strain
                    else:
                        max = np.nanmax(E_array, axis=0)[2]

                elif filename == "bl":
                    if modus == "bl":
                        if man_strain is None:
                            max = np.nanmax(E_array, axis=0)[2]
                        else:
                            max = man_strain
                    else:
                        max = np.nanmax(E_array, axis=0)[2]

                elif filename == "ba":
                    if modus == "ba":
                        if man_strain is None:
                            max = np.nanmax(E_array, axis=0)[2]
                        else:
                            max = man_strain
                    else:
                        max = np.nanmax(E_array, axis=0)[2]

                elif filename == "da":
                    if modus == "da":
                        if man_strain is None:
                            max = np.nanmax(E_array, axis=0)[2]
                        else:
                            max = man_strain
                    else:
                        max = np.nanmax(E_array, axis=0)[2]

                output[outindex].append(
f"""\ndisplay update off
display resetview
variable bar_mol



set old_top [molinfo top]
set bar_mol [mol new]
mol top $bar_mol

#bar can be fixed with mol fix 'molid of the bar' 



# We want to draw relative to the location of the top mol so that the bar 
# will always show up nicely.
set center [molinfo $old_top get center]
set center [regsub -all {{[{{}}]}} $center ""]
set center [split $center]
set min {min}
set max {max}
set length 30.0
set width [expr $length / 6]

# draw the color bar
set start_y [expr 1 + [lindex $center 1] ]

set use_x [expr 1 + [lindex $center 0] ]

set use_z [expr 1+ [lindex $center 2 ]]

set step [expr $length / {N_colors}]

set label_num 8

for {{set colorid 1 }} {{ $colorid <= {N_colors} }} {{incr colorid 1 }} {{
    draw color $colorid
    set cur_y [ expr $start_y + ($colorid -0.5 ) * $step ]
    draw line "$use_x $cur_y $use_z"  "[expr $use_x+$width] $cur_y $use_z" width 10000
}}

# draw the labels
set coord_x [expr (1.1*$width)+$use_x];
set step_size [expr $length / $label_num]
set color_step [expr {N_colors}/$label_num]
set value_step [expr ($max - $min ) / double ($label_num)]

for {{set i 0}} {{$i <= $label_num }} {{ incr i 1}} {{
    set cur_color_id 32
    draw color $cur_color_id
    set coord_y [expr $start_y+$i * $step_size ]
    set cur_text [expr $min + $i * $value_step ]
    draw text  " $coord_x $coord_y $use_z"  [format %6.3f  $cur_text]
}}
draw text " $coord_x [expr $coord_y + $step_size] $use_z"   "{unit}"
# re-set top
mol top $old_top
display update on """)


            #highresolution colorbar with matplotlib
                import matplotlib.pyplot as plt
                from matplotlib.colorbar import ColorbarBase
                from matplotlib.colors import LinearSegmentedColormap, Normalize
                plt.rc('font', size=20)
                fig = plt.figure()
                ax = fig.add_axes([0.05, 0.08, 0.1, 0.9])
                cmap_name = 'my_list'
                cmap = LinearSegmentedColormap.from_list(cmap_name, colorbar_colors, N=N_colors)
                cb = ColorbarBase(ax,
                                  orientation='vertical',
                                  cmap=cmap,
                                  norm=Normalize(min,round(max,3)),
                                  label=unit,
                                  ticks=np.round(np.linspace(min, max, 8), decimals=3))

                fig.savefig(f'{destination_dir / filename}colorbar.pdf', bbox_inches='tight')

            # total strain in the bonds
            proc_geom_RIMs = 100 * (sum(E_array[:, 2]) - self.energies[0]) / self.energies[0]
            warnings.filterwarnings('ignore')
            percent = 100 * E_array[:, 2] / sum(E_array[:, 2])
            jedi_printout_bonds(self.atoms0,
                                self.rim_list[0:2],
                                self.energies[0],
                                sum(E_array[:, 2]),
                                proc_geom_RIMs,
                                percent,
                                E_array[:, 2],
                                ase_units=self.ase_units,
                                file=f'E_{filename}')

        if not man_strain:
            print("\nAdding all energies for the stretch, bending and torsion of the bond with maximum strain...")
            print(f"Maximum energy in bond between atoms "
                  f"{atom_1_max_energy} and {atom_2_max_energy}: {float(max_energy):.3f} {unit}.")

        #write tcl scripts
        for _, mode in enumerate(output):
            if len(mode) > 0:
                f = open(f"{destination_dir / file_list[_]}.vmd", 'w')  # TODO _ variable is actually used, rename?
                f.writelines(mode)

        if self.custom_bonds is not None:
            print(f"\nTotal energy custom bonds: {custom_E} {unit}")


    def partial_analysis(self,indices,ase_units=False):
        '''
        Analyse a substructure with given indices. 

        Args:
            indices: 
                list of indices of atoms in desired substructure
        '''
        #for calculation with partial hessian
        self.ase_units = ase_units
        self.indices = np.arange(0,len(self.atoms0)).tolist()
        self.get_hessian()
        if 3*len(indices)<len(self.H):
            raise ValueError('to little indices for the given hessian')

        cbonds_flag = False
        if self.custom_bonds is not None:
            custom_bonds=self.custom_bonds.copy()
            cbonds_flag = True
            self.custom_bonds=self.custom_bonds[np.isin(self.custom_bonds, indices).all(axis=1)]

        self.rim_list=self.get_common_rims()




        rim_list=self.rim_list
        if len(rim_list)==0:
            raise ValueError('Chosen indexlist has no rims')

        self.B=self.get_b_matrix(indices=self.indices)
        B=self.B
        #set B matrix values of not considered atoms to 0
        for i in range(len(self.H)):
            if i not in indices:
                B[:,i*3:i*3+3]=0
        ind= np.array([[i*3,i*3+1,i*3+2] for i in indices]).ravel()
        B=np.take(self.B, ind,axis=1)

        self.delta_q = self.get_delta_q()
        delta_q = self.delta_q

        H_cart = self.H
        try:
            all_E_geometries = self.get_energies()
        except:
            all_E_geometries = self.energies
        E_geometries = all_E_geometries[0]


        self.proc_E_RIMs,self.E_RIMs,E_RIMs_total,proc_geom_RIMs,self.delta_q=jedi_analysis(self.atomsF,rim_list,B,H_cart,delta_q,E_geometries,ase_units=ase_units)
        #get values of rims inside the substructure
        self.post_process(indices)
        E_RIMs_total=sum(self.E_RIMs)
        proc_geom_RIMs=100*(sum(self.E_RIMs)-E_geometries)/E_geometries
        jedi_printout(self.atoms0,self.rim_list,self.delta_q,E_geometries, E_RIMs_total, proc_geom_RIMs,self.proc_E_RIMs, self.E_RIMs,ase_units=ase_units)

        if cbonds_flag == True:
            self.custom_bonds=custom_bonds #restore the user input



    def post_process(self,indices):             #a function to get segments of all full analysis for better understanding of local strain
        '''
        get only the values of RICs inside a defined substructure

        Args:
            indices: 
                list of indices of atoms in desired substructure
        Returns:
            Values for analyzed RIMs in the defined substructure
        '''
        #get rims with only the considered atoms
        self.indices=indices
        rim_list=self.rim_list
        cbonds_flag = False
        if self.custom_bonds is not None:
            custom_bonds=self.custom_bonds.copy()
            cbonds_flag = True
            self.custom_bonds=self.custom_bonds[np.isin(self.custom_bonds, indices).all(axis=1)]
        rim_p=self.get_common_rims() #get rimlist of substructure

        ind=[]
        rim_list_c=[] #preparing for stacking rim_list to be able to use np.unique

        for i in range(4):   #rim_list is always of length 4
            if rim_list[i].shape == (0,):
                rim_list_c.append([])
            else:
                if rim_p[i].shape[0]>0:
                    rim_list_c.append(np.vstack((rim_list[i],rim_p[i])))
                else:
                    rim_list_c.append(np.vstack((rim_list[i])))
            x,z=np.unique(rim_list_c[-1],return_counts=True,axis=0)

            ind.append(np.where(z>1)[0])                                        #get indices where ric is in both sets
        for i in range(4):
            ind[i]=ind[i]+np.sum([p.shape[0] for p in rim_list[0:i]])      # get correct indices for the stacked array
        ind = np.hstack(ind)
        ind = ind.astype(int)

        self.E_RIMs = np.array(self.E_RIMs)[ind]
        self.delta_q = self.delta_q[ind]
        E_RIMs_total = sum(self.E_RIMs)
        self.proc_E_RIMs = np.array(self.E_RIMs)/E_RIMs_total*100
        if cbonds_flag == True:
            self.custom_bonds=custom_bonds #restore the user input
        pass

    def add_custom_bonds(self, bonds):
        self.custom_bonds = bonds   # additional bonds for analysis of non covalent interactions
        '''Add custom bonds after creating the object.
        
        Args:
            bonds: 
                2Darray
            '''

    def set_bond_params(self,covf=1.3,vdwf=0.9):
        '''
        Args:
            covf: 
                float factor for  covalent radii to determine covalent bonds
            vdwf: 
                float factor for vdw radii to get the upper limit of the custom bond lengths
            '''
        self.covf=covf
        self.vdwf=vdwf

class JediAtoms(Jedi):

    E_atoms=None

    def run(self, ase_units=False,indices=None):
        """Runs the analysis. Calls all necessary functions to get the needed values.

        Args:
            indices: 
                list of indices of a substructure if desired
            ase_units: boolean
                flag to get eV for energies å fo lengths otherwise it is kcal/mol, Bohr  
        Returns:
            Indices, strain, energy in every RIM
        """
        self.ase_units = ase_units
        # get necessary data
        self.indices=np.arange(0,len(self.atoms0))
        if indices:
            self.indices=indices


        self.get_hessian()
        H_cart = self.H         #Hessian of optimized (ground state) structure
        delta_x= self.get_delta_x()

        if len(self.atoms0) != H_cart.shape[0]/3:
            raise ValueError('Hessian has not the fitting shape, possibly a partial hessian. Please try partial_analysis')
        try:
            all_E_geometries = self.get_energies()
        except:
            all_E_geometries = self.energies
        E_geometries = all_E_geometries[0]


    # Get the energy stored in every coordinate (take care to get the right multiplication for a diatomic molecule)
        E_coords = np.sum(0.5*(delta_x*H_cart).T*delta_x,axis=1)
        self.E_atoms=np.sum(E_coords.reshape(-1, 3), axis=1)
        if ase_units==True:

            self.E_atoms*=Hartree
            delta_x*=Bohr
        elif ase_units == False:
            self.E_atoms *= mol/kcal*Hartree
        self.printout(E_geometries)
        pass

    def get_delta_x(self):
        return (self.atomsF.positions.flatten()-self.atoms0.positions.flatten())/Bohr

    def printout(self,E_geometries):
        '''
        Printout of analysis of stored strain energy in the bonds.
        '''
        #############################################
        #	    	   Output section	        	#
        #############################################
        # Header
        output = []
        output.append("\n ************************************************")
        output.append("\n *                 JEDI ANALYSIS                *")
        output.append("\n *       Judgement of Energy DIstribution       *")
        output.append("\n ************************************************\n")

        # Comparison of total energies
        if self.ase_units==False:
            output.append("\n                   Strain Energy (kcal/mol)  Deviation (%)")
        elif self.ase_units==True:
            output.append("\n                   Strain Energy (eV)        Deviation (%)")
        E_atoms_total = sum(self.E_atoms[self.indices])
        output.append("\n      Ab initio     " + "%.8f" % E_geometries + "                  -")
        output.append('\n%5s%16.8f%21.2f' % (" JEDI           ", E_atoms_total, (E_atoms_total / E_geometries-1)*100))


        # strain in the bonds

        if self.ase_units == False:
            output.append("\n Atom No.       Element                              Percentage    Energy (kcal/mol)")
        elif self.ase_units == True:
            output.append("\n Atom No.       Element                              Percentage    Energy (eV)")


        for i, k in enumerate(self.E_atoms[self.indices]):
            output.append('\n%6i%7s%-11s%9.1f%17.7f' % (self.indices[i], " ", self.atoms0.symbols[self.indices[i]], k/E_atoms_total, k))
        print(*output)

    def partial_analysis(self, indices, ase_units=False):

        """Runs the analysis. Calls all necessary functions to get the needed values.

        Args:
            indices: 
                list of indices of a substructure 
            ase_units: boolean
                flag to get eV for energies å fo lengths otherwise it is kcal/mol, Bohr  
        Returns:
            Indices, strain, energy in every RIM
        """
        self.ase_units = ase_units
        # get necessary data
        self.indices=indices


        self.get_hessian()
        H_cart = self.H         #Hessian of optimized (ground state) structure
        #get strain in coordinates
        i = np.repeat(np.atleast_2d(indices),3,axis=0)*3
        i[1]+=1
        i[2]+=2
        i = i.ravel('F')
        delta_x= self.get_delta_x()[i]
#        print(delta_x)

        if len(indices) != H_cart.shape[0]/3:
            raise ValueError('Hessian has not the fitting shape')
        try:
            all_E_geometries = self.get_energies()
        except:
            all_E_geometries = self.energies
        E_geometries = all_E_geometries[0]


    # Get the energy stored in every coordinate (take care to get the right multiplication for a diatomic molecule)
        E_coords = np.sum(0.5*(delta_x*H_cart).T*delta_x,axis=1)
        self.E_atoms=np.sum(E_coords.reshape(-1, 3), axis=1)
        if ase_units==True:

            self.E_atoms*=Hartree
            delta_x*=Bohr
        elif ase_units == False:
            self.E_atoms *= mol/kcal*Hartree
        self.printout(E_geometries)

    def vmd_gen(self,des_colors=None,box=False,man_strain=None,colorbar=True,label='vmd'): #get vmd scripts
        # TODO change os.chdir to take specific folder name as label
        '''Generates vmd scripts and files to save the values for the color coding

        Args:
            des_colors: (dict)
                key: order number, value: [R,G,B]
            box: boolean
                True: draw box
                False: ignore box
            man_strain: float
                reference value for the strain energy used in the color scale
                default: 'None'
            colorbar: boolean
                draw colorbar or not
            label: string
                name of folder for the created files
        '''
        try:
            os.mkdir(label)
        except:
            pass
        os.chdir(label)
        #########################
        #       Basic stuff     #
        #########################

        if self.ase_units == False:
            unit = "kcal/mol"
        elif self.ase_units == True:
            unit = "eV"
        self.atomsF.write('xF.xyz')



        E_atoms = self.E_atoms

        # Write some basic stuff to the tcl scripts

        output = []
        output.append('# Load a molecule\nmol new xF.xyz\n\n')
        output.append('# Change bond radii and various resolution parameters\nmol representation cpk 0.8 0.0 30 '
                      '5\nmol representation bonds 0.2 30\n\n')


        output.append('# Change the color of the graphical representation 0 to white\ncolor change rgb 0 1.00 1.00 '
                      '1.00\n')
        output.append('# The background should be white ("blue" has the colorID 0, which we have changed to '
                      'white)\ncolor Display Background blue\n\n')
        output.append('# Define the other colorIDs\n')


        # Define colorcodes for various atomtypes

        #from .colors import colors
        if des_colors is not None:
            for i in des_colors:
                colors[i] = des_colors[i]         #desired colors overwrite the standard ones

        symbols = np.unique(self.atomsF.get_chemical_symbols())
        symbols = symbols[symbols != 'H']           #get all symbols except H, H is white

        N_colors_atoms = len(symbols)
        N_colors = 32 - N_colors_atoms - 1           #vmd only supports 32 colors for modcolor


        # Generate the color-code and write it to the tcl scripts

        colorbar_colors = []

        #get green to red gradient
        for i in range(N_colors):
            R_value = float(i)/(N_colors/2)
            if R_value > 1:
                R_value = 1
            if N_colors % 2 == 0:
                G_value = 2 - float(i+1)/(N_colors/2)
            if N_colors % 2 != 0:
                G_value = 2 - float(i)/(N_colors/2)
            if G_value > 1:
                G_value = 1

            B_value = 0

            output.append('%1s%5i%10.6f%10.6f%10.6f%1s' % ("color change rgb", i+1, R_value, G_value, B_value, "\n"))
            colorbar_colors.append((R_value, G_value, B_value))

        # add color codes of atoms
        for j in range(N_colors_atoms):
            output.append('%1s%5i%10.6f%10.6f%10.6f%1s' % ("color change rgb", N_colors+j+1, float(colors[symbols[j]][0]), float(colors[symbols[j]][1]), float(colors[symbols[j]][2]), "\n"))

        #add color code for axes and box
        output.append('%1s%5i%10.6f%10.6f%10.6f%1s' % ("color change rgb", 32, float(0), float(0), float(0), "\n"))#black
        output.append('%1s%5i%10.6f%10.6f%10.6f%1s' % ("color change rgb", 1039, float(1), float(0), float(0), "\n"))#red
        output.append('%1s%5i%10.6f%10.6f%10.6f%1s' % ("color change rgb", 1038, float(0), float(1), float(0), "\n"))#green
        output.append('%1s%5i%10.6f%10.6f%10.6f%1s' % ("color change rgb", 1037, float(0), float(0), float(1), "\n"))#blue
        output.append('%1s%5i%10.6f%10.6f%10.6f%1s' % ("color change rgb", 1036, float(0.25), float(0.75), float(0.75), "\n"))#cyan
        output.append('''color Axes X 1039
color Axes Y 1038
color Axes Z 1037
color Axes Origin 1036
color Axes Labels 32
''')
        #define color of atoms with the color code above
        for j in range(N_colors_atoms):
            output.append('\n\nmol representation cpk 0.7 0.0 30 5')
            output.append('\nmol addrep top')
            output.append('\n%s%i%s' % ("mol modstyle ", j+1, " top cpk"))
            output.append('\n%s%i%s%i%s' % ("mol modcolor ", j+1, " top {colorid ", N_colors+j+1, "}"))
            output.append('\n%s%i%s%s%s' % ("mol modselect ", j+1, " top {name ", symbols[j], "}"))





        #########################
        #	Binning		#
        #########################


        # Welcome
        print("\n\nCreating tcl scripts for generating color-coded structures in VMD...")


        # Create an array that stores the atom as the first entry The energy will be added as the secondd entry.
        E_array = np.full((len(self.E_atoms)),np.nan)



        # Create an array that stores only the energies in the coordinate of interest and print some information
        # Get rid of ridiculously small values and treat diatomic molecules explicitly
        # (in order to create a unified picture, we have to create all these arrays in any case)


        if E_atoms.max() <= 0.001:
            E_atoms = np.zeros(len(self.indices))
        E_array[list([*self.indices])]=E_atoms[self.indices] if  len(self.indices) != len(E_atoms) else E_atoms
        E_array=np.vstack((np.arange(len(self.indices)),E_array[self.indices]))
        print("\nProcessing atoms...")

    # Store the maximum energy in a variable for later call

        max_energy = float(np.nanmax(E_array, axis=1)[1])  # maximum energy in one bond

    # Generate the binning windows by splitting bond_E_array into N_colors equal windows


        if man_strain == None:

            binning_windows = np.linspace(0, np.nanmax(E_array, axis=1)[1], num=N_colors )
        else:
            binning_windows = np.linspace(0, float(man_strain), num=N_colors )





        if box  :

            output.append("\n\n# Adding a pbc box")
            output.append('\npbc set {%f %f %f %f %f %f}'%(self.atomsF.cell.cellpar()[0],self.atomsF.cell.cellpar()[1],self.atomsF.cell.cellpar()[2],self.atomsF.cell.cellpar()[3],self.atomsF.cell.cellpar()[4],self.atomsF.cell.cellpar()[5]))
            output.append("\npbc box -color 32")
        output.append("\n\n# Adding a representation with the appropriate colorID for each atom")
            # Calculate which binning_windows value is closest to the bond-percentage and do the output


        for i, b in zip(E_array[0],E_array[1]):
            if np.isnan(b):
                colorID = 32                       #black
            else:
                colorID = np.abs( binning_windows - b ).argmin() + 1

            output.append('\n\nmol representation cpk 0.7 0.0 30 5')
            output.append('\nmol addrep top')
            output.append('\n%s%i%s' % ("mol modstyle ", N_colors_atoms+i+1, " top cpk"))
            output.append('\n%s%i%s%i%s' % ("mol modcolor ", N_colors_atoms+i+1, " top {colorid ", colorID, "}"))
            output.append('\n%s%i%s%s%s' % ("mol modselect ", N_colors_atoms+i+1, " top {index ", int(i), "}\n"))
        f = open('atoms.vmd', 'w')
        f.writelines(output)
        f.close()

        #colorbar
        if colorbar==True:
            min=0.000

            if man_strain == None:
                max = np.nanmax(E_array, axis=1)[1]
            else:
                max = man_strain


            #highresolution colorbar with matplotlib
            import matplotlib.pyplot as plt
            from matplotlib.colorbar import ColorbarBase
            from matplotlib.colors import LinearSegmentedColormap, Normalize
            plt.rc('font', size=20)
            fig = plt.figure()
            ax = fig.add_axes([0.05, 0.08, 0.1, 0.9])
            cmap_name = 'my_list'
            cmap = LinearSegmentedColormap.from_list(cmap_name, colorbar_colors, N=N_colors)
            cb = ColorbarBase(ax, orientation='vertical',
                                        cmap=cmap,
                                        norm=Normalize(min,round(max,3)),
                                        label=unit,
                                        ticks=np.round(np.linspace(min, max, 8),decimals=3))

            fig.savefig('atomscolorbar.pdf', bbox_inches='tight')

        if man_strain==None:
            print("\nAdding all energies for the stretch, bending and torsion of the bond with maximum strain...")
            print(f"Maximum energy in  atom {int(np.argmax(E_atoms)+1)}: {float(max_energy):.3f} {unit}.")


        os.chdir('..')
        pass


