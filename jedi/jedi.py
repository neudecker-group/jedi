"""A class for Jedi analysis"""

import collections
import numpy as np
import ase.neighborlist
import ase.geometry
from collections import Counter
from typing import  Any, Dict
from ase.atoms import Atoms
from ase.vibrations import VibrationsData
from ase.atoms import Atom
from ase.utils import jsonable
import os
import ase.io
import scipy
from ase.units import Hartree, Bohr, mol, kcal
def jedi_analysis(atoms,rim_list,B,H_cart,delta_q,E_geometries,printout=None,ase_units=False):
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
    np.savetxt('Hq',H_q)
    # Calculate the total energies in RIMs and its deviation from E_geometries
    E_RIMs_total = 0.5 * np.transpose( delta_q ).dot( H_q ).dot( delta_q )



    # Get the energy stored in every RIM (take care to get the right multiplication for a diatomic molecule)
    E_RIMs = []
    if B.ndim == 1:
        E_current = 0.5 * delta_q[0] * H_q * delta_q[0]
        E_RIMs.append(E_current)
    else:
        for i in range(NRIMs):
            E_current = 0
            for j in range(NRIMs):
                E_current += 0.5 * delta_q[i] * H_q[i,j] * delta_q[j]
            E_RIMs.append(E_current)
    # Get the percentage of the energy stored in every RIM
    proc_E_RIMs = []
    for i in range(NRIMs):
        proc_E_RIMs.append( 100 * E_RIMs[i] / E_RIMs_total )
    if ase_units==True:
        b=np.where(rim_list[:,1]>-1)[0][0]
        delta_q[0:b]*=Bohr
        delta_q[b::]=np.degrees(delta_q[b::])
        E_RIMs=np.array(E_RIMs)*Hartree
        E_RIMs_total*=Hartree
    elif ase_units==False:
        E_RIMs=np.array(E_RIMs)/kcal*mol*Hartree
        E_RIMs_total*=mol/kcal*Hartree

    proc_geom_RIMs = 100 * ( E_RIMs_total - E_geometries ) / E_geometries
 
    if printout:
        jedi_printout(atoms,rim_list,delta_q,E_geometries, E_RIMs_total, proc_geom_RIMs,proc_E_RIMs, E_RIMs,ase_units=ase_units)

    return proc_E_RIMs,E_RIMs, E_RIMs_total, proc_geom_RIMs,delta_q

def jedi_printout(atoms,rim_list,delta_q,E_geometries, E_RIMs_total, proc_geom_RIMs,proc_E_RIMs, E_RIMs,ase_units=False):
    #############################################
    #	    	   Output section	        	#
    #############################################

    # Header
    print("\n \n")
    print(" ************************************************")
    print(" *                 JEDI ANALYSIS                *")
    print(" *       Judgement of Energy DIstribution       *")
    print(" ************************************************\n")
    
    # Comparison of total energies
    if ase_units==False:
        print("                   Strain Energy (kcal/mol)  Deviation (%)")
    elif ase_units==True:
        print("                   Strain Energy (eV)        Deviation (%)")
    print("      Geometries     " + "%.8f" % E_geometries + "                  -" )
    print('%5s%16.8f%21.2f' % (" Red. Int. Modes", E_RIMs_total, proc_geom_RIMs))


    # JEDI analysis
    NRIMs=len(rim_list)
    if ase_units==False:
        print("\n RIM No.       RIM type                       indices        delta_q (au) Percentage    Energy (kcal/mol)")
    elif ase_units==True:
        print("\n RIM No.       RIM type                       indices        delta_q (Å,°) Percentage    Energy (eV)")
    for i in range(NRIMs):
        if rim_list[i][0] == -1 and rim_list[i][1] == -1:
            rim="bond" 
            ind="%s%d %s%d"%(atoms.symbols[rim_list[i][2]],rim_list[i][2],atoms.symbols[rim_list[i][3]],rim_list[i][3])
            print('%6i%7s%-11s%30s%17.7f%9.1f%17.7f' % (i+1, " ", rim, ind,delta_q[i], proc_E_RIMs[i], E_RIMs[i]))
        elif rim_list[i][0] == -1 :
            rim="bond angle"
            ind="%s%d %s%d %s%d"%(atoms.symbols[rim_list[i][1]],rim_list[i][1],atoms.symbols[rim_list[i][2]],rim_list[i][2],atoms.symbols[rim_list[i][3]],rim_list[i][3])
            print('%6i%7s%-11s%30s%17.7f%9.1f%17.7f' % (i+1, " ", rim, ind, delta_q[i], proc_E_RIMs[i], E_RIMs[i]))
        else:
            rim="dihedral"
            ind="%s%d %s%d %s%d %s%d"%(atoms.symbols[rim_list[i][0]],rim_list[i][0],atoms.symbols[rim_list[i][1]],rim_list[i][1],atoms.symbols[rim_list[i][2]],rim_list[i][2],atoms.symbols[rim_list[i][3]],rim_list[i][3])
            print('%6i%7s%-11s%30s%17.7f%9.1f%17.7f' % (i+1, " ", rim,ind, delta_q[i], proc_E_RIMs[i], E_RIMs[i]))


@jsonable('jedi')
class Jedi:
    def __init__(self, atoms0, atomsF, modes): #indices=None
        self.atoms0 = atoms0        #ref state
        self.atomsF = atomsF        #strained state
        self.modes = modes          #VibrationsData object
        self.B = None               #Wilson#s B
        self.delta_q = None         #strain in internal coordinates
        self.rim_list = None        #list of Redundant internal modes
        self.H = None               #cartesian Hessian of ref state
        self.energies = None        #energies of the geometries 
        self.proc_E_RIMs = None     #list of procentual energy stored in single RIMs
        self.part_rim_list=None     #rim list for election of atoms
        self.indices=None           #indices to chose special atoms
        self.E_RIMs=None            #list of energies stored in the rims
        self.hbond=None             #list of hbonds
        self.ase_units=False
    def todict(self) -> Dict[str, Any]:
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
                'hbond': self.hbond}
    @classmethod
    def fromdict(cls, data: Dict[str, Any]) -> 'Jedi':
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
        if data['hbond'] is not None:
            assert isinstance(data['hbond'], (collections.abc.Sequence,
                                                list))
        return cl

    def run(self,indices=None,ase_units=False,hbond=False):
        self.ase_units=ase_units
        # get necessary data
        self.indices=np.arange(0,len(self.atoms0))
        self.get_common_rims(hbond=hbond)
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
            all_E_geometries= self.get_energies()
        except:
            all_E_geometries= self.energies
        E_geometries=all_E_geometries[0]


        #run the analysis
        self.proc_E_RIMs,self.E_RIMs,E_RIMs_total,proc_geom_RIMs,self.delta_q=jedi_analysis(self.atoms0,rim_list,B,H_cart,delta_q,E_geometries,ase_units=ase_units)
        
        if indices:
            
            self.post_process(indices,hbond=hbond)
            E_RIMs_total=sum(self.E_RIMs)
            proc_geom_RIMs=100*(sum(self.E_RIMs)-E_geometries)/E_geometries
            
        jedi_printout(self.atoms0,self.rim_list,self.delta_q,E_geometries, E_RIMs_total, proc_geom_RIMs,self.proc_E_RIMs, self.E_RIMs,ase_units=ase_units)
        pass
        




    def get_rims(self,mol,hbond=False):
        
        ###bondlengths####
        mol = mol

        indices=self.indices
        cutoff=ase.neighborlist.natural_cutoffs(mol,mult=1.3)   ## cutoff for covalent bonds see Bakken et al.
        bl=np.vstack(ase.neighborlist.neighbor_list('ij',a=mol,cutoff=cutoff)).T   #determine covalent bonds

        bl=bl[bl[:,0]<bl[:,1]]      #remove double metioned
        bl, counts = np.unique(bl,return_counts=True,axis=0)
        #if self.indices != None:
          #  bl[np.any([np.in1d(bl[:,3], self.indices),  np.in1d(bl[:,2], self.indices)],axis=0)] take only bl with desired atoms
        if ~ np.all(counts==1):
            print('unit cell too small hessian not calculated for self interaction \
                   jedi analysis for a finite system consisting of the cell will be conducted')
        bl=np.atleast_2d(bl)

        if  len(indices)!=len(mol):
            bl=bl[np.all([np.in1d(bl[:,0], indices),  np.in1d(bl[:,1], indices)],axis=0)]
        nan=np.full((len(bl),2),-1)         #setup array for rims nan used to be able to stack bl with ba and da and to distinguish them
        rim_list=np.hstack((nan,bl))
        
        #hbonds
        if hbond==True:
            from ase.data.vdw import vdw_radii
            hpartner=['N','O','C','F']
            hpartner_ls=[]
            hcutoff={('H','N'):0.9*(vdw_radii[1]+vdw_radii[7]),
                     ('H','C'):0.9*(vdw_radii[1]+vdw_radii[6]),
            ('H','O'):0.9*(vdw_radii[1]+vdw_radii[8]),
            ('H','F'):0.9*(vdw_radii[1]+vdw_radii[9])}
    
            hbond_ls=[]
            for i in range(len(mol)):
                if mol.symbols[i] in hpartner:
                    hpartner_ls.append(i)

            for i in bl:
            
                if mol.symbols[i[0]]=='H' and mol.symbols[i[1]] in hpartner:
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
            if len(hbond_ls)>0:
                self.hbond=hbond_ls
              #  bl=np.vstack((bl,hbond_ls))
                hbond_ls=np.array(hbond_ls)  
                hbond_ls=np.atleast_2d(hbond_ls)     
            
                nan=np.full((len(hbond_ls),2),-1)
                nan=np.hstack((nan,hbond_ls))
                rim_list=np.vstack((rim_list,nan))  
            
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
                            ba=np.array([other_atoms[0], connecting_atom[0], other_atoms[1]])
                            ba_flag=True
                        else:
                            ba=np.vstack((ba, [other_atoms[0], connecting_atom[0], other_atoms[1]])) # add bondlengths to dataframe
                        row_index+=1
        ######sort needed?########## 
        
        if ba_flag==True :      
            ba=np.atleast_2d(ba)     
            ba = ba[ba[:, 1].argsort()]  #sort by atom2
            ba = ba[ba[:, 0].argsort(kind='mergesort')]  # sort by atom1
        
            nan=np.full((len(ba),1),-1)
            nan=np.hstack((nan,ba))
            rim_list=np.vstack((rim_list,nan))
           
            

        ###torsion angles###########


        tb_flag=False
        row_index=0
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
                        if row_index==0:
                            torsionable_bonds=np.array([self_row[0],self_row[1]])
                            tb_flag=True
                        else:
                            torsionable_bonds=np.vstack((torsionable_bonds, [self_row[0],self_row[1]]))
                        bond_partner1 = False
                        bond_partner2 = False
                        row_index+=1              
                        break
        if tb_flag==True:    
            da_flag=False
            torsionable_bonds=np.atleast_2d(torsionable_bonds)
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
                                mol.get_dihedral(int(single_TA_Atom_0),int(torsionable_row[0]),int(torsionable_row[1]),int(single_TA_Atom_3),mic=True)
                                if row_index==0:
                                    da = np.array([single_TA_Atom_0,  torsionable_row[0], torsionable_row[1], single_TA_Atom_3])
                                    da_flag=True
                                    
                                else:
                                    da = np.vstack((da,[single_TA_Atom_0,  torsionable_row[0], torsionable_row[1], single_TA_Atom_3]))
                                  
                                row_index += 1
                            except:
                                continue
                
            if da_flag==True:
                rim_list=np.vstack((rim_list,da))
 
            
        return rim_list
    
    def get_common_rims(self,hbond=False):
        rim_atoms0=self.get_rims(self.atoms0,hbond=hbond)
        rim_atomsF=self.get_rims(self.atomsF,hbond=hbond)
        rim_atoms0v=rim_atoms0.view([('', rim_atoms0.dtype)] * rim_atoms0.shape[1]).ravel()
        rim_atomsFv=rim_atomsF.view([('', rim_atomsF.dtype)] * rim_atomsF.shape[1]).ravel()
        
        rim_l,ind,z=np.intersect1d(rim_atoms0v, rim_atomsFv,return_indices=True)
      
        #if len(self.indices) < len(self.atoms0)  :
        rim_l=rim_l[ind.argsort()]
     
        rim_l=rim_l.view(rim_atoms0.dtype).reshape(-1, rim_atoms0.shape[1])
        self.rim_list=rim_l
        
        return rim_l#np.intersect1d(rim_atoms0v, rim_atomsFv).view(rim_atoms0.dtype).reshape(-1, rim_atoms0.shape[1])
    
    def get_hessian(self):
        hessian = self.modes._hessian2d
        self.H = hessian /(Hartree/Bohr**2)
        return hessian
    
    def get_b_matrix(self,indices=None):
        mol = self.atoms0
        if indices==None:
            indices=np.arange(0,len(mol))
        if  len(self.rim_list) == 0:
            self.get_common_rims()
        rim_list = self.rim_list 
        rim_size=np.size(rim_list,axis=0)
        #NCart_coords = 3*len(mol)
        column = 0
        b = np.zeros([int(len(indices)*3), int(rim_size)], dtype=float)
    
        for q in rim_list:  # for loop for all redunant internal coordinates
            row = 0  # Initilization of columns to specifiy position in B-Matrix
            if len(q) == 0:  
                break
        
            ########  Section for stretches  #########
            if (q[0]) == -1 and (q[1]) == -1:
            
                BL = []
                BL = [int(q[2]), int(q[3])]  # create list of involved atoms
                q_i, q_j = BL
            
                u=mol.get_distance(q_i,q_j,mic=True,vector=True)
                for NAtom in indices:  # for-loop of Number of Atoms (3N)
                    # for-loop of cartesian coordinates of each Atom (x, y and z)
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
                
                


                
        #################ba###############################

            elif (q[0]) == -1:
                BA = []
                row = 0
                
                center_atom = int(q[2])  # define center_atom (redundant step)
                BA = [int(q[1]), int(q[2]), int(q[3])]  # create list of involved atoms
                q_i, q_j, q_k = BA
                u = mol.get_distance(q_i,q_j,mic=True,vector=True)
                v = mol.get_distance(q_k,q_j,mic=True,vector=True)


        
                def get_B_matrix_angles_derivatives(u,v):
                    
                    
                    angle = ase.geometry.get_angles(u,v) # angle between v and u

                    if angle == 180:
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
                        if NAtom == q:  # derivative of redundnat internal coordinate w/ respect to cartesian coordinates is not equal zero
                                        # if redundant internal coordinate (q) contains the Atomnumber (NAtoms) of the cartesian coordinate (x0_coords) from which is derived from.
                            b_j = 0
                            if q == q_j:  # if-Statements for sign-factors
                                b_j =  get_B_matrix_angles_derivatives(np.atleast_2d(u),np.atleast_2d(v))[0][1] # Q-Chem defines BA as m-o-n, where the atomnr. in the middle is the center-atom
                                b[row:row+3, column] = -b_j                                # making it possible to define the center atom as RIM[2]
                            elif q == q_i:
                                b_j = get_B_matrix_angles_derivatives(np.atleast_2d(u),np.atleast_2d(v))[0][0]
                                b[row:row+3, column] = -b_j
                            elif q == q_k:
                                b_j = get_B_matrix_angles_derivatives(np.atleast_2d(u),np.atleast_2d(v))[0][2]
                                b[row:row+3, column] = -b_j
                    row += 3
                column += 1




            elif q[0] > -1:
                
                DA = []
                row = 0
        
                DA = [int(q[0]), int(q[1]), int(q[2]), int(q[3])]  # create list of involved atoms
                q_i, q_j, q_k, q_l = DA

                u = np.copy(np.atleast_2d(mol.get_distance(q_i,q_j,mic=True,vector=True))) #####copy needed because derivative function rewrites vector variable as normed vector
                w = np.copy(np.atleast_2d(mol.get_distance(q_k,q_l,mic=True,vector=True)))
                v = np.copy(np.atleast_2d(mol.get_distance(q_j,q_k,mic=True,vector=True)))

                
                #DA.sort()  # sort list of involved atoms (sorted list necessary for correct stacking of B-Matrix Elements)
                for NAtom in indices:  # for-loop of Number of Atoms (3N)
                    
                    for q in DA:
                        
                        if NAtom == q:  # derivative of redundant internal coordinate w/ respect to cartesian coordinates is not equal zero
                                        # if redundant internal coordinate (q) contains the Atomnumber (NAtoms) of the cartesian coordinate (x0_coords) from which is derived from.
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
        e0 = self.atoms0.calc.get_potential_energy()
        eF = self.atomsF.calc.get_potential_energy()
        if self.ase_units==False:
            e0*=mol/kcal
            eF*=mol/kcal
        deltaE = eF - e0
        self.energies=[deltaE, eF, e0]
        return [deltaE, eF, e0]
        
    def get_delta_q(self):

        indices=self.indices
        try:
            len(self.rim_list)
        except:
            self.get_common_rims()
        rim_list = self.rim_list
        if  len(self.B) == 0:
            self.get_b_matrix()
        B = self.B
        q0 = []
        qF = []
        dq_da = []

        for q_index, q in enumerate(rim_list):  # for loop for all redunant internal coordinates
            
            if len(q) == 0:  # no redundant internal coordinates, prints user-info (see line 75)
                break
            

            if q[0] == -1 and q[1] == -1:
                q0.append(self.atoms0.get_distance(int(q[2]),int(q[3]),mic=True)/Bohr)
                
                qF.append(self.atomsF.get_distance(int(q[2]),int(q[3]),mic=True)/Bohr)

            elif q[0] == -1:
                
                q0.append(np.radians(self.atoms0.get_angle(int(q[1]),int(q[2]),int(q[3]),mic=True)))
                qF.append(np.radians(self.atomsF.get_angle(int(q[1]),int(q[2]),int(q[3]),mic=True)))
            elif  q[0] > -1:
   
                q0_preliminary=np.radians(self.atoms0.get_dihedral(int(q[0]),int(q[1]),int(q[2]),int(q[3]),mic=True))
                qF_preliminary=np.radians(self.atomsF.get_dihedral(int(q[0]),int(q[1]),int(q[2]),int(q[3]),mic=True))
                
                delta_x = self.atomsF.get_positions()-self.atoms0.get_positions()


                delta_x_list = delta_x.flatten().tolist()
                #for partial analysis
                if len(delta_x_list) != len(indices)*3 & len(indices)!=0:
                    indx=np.vstack((np.array(indices),np.array(indices)+1,np.array(indices)+2)).T.ravel()
                    delta_x_list=np.array([delta_x_list[i] for i in indx]) 

#                check_dihedral = np.dot(B, delta_x_list)

                #dq_da.append(np.sign(check_dihedral[q_index])*min((360-abs(qF_preliminary-q0_preliminary)),abs(qF_preliminary-q0_preliminary)))
                q0_final = q0_preliminary
                qF_final = qF_preliminary
                dda=qF_preliminary-q0_preliminary
                if 2*np.pi-abs(dda)<abs(dda):
                    dda=(2*np.pi-abs(dda))*-np.sign(dda)
                dq_da.append(dda)
         
                # if ( check_dihedral[q_index] < 0 ) and ( ( qF_preliminary - q0_preliminary ) > 0 ):
                #     q0_final = -q0_preliminary
                #     qF_final = -qF_preliminary
                # elif ( check_dihedral[q_index] > 0 ) and ( ( qF_preliminary - q0_preliminary ) < 0 ):
                #     q0_final = -q0_preliminary
                #     qF_final = -qF_preliminary

                # q0.append(q0_final)
                # qF.append(qF_final)
        delta_q = np.subtract(qF, q0)
     
        try:
            delta_q=np.append(delta_q,dq_da)
        except:
            pass
        #delta_q=np.abs(delta_q)
        self.delta_q = delta_q
        
        return delta_q

    def vmd_gen(self,des_colors=None,box=False,man_strain=None,colorbar=True):
        #########################
        #       Basic stuff     #
        #########################
        if self.ase_units==False:
            unit="kcal/mol"
        elif self.ase_units==True:
            unit="eV"


        rim_list = self.rim_list
        self.atomsF.write('xF.xyz')
        if  len(self.proc_E_RIMs) == 0:
            self.run()
        proc_E_RIMs = self.proc_E_RIMs
        pbc_flag=True
        if box==True and self.atomsF.get_pbc().any()==True:
            pbc_flag=True
        # Check whether we need to write vmd_ba.tcl, vmd_da.tcl and vmd_all.tcl and read basic stuff
        file_list = []
        bl = []
        ba = []
        da = []
        ba_flag=False
        da_flag=False
        for i in rim_list:
        # Bond lengths (a molecule has at least one bond):

            if i[0] == -1 and (i[1]) == -1:

                
                numbers = [int(i[2]),int(i[3])]
                bl.append(numbers)
                if 'vmd_bl.tcl' not in file_list:
                    open('vmd_bl.tcl', 'w').close()
                    file_list.append('vmd_bl.tcl')
                
                

            # Bond angles:

            elif i[0] == -1 :
                ba_flag = True
                


                numbers = [int(i[1]),int(i[2]),int(i[3])]
                ba.append(numbers)
                if 'vmd_ba.tcl' not in file_list:
                    open('vmd_ba.tcl', 'w').close()
                    file_list.append('vmd_ba.tcl')
                # All (for this, at least bond angles have to be present):
                    open('vmd_all.tcl', 'w').close()
                    file_list.append('vmd_all.tcl')
                
            # Dihedral angles:

            elif i[0] != -1 :
                da_flag = True
                
        

                numbers = [int(n) for n in i]
                da.append(numbers)
                if 'vmd_da.tcl' not in file_list:
                    open('vmd_da.tcl', 'w').close()
                    file_list.append('vmd_da.tcl')


        # E_RIMs_perc

        E_RIMs_perc = np.array(proc_E_RIMs)

        E_RIMs = self.E_RIMs
        # Write some basic stuff to the tcl scripts
        for filename in file_list:
            if filename == "vmd_bl.tcl" or filename == "vmd_ba.tcl" or filename == "vmd_da.tcl" or filename == "vmd_all.tcl":
                f = open(filename, 'w')
                f.write('# Load a molecule\nmol new xF.xyz\n\n')
                f.write('# Change bond radii and various resolution parameters\nmol representation cpk 0.8 0.0 30 5\nmol representation bonds 0.2 30\n\n')
                f.write('# Change the drawing method of the first graphical representation to CPK\nmol modstyle 0 top cpk\n')
                f.write('# Color only H atoms white\nmol modselect 0 top {name H}\n')
                f.write('# Change the color of the graphical representation 0 to white\ncolor change rgb 0 1.00 1.00 1.00\nmol modcolor 0 top {colorid 0}\n')
                f.write('# The background should be white ("blue" has the colorID 0, which we have changed to white)\ncolor Display Background blue\n\n')
                f.write('# Define the other colorIDs\n')
                f.close()

        # Define colorcodes for various atomtypes

            # colors = {'C': [0.5, 0.5, 0.5],        
            #       'N': [0.0, 0.0, 1.0],
            #       'O': [1.0, 0.0, 0.0],
            #       'S': [1.0, 1.0, 0.0]}
        from ase.jedi.colors import colors 
        if des_colors!=None:
            for i in des_colors:
                colors[i]=des_colors[i]
            
        symbols=np.unique(self.atomsF.get_chemical_symbols())
        symbols=symbols[symbols!='H']
        
        N_colors_atoms=len(symbols)
        N_colors = 32 -N_colors_atoms-1           #vmd only supports 32 colors for modcolor



  




        # Generate the color-code and write it to the tcl scripts
        for filename in file_list:
            if filename == "vmd_bl.tcl" or filename == "vmd_ba.tcl" or filename == "vmd_da.tcl" or filename == "vmd_all.tcl":
                f = open(filename, 'a')

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

                    f.write('%1s%5i%10.6f%10.6f%10.6f%1s' % ("color change rgb", i+1, R_value, G_value, B_value, "\n"))

                # add color codes of "standard" atoms
                for j in range(N_colors_atoms):
                    f.write('%1s%5i%10.6f%10.6f%10.6f%1s' % ("color change rgb", N_colors+j+1, float(colors[symbols[j]][0]), float(colors[symbols[j]][1]), float(colors[symbols[j]][2]), "\n"))
                f.write('%1s%5i%10.6f%10.6f%10.6f%1s' % ("color change rgb", 32, float(0), float(0), float(0), "\n"))#black
                f.write('%1s%5i%10.6f%10.6f%10.6f%1s' % ("color change rgb", 1039, float(1), float(0), float(0), "\n"))#red
                f.write('%1s%5i%10.6f%10.6f%10.6f%1s' % ("color change rgb", 1038, float(0), float(1), float(0), "\n"))#green
                f.write('%1s%5i%10.6f%10.6f%10.6f%1s' % ("color change rgb", 1037, float(0), float(0), float(1), "\n"))#blue
                f.write('%1s%5i%10.6f%10.6f%10.6f%1s' % ("color change rgb", 1036, float(0.25), float(0.75), float(0.75), "\n"))#cyan
                f.write('''color Axes X 1039
color Axes Y 1038
color Axes Z 1037
color Axes Origin 1036
color Axes Labels 32
''')
                for j in range(N_colors_atoms):
                    f.write('\n\nmol representation cpk 0.7 0.0 30 5')
                    f.write('\nmol addrep top') 	
                    f.write('\n%s%i%s' % ("mol modstyle ", j+1, " top cpk"))
                    f.write('\n%s%i%s%i%s' % ("mol modcolor ", j+1, " top {colorid ", N_colors+j+1, "}"))
                    f.write('\n%s%i%s%s%s' % ("mol modselect ", j+1, " top {name ", symbols[j], "}"))
                
            
                f.close()


        #########################
        #	Binning		#
        #########################
      
        
        # Welcome
        print("\n\nCreating tcl scripts for generating color-coded structures in VMD...")

        # Achieve the binning for bl, ba, da an all simultaneously
        sum_energy = 0  # variable to add up all energies in the molecule
        for filename in file_list:
            if filename == "vmd_bl.tcl" or filename == "vmd_ba.tcl" or filename == "vmd_da.tcl" or filename == "vmd_all.tcl":

        # Create an array that stores the bond connectivity as the first two entries. The energy will be added as the third entry.
                bond_E_array = np.full((len(bl),3),np.nan)
                for i in range(len(bl)):
                    bond_E_array[i][0] = bl[i][0]
                    bond_E_array[i][1] = bl[i][1]

        # Create an array that stores only the energies in the coordinate of interest and print some information
        # Get rid of ridiculously small values and treat diatomic molecules explicitly
        # (in order to create a unified picture, we have to create all these arrays in any case)
        # Bonds
                if filename == "vmd_bl.tcl" or filename == "vmd_all.tcl":
                    if len(bl) == 1:
                        E_bl_perc = E_RIMs_perc[0]
                        E_bl = E_RIMs
                    else:
                        E_bl_perc = E_RIMs_perc[0:len(bl)]
                        E_bl = E_RIMs[0:len(bl)]
                        if E_bl_perc.max() <= 0.001:
                            E_bl_perc = np.zeros(len(bl))
                    if filename == "vmd_bl.tcl":
                        print("\nProcessing bond lengths...")
                        print("%s%6.2f%s" % ("Maximum energy in a bond length:      ", E_bl_perc.max(), '%'))
                        print("%s%6.2f%s" % ("Total energy in the bond lengths:     ", E_bl_perc.sum(),'%'))

        # Bendings
                if (filename == "vmd_ba.tcl" and ba_flag == True) or (filename == "vmd_all.tcl" and ba_flag == True):
                    E_ba_perc = E_RIMs_perc[len(bl):len(bl)+len(ba)]
                    E_ba = E_RIMs[len(bl):len(bl)+len(ba)]
                    if E_ba_perc.max() <= 0.001:
                        E_ba_perc = np.zeros(len(ba))
                    if filename == "vmd_ba.tcl":
                        print("\nProcessing bond angles...")
                        print("%s%6.2f%s" % ("Maximum energy in a bond angle:       ", E_ba_perc.max(), '%'))
                        print("%s%6.2f%s" % ("Total energy in the bond angles:      ", E_ba_perc.sum(),'%'))

        # Torsions (handle stdout separately)
                if (filename == "vmd_da.tcl" and da_flag == True ) or (filename == "vmd_all.tcl" and da_flag == True):
                    E_da_perc = E_RIMs_perc[len(bl)+len(ba):len(bl)+len(ba)+len(da)]
                    E_da = E_RIMs[len(bl)+len(ba):len(bl)+len(ba)+len(da)]
                    if E_da_perc.max() <= 0.001:
                        E_da_perc = np.zeros(len(da))
                if filename == "vmd_da.tcl" and da_flag == True:
                    print("\nProcessing dihedral angles...")
                    print("%s%6.2f%s" % ("Maximum energy in a dihedral angle:   ", E_da_perc.max(), '%'))
                    print("%s%6.2f%s" % ("Total energy in the dihedral angles:  ", E_da_perc.sum(),'%'))

        # Map onto the bonds (create "all" on the fly and treat diatomic molecules explicitly)
        # Bonds (trivial)
                if filename == "vmd_bl.tcl" or filename == "vmd_all.tcl":
                    for i in range(len(bl)):
                        if len(bl) == 1:
                            bond_E_array[i][2] = E_bl[i]
                        else:
                            bond_E_array[i][2] = E_bl[i]
                
                            
        # Bendings
                if (filename == "vmd_ba.tcl" and ba_flag == True) or (filename == "vmd_all.tcl" and ba_flag == True):
                    for i in range(len(ba)):
                        for j in range(len(bl)):
                            if ((ba[i][0] == bl[j][0] and ba[i][1] == bl[j][1]) or  # look for the right connectivity
                                (ba[i][0] == bl[j][1] and ba[i][1] == bl[j][0]) or
                                (ba[i][1] == bl[j][0] and ba[i][2] == bl[j][1]) or
                                (ba[i][1] == bl[j][1] and ba[i][2] == bl[j][0])):
                                bond_E_array[j][2] += 0.5 * E_ba[i]
                                if np.isnan(bond_E_array[j][2] ):
                                    bond_E_array[j][2] = 0.5 * E_ba[i]
                                

        # Torsions
                if (filename == "vmd_da.tcl" and da_flag == True) or ( filename == "vmd_all.tcl" and da_flag == True ):
                    for i in range(len(da)):
                        for j in range(len(bl)):
                            if ((da[i][0] == bl[j][0] and da[i][1] == bl[j][1]) or 
                                (da[i][0] == bl[j][1] and da[i][1] == bl[j][0]) or
                                (da[i][1] == bl[j][0] and da[i][2] == bl[j][1]) or
                                (da[i][1] == bl[j][1] and da[i][2] == bl[j][0]) or
                                (da[i][2] == bl[j][0] and da[i][3] == bl[j][1]) or
                                (da[i][2] == bl[j][1] and da[i][3] == bl[j][0])):
                                bond_E_array[j][2] += (float(1)/3) * E_da[i]
                                if np.isnan(bond_E_array[j][2] ):
                                    bond_E_array[j][2] = (float(1)/3) * E_da[i]
                        
                if (filename == "vmd_all.tcl" and self.hbond != None):
                    
                    hbond_E=sum(bond_E_array[:,2][len(bl)-len(self.hbond):len(bl)])
                    
                translate={}                        # the new indices need to get the same values as the original ones inside the cell
                for i in range(len(bond_E_array)):
                    translate[(np.min([bond_E_array[i,0],bond_E_array[i,1]]),np.max([bond_E_array[i,0],bond_E_array[i,1]]))]=bond_E_array[i,2]                
                
                if len(self.indices)<len(self.atomsF):
                    p_rim=self.rim_list.copy()
                    p_indices=self.indices
                    self.indices=range(len(self.atomsF))
                    rim=self.get_common_rims(hbond=self.hbond!=None)
                    
                    b=np.where(rim[:,1]>-1)[0][0]
                    
                    rim=rim[:,[2,3]]
                    rim=np.ascontiguousarray(rim)
                    a=np.array(bl).view([('', np.array(bl).dtype)] * np.array(bl).shape[1]).ravel()
                
                    
                    
                    b=np.array(rim).view([('', np.array(bl).dtype)] * np.array(bl).shape[1]).ravel()
                    
                    
                    rim=np.setxor1d(a, b) 
                    #print(rim)       
                    rim=rim.view(np.array(bl).dtype).reshape(-1, 2)
                 
                    nan=np.full((len(rim),1),np.nan)         #nan for special color (black)
                    rim=np.hstack((rim,nan))              #stack for later vmd visualization
                    bond_E_array=np.vstack((bond_E_array,rim))
                    self.rim_list=p_rim
                    self.indices=p_indices
        # get bonds that reach out of the unit cell
                if pbc_flag==True:
                    from ase.data.vdw import vdw_radii
                    cutoff=[ vdw_radii[atom.number] * 0.9 for atom in self.atomsF]
                    #cutoff=ase.neighborlist.natural_cutoffs(self.atomsF,mult=1.3)   ## cutoff for covalent bonds see Bakken et al.
                    ex_bl=np.vstack(ase.neighborlist.neighbor_list('ij',a=self.atomsF,cutoff=cutoff)).T 
                    #print(ase.geometry.get_distances(self.atomsF.positions,self.atomsF.positions,cell=self.atomsF.cell,pbc=self.atomsF.pbc)[1].shape)
                    ex_bl=np.hstack((ex_bl,ase.neighborlist.neighbor_list('S',a=self.atomsF,cutoff=cutoff)))
                    ex_bl=np.hstack((ex_bl,ase.neighborlist.neighbor_list('D',a=self.atomsF,cutoff=cutoff)))

                    bond_ex_cell=ex_bl[(ex_bl[:,2]!=0) | (ex_bl[:,3]!=0) |(ex_bl[:,4]!=0)]          #determines which nearest neighbors are outside the unit cell
                    atoms_ex_cell=bond_ex_cell                    #[np.unique(bond_ex_cell[:,1:5],return_index=True,axis=0)[1]]  # neglects double mentioned
                    mol=self.atomsF.copy()                   # a extended cell is needed for vmd since it does not show intercellular bonds
                    mol.wrap()                              #wrap molecule important for atoms close to the boundaries
                    #ex_indx=np.array([])                # auxillary indices are used
                    bond_E_array_app=np.zeros([len(atoms_ex_cell),3])  # bond list has to be appended
                    translate={}                        # the new indices need to get the same values as the original ones inside the cell
                    for i in range(len(bond_E_array)):
                        translate[(np.min([bond_E_array[i,0],bond_E_array[i,1]]),np.max([bond_E_array[i,0],bond_E_array[i,1]]))]=bond_E_array[i,2]
                    
                    for i in range(len(atoms_ex_cell)):
                        pos_ex_atom=mol.get_positions()[int(atoms_ex_cell[i,0])]+atoms_ex_cell[i,5:8]       # get positions of cell external atoms by adding the vector
                        #ex_indx=np.append(ex_indx,atoms_ex_cell[i,1])            
                        original_rim=[int(atoms_ex_cell[i,0]),int(atoms_ex_cell[i,1])]                      # get the indices of the corresponding atoms inside the cell
                        original_rim.sort()                                                                 # needs to be sorted because rim list only covers one direction                           
                        try:
                            bond_E_array_app[i,0:3]=[atoms_ex_cell[i,0],len(mol),translate[tuple(original_rim)]]                          # add to bond list with auxillary index
                            mol.append(Atom(symbol=mol.symbols[int(atoms_ex_cell[i,1])],position=pos_ex_atom))  # append to the virtual atoms object
                        
                        except:
                            #bond_E_array_app[i,0:3]=[atoms_ex_cell[i,0],len(self.atomsF)+i,np.nan]  #if partial analysis not analyzed bonds get a different color
                            pass    
                    bond_E_array_app=bond_E_array_app[~np.all(bond_E_array_app == 0, axis=1)]

                    bond_E_array=np.vstack((bond_E_array,bond_E_array_app))
                    
                    
                    mol.write('xF.xyz')
   
########################## get left out bonds ########################

                # if len(self.indices)< len(self.atomsF):
                #     cutoff=ase.neighborlist.natural_cutoffs(self.atomsF,mult=1.3)   ## cutoff for covalent bonds see Bakken et al.
                #     add_bl=np.vstack(ase.neighborlist.neighbor_list('ij',a=self.atomsF,cutoff=cutoff)).T 
                #     add_bl=add_bl[add_bl[:,0]<add_bl[:,1]]  #get rid of doubles
                #     leftouts=np.arange(0,len(self.atomsF))   # get index of left outs
                #     leftouts=np.delete(leftouts,self.indices) #by deleting the chosen indices
                #     add_bl=add_bl[np.any([np.in1d(add_bl[:,0], leftouts),  np.in1d(add_bl[:,1], leftouts)],axis=0)]   # get the left out bonds
                #     nan=np.full((len(add_bl),1),np.nan)         #nan for special color (black)
                #     add_bl=np.hstack((add_bl,nan))              #stack for later vmd visualization
                #     bond_E_array=np.vstack((bond_E_array,add_bl))

                    # if self.hbond!=None:
                    #     from ase.data.vdw import vdw_radii
                    #     hpartner=['N','O','C','F']
                    #     hpartner_ls=[]
                    #     hcutoff={('H','N'):0.9*(vdw_radii[1]+vdw_radii[7]),
                    #             ('H','C'):0.9*(vdw_radii[1]+vdw_radii[6]),
                    #     ('H','O'):0.9*(vdw_radii[1]+vdw_radii[8]),
                    #     ('H','F'):0.9*(vdw_radii[1]+vdw_radii[9])}
                
                    #     hbond_ls=[]
                    #     for i in range(len(mol)):
                    #         if mol.symbols[i] in hpartner:
                    #             hpartner_ls.append(i)
                        
                    #     for i in bond_E_array:
                        
                    #         if mol.symbols[int(i[0])]=='H' and mol.symbols[int(i[1])] in hpartner:
                    #             for j in hpartner_ls:  
                    #                 if j != int(i[1]):                   
                    #                     if mol.get_distance(int(i[0]),int(j),mic=True)<  hcutoff[(mol.symbols[int(i[0])], mol.symbols[int(j)])] \
                    #                         and mol.get_angle(int(i[1]),int(i[0]),int(j),mic=True)>90:
                                            
                    #                         hbond_ls.append([int(i[0]), int(j)])
                                
                    #         elif mol.symbols[int(i[0])] in hpartner and mol.symbols[int(i[1])]=='H':
                    #             for j in hpartner_ls:   
                    #                 if j != int(i[0]):       
                                    
                    #                     if mol.get_distance(int(i[1]),int(j),mic=True) < hcutoff[(mol.symbols[int(i[1])], mol.symbols[int(j)])] and mol.get_angle(int(i[0]),int(i[1]),int(j),mic=True) >90:
                                    
                    #                         hbond_ls.append([int(i[1]), int(j)])
                    #     if len(hbond_ls)>0:
                    #         hbond_ls=np.array(hbond_ls)
                    #         a=hbond_ls.view([('', hbond_ls.dtype)] * hbond_ls.shape[1]).ravel()
                    #         b=np.array(self.hbond).view([('', np.array(self.hbond).dtype)] * np.array(self.hbond).shape[1]).ravel()
                    #         hbond_ls=np.setxor1d(a, b)
                            
                    #         hbond_ls=hbond_ls.view(np.array(self.hbond).dtype).reshape(-1, 2)
                    
                    #     #  bl=np.vstack((bl,hbond_ls))
                    #         hbond_ls=np.array(hbond_ls)  
                    #         hbond_ls=np.atleast_2d(hbond_ls)     
                        
                    #     nan=np.full((len(hbond_ls),1),np.nan)         #nan for special color (black)
                    #     hbond_ls=np.hstack((hbond_ls,nan))              #stack for later vmd visualization
                    #     bond_E_array=np.vstack((bond_E_array,hbond_ls))
                           

                    # if pbc_flag==True:
                    #     from ase.data.vdw import vdw_radii
                    #     cutoff=[ vdw_radii[atom.number] * 0.9 for atom in self.atomsF]
                    #     #cutoff=ase.neighborlist.natural_cutoffs(self.atomsF,mult=1.3)   ## cutoff for covalent bonds see Bakken et al.
                    #     ex_bl=np.vstack(ase.neighborlist.neighbor_list('ij',a=self.atomsF,cutoff=cutoff)).T 
                    #     #print(ase.geometry.get_distances(self.atomsF.positions,self.atomsF.positions,cell=self.atomsF.cell,pbc=self.atomsF.pbc)[1].shape)
                    #     ex_bl=np.hstack((ex_bl,ase.neighborlist.neighbor_list('S',a=self.atomsF,cutoff=cutoff)))
                    #     ex_bl=np.hstack((ex_bl,ase.neighborlist.neighbor_list('D',a=self.atomsF,cutoff=cutoff)))

                    #     bond_ex_cell=ex_bl[(ex_bl[:,2]!=0) | (ex_bl[:,3]!=0) |(ex_bl[:,4]!=0)]          #determines which nearest neighbors are outside the unit cell
                    #     atoms_ex_cell=bond_ex_cell                    #[np.unique(bond_ex_cell[:,1:5],return_index=True,axis=0)[1]]  # neglects double mentioned
                    #     mol=self.atomsF.copy()                   # a extended cell is needed for vmd since it does not show intercellular bonds
                    #     mol.wrap()                              #wrap molecule important for atoms close to the boundaries
                    #     #ex_indx=np.array([])                # auxillary indices are used
                    #     bond_E_array_app=np.zeros([len(atoms_ex_cell),3])  # bond list has to be appended
                    #     translate={}                        # the new indices need to get the same values as the original ones inside the cell
                    #     for i in range(len(bond_E_array)):
                    #         translate[(np.min([bond_E_array[i,0],bond_E_array[i,1]]),np.max([bond_E_array[i,0],bond_E_array[i,1]]))]=bond_E_array[i,2]
                        
                    #     for i in range(len(atoms_ex_cell)):
                    #         pos_ex_atom=mol.get_positions()[int(atoms_ex_cell[i,0])]+atoms_ex_cell[i,5:8]       # get positions of cell external atoms by adding the vector
                    #         #ex_indx=np.append(ex_indx,atoms_ex_cell[i,1])            
                    #         original_rim=[int(atoms_ex_cell[i,0]),int(atoms_ex_cell[i,1])]                      # get the indices of the corresponding atoms inside the cell
                    #         original_rim.sort()                                                                 # needs to be sorted because rim list only covers one direction                           
                    #         try:
                    #             bond_E_array_app[i,0:3]=[atoms_ex_cell[i,0],len(mol),translate[tuple(original_rim)]]                          # add to bond list with auxillary index
                    #             mol.append(Atom(symbol=mol.symbols[int(atoms_ex_cell[i,1])],position=pos_ex_atom))  # append to the virtual atoms object
                            
                    #         except:
                    #             #bond_E_array_app[i,0:3]=[atoms_ex_cell[i,0],len(self.atomsF)+i,np.nan]  #if partial analysis not analyzed bonds get a different color
                    #             pass      
                    #     bond_E_array=np.vstack((bond_E_array,bond_E_array_app))
                    
                     #   mol.write('xF.xyz')

             
        # Store the maximum energy in a variable for later call
                # if filename == "vmd_all.tcl":
                #     if not modus == "all":  # only do this, when the user didn't call the --v flag 
                #         max_energy = float(np.nanmax(bond_E_array, axis=0)[2])  # maximum energy in one bond
                #         for row in bond_E_array: 
                          
                #             if max_energy in row:
                #                 atom_1_max_energy = int(row[0])
                #                 atom_2_max_energy = int(row[1])

        # Generate the binning windows by splitting bond_E_array into N_colors equal windows
            if filename == "vmd_all.tcl":
                if man_strain != None:
                    binning_windows = np.linspace( 0, float(man_strain), num=N_colors )
                else: 
                    binning_windows = np.linspace( 0, np.nanmax(bond_E_array, axis=0)[2], num=N_colors )
                
            elif filename == "vmd_bl.tcl":
                if man_strain != None:
                    binning_windows = np.linspace( 0, float(man_strain), num=N_colors )
                else: 
                    binning_windows = np.linspace( 0, np.nanmax(bond_E_array, axis=0)[2], num=N_colors )
                
            elif filename == "vmd_ba.tcl":
                if man_strain != None:
                    binning_windows = np.linspace( 0, float(man_strain), num=N_colors )
                else: 
                    binning_windows = np.linspace( 0, np.nanmax(bond_E_array, axis=0)[2], num=N_colors )
                
            elif filename == "vmd_da.tcl":
                if man_strain != None:
                    binning_windows = np.linspace( 0, float(man_strain), num=N_colors )
                
                else: 
                    binning_windows = np.linspace( 0, np.nanmax(bond_E_array, axis=0)[2], num=N_colors )
            
    
            f = open(filename, 'a')
            if pbc_flag==True:
             
                f.write("\n\n# Adding a pbc box")
                f.write('\npbc set {%f %f %f %f %f %f}'%(self.atomsF.cell.cellpar()[0],self.atomsF.cell.cellpar()[1],self.atomsF.cell.cellpar()[2],self.atomsF.cell.cellpar()[3],self.atomsF.cell.cellpar()[4],self.atomsF.cell.cellpar()[5]))
                f.write("\npbc box -color 32")
            f.write("\n\n# Adding a representation with the appropriate colorID for each bond")
                # Calculate which binning_windows value is closest to the bond-percentage and do the output
            if self.hbond != None:
                lih=len(self.hbond)#-len(self.hbond)
                lim=len(bond_E_array)
               
            else:
                lim=len(bond_E_array)
                lih=len(bl)-len(bond_E_array)
            for i in range(lim):
                if np.isnan(bond_E_array[i][2]):
                    colorID = 32                       #black
                else:
                    colorID = np.abs( binning_windows - bond_E_array[i][2] ).argmin() + 1
                f.write('\nmol addrep top')
                f.write('\n%s%i%s' % ("mol modstyle ", N_colors_atoms+i+1, " top bonds"))
                f.write('\n%s%i%s%i%s' % ("mol modcolor ", N_colors_atoms+i+1, " top {colorid ", colorID, "}"))
                f.write('\n%s%i%s%i%s%i%s' % ("mol modselect ", N_colors_atoms+i+1, " top {index ", bond_E_array[i][0], " ", bond_E_array[i][1], "}\n"))
            for i in range(len(bl)-lih,len(bond_E_array)):
              
                if np.isnan(bond_E_array[i][2]):
                    colorID = 32                       #black
                else:
                    colorID = np.abs( binning_windows - bond_E_array[i][2] ).argmin() + 1
                   
                f.write('\nset x [[atomselect top "index %d %d"] get {x y z}]'%(bond_E_array[i][0],bond_E_array[i][1]))
                f.write('\nset a [lindex $x 0] ')
                f.write('\nset b [lindex $x 1] ')
                f.write('\ndraw  color %d'%(colorID))
                f.write('\ndraw line  $a $b width 3 style dashed' )

            f.close()

        #colorbar
            if colorbar==True:
                min=0
            
                if man_strain==None:
                    max=np.nanmax(bond_E_array, axis=0)[2]
                else:
                    max=man_strain
                f = open(filename, 'a')
                f.write(f'''\n		display update off
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
display update on ''')
                f.close()

        if not man_strain:
            print("\nAdding all energies for the stretch, bending and torsion of the bond with maximum strain...")
            print(f"Maximum energy in bond between atoms {atom_1_max_energy} and {atom_2_max_energy}: {float(max_energy):.3f} {unit}.")
           
        if self.hbond:
            print(f"\nTotal energy in hbonds: {hbond_E} {unit}")
        pass


    def partial_analysis(self,indices,ase_units=False,hbond=False):   
        #for calculation with partial hessian
        self.ase_units=ase_units
        self.indices=np.arange(0,len(self.atoms0)).tolist()
        self.get_hessian()
        if 3*len(indices)<len(self.H):
            raise ValueError('to little indices for the given hessian')
        
        #make the hessian for the complete system by filling zeros
        H=np.zeros((3*len(self.atoms0),3*len(self.atoms0)))
        for i in range(len(indices)):
            for j in range(len(indices)):
                H[indices[i]*3:indices[i]*3+3,indices[j]*3:indices[j]*3+3]=self.H[i*3:i*3+3,j*3:j*3+3]
        self.H=H
        

        self.rim_list=self.get_common_rims(hbond=hbond)
      
        rim_list=self.rim_list
        if len(rim_list)==0:
            raise ValueError('Chosen indexlist has no rims')

        self.B=self.get_b_matrix(indices=self.indices)
        B=self.B
        #set B matrix values of not considered atoms to 0
        for i in range(len(self.H)):
            if i not in indices:
                B[:,i*3:i*3+3]=0

        self.delta_q=self.get_delta_q()
        delta_q = self.delta_q
                    
        H_cart = self.H 
        try:
            all_E_geometries= self.get_energies()
        except:
            all_E_geometries= self.energies
        E_geometries=all_E_geometries[0]

        
        self.proc_E_RIMs,self.E_RIMs,E_RIMs_total,proc_geom_RIMs,self.delta_q=jedi_analysis(self.atoms,rim_list,B,H_cart,delta_q,E_geometries,ase_units=ase_units)
        self.post_process(indices,hbond=hbond)
        E_RIMs_total=sum(self.E_RIMs)
        proc_geom_RIMs=100*(sum(self.E_RIMs)-E_geometries)/E_geometries
        jedi_printout(self.rim_list,self.delta_q,E_geometries, E_RIMs_total, proc_geom_RIMs,self.proc_E_RIMs, self.E_RIMs,ase_units=ase_units)
          
    

            
      
    def post_process(self,indices,hbond=False):             #a function to get segments of all full analysis for better understanding of local strain
        #get rims with only the considered atoms
        self.indices=indices
        rim_list=self.rim_list
        rim_l=self.get_common_rims(hbond=hbond)
        a=np.vstack((rim_list,rim_l ))
        x,indx,z=np.unique(a,return_counts=True,axis=0,return_index=True)
     
        z=z[indx.argsort()]
      

        ind=np.where(z>1)[0]
   
        self.E_RIMs=np.array(self.E_RIMs)[ind]
        self.delta_q=self.delta_q[ind]
        E_RIMs_total=sum(self.E_RIMs)
        self.proc_E_RIMs=np.array(self.E_RIMs)/E_RIMs_total*100
        pass


