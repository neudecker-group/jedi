import numpy as np
import ase.neighborlist

def get_hbonds(mol):
    '''
    get all hbonds in a structure
    defined as X-H···Y where X and Y can be O, N, F and the angle XHY is larger than 90° and the distance between HY is shorter than 0.9 times the sum of the vdw radii of H and Y
    mol: class
        structure of which the hbonds should be determined
    returns
        2D array of indices
    '''
    cutoff=ase.neighborlist.natural_cutoffs(mol,mult=1.3)   ## cutoff for covalent bonds see Bakken et al.
    bl=np.vstack(ase.neighborlist.neighbor_list('ij',a=mol,cutoff=cutoff)).T   #determine covalent bonds

    bl=bl[bl[:,0]<bl[:,1]]      #remove double mentioned 
    bl = np.unique(bl,axis=0)
    from ase.data.vdw import vdw_radii
    hpartner = ['N','O','F','C']
    hpartner_ls = []
    hcutoff = {('H','N'):0.9*(vdw_radii[1]+vdw_radii[7]),
    ('H','O'):0.9*(vdw_radii[1]+vdw_radii[8]),
    ('H','F'):0.9*(vdw_radii[1]+vdw_radii[9]),
    ('H','C'):0.9*(vdw_radii[1]+vdw_radii[6])}    #save the maximum distances for given pairs to be taken account as interactions
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