================
Further Analysis
================

Due to the larger outputs of the following examples, the outputs are attached as files and only the visualizations are shown. 
The calculations were performed with low accuracy to keep the computational effort as low as possible so that the first time user can perform them to getting used to Jedi. Nevertheless, the Vasp calculations might still take very long.
For better results, a higher accuracy is needed.

Custom bonds
============

Custom bonds can be analyzed too. The Jedi package has a function to determine hydrogen bonds in a structure. These will be added to the RIC with add_custom_bonds() after generating the Jedi object. 

Stretching Hydrogen Bonds Using COGEF 
--------------------------------------

Here, cytosine and guanine are examined.

.. code-block:: python

    from ase import Atoms
    from ase.calculators.gaussian import Gaussian, GaussianOptimizer
    from ase.build import molecule
    import ase.io
    from jedi.jedi import Jedi, get_hbonds
    from ase.vibrations.vibrations import VibrationsData
    
    mol = ase.io.read('cg.xyz')

    calc = Gaussian(mem='6GB',
                        label='opt',
                        method='blyp',
                        basis='4-21G',
                        EmpiricalDispersion='GD3BJ',
                        scf='qc')
    opt=GaussianOptimizer(mol,calc)
    opt.run(fmax='tight', steps=100)
    ase.io.write('opt.json',mol)

    mol=ase.io.read('opt.json')
    mol.calc = Gaussian(mem='6GB',
                        iop='7/33=1',
                        freq='',
                        label='freq',
                        chk='freq.chk',
                        save=None,
                        method='blyp',
                        basis='4-21G',
                        EmpiricalDispersion='GD3BJ',
                        scf='qc')
    mol.get_potential_energy()
    modes = get_vibrations('freq',mol)
    modes.write('modes.json')

    mol2 = mol.copy()
    v = mol.get_distance(10,27)+0.1
    w = mol.get_distance(13,21)+0.1
    x = mol.get_distance(15,20)+0.1

    calc = Gaussian(mem='6GB',

                        label='dist',
                        

                        method='blyp',
                        basis='4-21G',
                        EmpiricalDispersion='GD3BJ',
                        scf='qc',addsec='''11 28 ={} B
    14 22 ={} B
    16 21 ={} B
    11 28 F
    14 22 F
    16 21 F'''.format(v,w,x))
    opt=GaussianOptimizer(mol2,calc)
    opt.run(fmax='tight', steps=100,opt='ModRedundant')

    j = Jedi(mol,mol2,modes)
    j.add_custom_bonds(get_hbonds(mol2))
    j.run()
    j.vmd_gen()

:download:`Starting geometry <dna/cg.xyz>`


The input written by ase for the cogef calculation has a line break too much between the coordinates section and the constraints section. So it has to be corrected manually and the job needs to be sent manually.
The gaussian outputs can be read by the funtions delivered with the Jedi package.

.. code-block:: python

    from ase.vibrations.vibrations import VibrationsData
    from jedi.jedi import Jedi
    from jedi.jedi import get_hbonds
    from jedi.io.gaussian import get_vibrations,read_gaussian_out

    file=open('output/opt.log')
    mol=read_gaussian_out(file)
    file2=open('output/dist.log')
    mol2=read_gaussian_out(file2)
    modes=get_vibrations('output/freq',mol)
    j=Jedi(mol,mol2,modes)
    j.add_custom_bonds(get_hbonds(mol2))

    j.run()
    j.vmd_gen()

.. image:: dna/cg.pdf
    :width: 30%

.. image:: dna/vmd/allcolorbar.pdf
    :width: 10%


:download:`Analysis output <dna/jedi.txt>`
:download:`All data <dna/dna.zip>`

Other types of interactions that can be localized between two atoms can added on the same way by giving a 2D array to the add_custom_bonds function. 

Analysis of a Substructure
==========================

Biphenyl
--------

It is possible to analyse substructures. This is desired when local changes of large structures need to be analysed. Here, a Hydrogen atom in a biphenyl molecule is pulled 0.1 Å away from its relaxed position. 
For the partial analysis, the hessian of only one phenyl ring is calculated yielding near identical values as when calculated for the whole system. 

.. code-block:: python

    import ase.io 
    from ase.calculators.vasp import Vasp
    from ase.vibratrions.vibrations import VibrationsData
    from jedi.jedi import Jedi
    import os

    mol=ase.io.read('start.xyz')

    #optimize the molecule
    label="opt"
    mol.calc=Vasp(label='%s/%s'%(label,label),
                    prec='Accurate',
                    xc='PBE',pp='PBE', 
                    nsw=0,ivdw=12,
                    lreal=False,ibrion=2, 
                    isym=0,symprec=1.0e-5,
                    encut=315,ediff=0.00001,isif=2,
                    command= "your command to start vasp jobs")

    mol.calc.write_input(mol)
    mol=ase.io.read('opt/vasprun.xml')  #vasp needs a specific ordering of the atoms writing and rereading will adapt this indexing
    mol.get_potential_energy()

    #frequency analysis
    label="freq"
    mol.calc=Vasp(label='%s/%s'%(label,label),
                    prec='Accurate',
                    xc='PBE',pp='PBE', 
                    nsw=0,ivdw=12,
                    lreal=False,ibrion=5, 
                    isym=0,symprec=1.0e-5,
                    encut=315,ediff=0.00001,isif=2,
                    command= "your command to start vasp jobs")
    mol.get_potential_energy()
    modes=mol.calc.get_vibrations()

    c = FixAtoms(indices=[6,7,8,9,10,11,17,18,19,20,21])
    mol.set_constraint(c)

    label='pfreq'
    calc3 = Vasp(label='pfreq/%s'%(label),prec='Accurate', ibrion=5,ediff=0.00001,
                xc='PBE',pp='PBE',ivdw=12,symprec=1.0e-5,encut=315,isym=0,
                lreal=False,command= "sh /home1/wang/vasp/submit-vasp-job.sh -la %s"%(label))

    mol.calc=calc3
    mol.get_potential_energy()
    partmodes=mol.calc.get_vibrations()
    np.savetxt('p-hessian',partmodes._hessian2d,fmt='%25s') #VibrationsData.write does not allow saving partial hessian

    mol.set_constraint()
    #distort molecule
    mol2=mol.copy()
    v=mol2.get_distance(3,14,vector=True)
    v/=np.linalg.norm(v)
    positions=mol2.get_positions()
    positions[14]+=v*0.1
    label='para-C-H'
    mol2.set_positions(positions)
    calc = Vasp(label='%s/%s'%(label,label),
                prec='Accurate',
                xc='PBE',pp='PBE', 
                nsw=0,ivdw=12,
                lreal=False,ibrion=2, 
                isym=0,symprec=1.0e-5,
                encut=315,ediff=0.00001,isif=2,
                command= "your command to start vasp jobs")
    mol2.calc=calc
    mol2.get_potential_energy()

    os.mkdir('all')
    os.chdir('all')
    j=Jedi(mol,mol2,modes)
    j.run()
    j.vmd_gen()

    os.chdir('../..')
    os.mkdir('partial')
    os.chdir('partial')
    jpart=Jedi(mol,mol2,partmodes)
    jpart.partial_analysis(indices=[0,1,2,3,4,5,12,13,14,15,16])
    jpart.vmd_gen()


:download:`Starting geometry <biphenyl/start.xyz>`

.. image:: biphenyl/biphg.png
    :width: 20%

.. image:: biphenyl/analysis/all/vmd/allcolorbar.pdf
    :width: 10%

.. image:: biphenyl/biphp.png
    :width: 20%

.. image:: biphenyl/analysis/partial/vmd/allcolorbar.pdf
    :width: 10%

:download:`Analysis output <biphenyl/analysis/all/jedi.txt>`
:download:`Analysis output <biphenyl/analysis/partial/jedi.txt>`


It is possible to only show specific RIC after calculating the whole analysis by giving a list of the desired atoms' indices to the run function.

.. code-block:: python
    
    os.chdir('../..')
    os.mkdir('special')
    os.chdir('special')
    jpart=Jedi(mol,mol2,modes)
    jpart.run(indices=[0,1,2,3,4,5,12,13,14,15,16])
    jpart.vmd_gen()

.. image:: biphenyl/biphs.png
    :width: 20%

.. image:: biphenyl/analysis/special/vmd/allcolorbar.pdf
    :width: 10%

:download:`Analysis output <biphenyl/analysis/special/jedi.txt>`
:download:`All data <biphenyl/biphenyl.zip>`

More Examples
=============

The following is intended to be an inspiration of what can also be analyzed.



Benzene
-------

Another way to distort molecules is to shrink boxes in periodic boundary conditions.

.. code-block:: python

    from ase.io import read
    from ase.build import molecule
    from ase.calculators.vasp import Vasp
    from ase.vibratrions.vibrations import VibrationsData
    from jedi.jedi import Jedi
    import os

    mol=molecule("C6H6")
    mol.set_cell=([20,20,20])
    mol.set_pbc([True,True,True])
    mol.center()
    #optimize molecule
    label='opt'
    mol.calc=Vasp(label='%s/%s'%(label,label),
                prec='Accurate',
                xc='PBE',pp='PBE',
                nsw=0,ivdw=12,
                lreal=False,ibrion=2, 
                isym=0,symprec=1.0e-5,
                encut=315,ediff=0.00001,isif=2,
                command= "your command to start vasp jobs")
    mol.calc.write_input(mol)
    mol=ase.io.read('opt/vasprun.xml')  
    mol.get_potential_energy()
    #frequency analysis
    label='freq'
    mol.calc=Vasp(label='%s/%s'%(label,label),
                prec='Accurate',
                xc='PBE',pp='PBE', 
                nsw=0,ivdw=12,
                lreal=False,ibrion=5, 
                isym=0,symprec=1.0e-5,
                encut=315,ediff=0.00001,isif=2,
                command= "your command to start vasp jobs")
    mol.get_potential_energy()
    modes=mol.calc.get_vibrations()
    #distort molecule
    mol2=mol.copy()
    mol2.cell=[8,8,8]
    label='8_8_8'
    calc = Vasp(label='%s/%s'%(label,label),
                prec='Accurate',
                xc='PBE',pp='PBE', 
                nsw=0,ivdw=12,
                lreal=False,ibrion=2, 
                isym=0,symprec=1.0e-5,
                encut=315,ediff=0.00001,isif=2,
                command= "your command to start vasp jobs")
    mol2.calc=calc
    mol2.get_potential_energy()

    os.chdir('8_8_8')
    j=Jedi(mol,mol2,modes)
    j.run()
    j.vmd_gen(modus='all', man_strain=0.655)

    os.chdir('../..')
    mol3=mol2.copy()
    mol3.cell=[6,6,6]
    label='6_6_6'
    calc = Vasp(label='%s/%s'%(label,label),
                prec='Accurate',
                xc='PBE',pp='PBE', 
                nsw=0,ivdw=12,
                lreal=False,ibrion=2, 
                isym=0,symprec=1.0e-5,
                encut=315,ediff=0.00001,isif=2,
                command= "your command to start vasp jobs")
    mol3.calc=calc
    mol3.get_potential_energy()
    os.chdir('6_6_6')
    j2=Jedi(mol,mol3,modes)
    j2.run()
    j2.vmd_gen(modus='all', man_strain=0.655)

.. image:: benzene/ben666.png
    :width: 18%

.. image:: benzene/analysis/6_6_6/vmd/allcolorbar.pdf
    :width: 10%

.. image:: benzene/ben888.png
    :width: 24%

.. image:: benzene/analysis/8_8_8/vmd/allcolorbar.pdf
    :width: 10%


:download:`8_8_8 analysis <benzene/analysis/8_8_8/jedi.txt>`
:download:`6_6_6 analysis <benzene/analysis/6_6_6/jedi.txt>`
:download:`All data <benzene/benzene.zip>`

For a better comparison of two seperate analyzes, one can set a reference strain energy for the coloring by using the man_strain parameter.



Using EFEI
-----------

Stretching bonds using a predefined force is possible with the EFEI method. The following example shows an ethane molecule of which the C-C bond is stretched with a force of 4 nN.

.. code-block:: python

    from ase.build import molecule
    from ase.vibratrions.vibrations import VibrationsData
    from jedi.jedi import Jedi
    from jedi.io.orca import get_vibrations
    from jedi.io.orca import OrcaOptimizer, ORCA
    import ase.io
    mol=molecule('C2H6')


    calc = ORCA(label='opt',
                orcasimpleinput='pbe cc-pVDZ OPT'
                ,task='opt')
    opt=OrcaOptimizer(mol,calc)
    opt.run()

    ase.io.write('opt.json',mol)
    mol=ase.io.read('opt.json')
    mol.calc=ORCA(label='orcafreq',
                orcasimpleinput='pbe cc-pVDZ FREQ',
                task='sp')
    mol.get_potential_energy()

    modes=get_vibrations('orcafreq',mol)

    mol2=mol.copy()
    calc = ORCA(label='stretch',
                orcasimpleinput='pbe cc-pVDZ  OPT',
                orcablocks='''%geom
        POTENTIALS
            { C 0 1 4.0 }
        end 
    end ''',task='opt')
    opt=OrcaOptimizer(mol2,calc)
    opt.run()
    ase.io.write('force.json',mol)

    j=Jedi(mol,mol2,modes)
    j.run()
    j.vmd_gen()

.. image:: ethane/ethan.png
    :width: 20%

.. image:: ethane/vmd/allcolorbar.pdf
    :width: 10%

:download:`Analysis output <ethane/jedi.txt>`
:download:`All data <ethane/ethane.zip>`


Hydrostatic Pressure using X-HCFF
---------------------------------

A lot of models have been developed to simulate pressure. X-HCFF is one of them that simulates Hydrostatic pressure. Here, Dewar and Ladenburg benzene are analyzed under 50GPa of pressure.

.. code-block:: python

    import ase.io
    from ase.calculators.qchem import QChem
    from ase.vibrations.data import VibrationsData
    from jedi.jedi import Jedi
    from ase.calculators.qchem import QChemOptimizer
    from jedi.jedi.io.qchem import get_vibrations
    mol = ase.io.read('Dewar.xyz')

    label='opt'
    
    calc=QChem(jobtype='sp',
                label='xhcff/50GB/%s'%(label),          
                method='pbe',dft_d='D3_BJ',
                basis='cc-pvdz',GEOM_OPT_MAX_CYCLES='150',
                USE_LIBQINTS='1',MAX_SCF_CYCLES='150',
                command='your command')
    mol.calc = calc
    opt = QChemOptimizer(mol)
    opt.run()


    label='freq'
    calc=QChem(jobtype='freq',
                label='xhcff/50GB/%s'%(label),          
                method='pbe',dft_d='D3_BJ',
                basis='cc-pvdz',vibman_print= '7',
                command='your command')
    mol.calc = calc
    mol.calc.calculate(properties=['hessian'],atoms=mol)

    modes=get_vibrations(label,mol)

    label='force'
    mol2=ase.io.read('%s.json'%(label))
    calc=QChem(jobtype='sp',
                label='xhcff/50GB/%s'%(label), 
                method='pbe',dft_d='D3_BJ',
                basis='cc-pvdz',
                GEOM_OPT_MAX_CYCLES='150',
                MAX_SCF_CYCLES='150',
                distort={'model':'xhcff','pressure':'50000','npoints_heavy':'302','npoints_hydrogen':'302','302','scaling':'1.0'},
                command='your command')
    mol2.calc = calc
    opt = QChemOptimizer(mol2)
    opt.run()
    ase.io.write('xhcff/50GB/%s.json'%(label),mol2)

    j=Jedi(mol,mol2,modes)
    j.run()
    j.vmd_gen()

In another folder the same for Ladenburg benzene:

.. code-block:: python

    import ase.io
    from ase.calculators.qchem import QChem
    from ase.vibrations.data import VibrationsData
    from jedi.jedi import Jedi
    from ase.calculators.qchem import QChemOptimizer
    from jedi.io.qchem import get_vibrations
    mol = ase.io.read('Ladenburg.xyz')

    label='opt'
    
    calc=QChem(jobtype='sp',
                label='xhcff/50GB/%s'%(label),          
                method='pbe',dft_d='D3_BJ',
                basis='cc-pvdz',GEOM_OPT_MAX_CYCLES='150',
                USE_LIBQINTS='1',MAX_SCF_CYCLES='150',
                command='your command')
    mol.calc = calc
    opt = QChemOptimizer(mol)
    opt.run()


    label='freq'
    calc=QChem(jobtype='freq',
                label='xhcff/50GB/%s'%(label),          
                method='pbe',dft_d='D3_BJ',
                basis='cc-pvdz',vibman_print= '7',
                command='your command')
    mol.calc = calc
    mol.calc.calculate(properties=['hessian'],atoms=mol)

    modes=get_vibrations(label,mol)

    label='force'
    mol2=ase.io.read('%s.json'%(label))
    calc=QChem(jobtype='sp',
                label='xhcff/50GB/%s'%(label), 
                method='pbe',dft_d='D3_BJ',
                basis='cc-pvdz',
                GEOM_OPT_MAX_CYCLES='150',
                MAX_SCF_CYCLES='150',
                distort={'model':'xhcff','pressure':'50000','npoints_heavy':'302','npoints_hydrogen':'302','302','scaling':'1.0'},
                command='your command')
    mol2.calc = calc
    opt = QChemOptimizer(mol2)
    opt.run()
    ase.io.write('xhcff/50GB/%s.json'%(label),mol2)
    
    j=Jedi(mol,mol2,modes)
    j.run()
    j.vmd_gen()

.. image:: xhcff/prisxh.pdf
    :width: 20%

:download:`dewar.xyz <xhcff/dewar/dewar.xyz>`
:download:`ladenburg.xyz <xhcff/ladenburg/ladenburg.xyz>`

:download:`Dewar analysis <xhcff/dewar/jedi.txt>`
:download:`Ladenburg analysis <xhcff/ladenburg/jedi.txt>`
:download:`All data <xhcff/xhcff.zip>`


Graphene
--------

Analysing functional materials is of particular interest. Graphene is shown as an example.

.. code-block:: python

    import ase.io
    from ase.vibrations.data import VibrationsData
    import ase.units
    from jedi.jedi import Jedi
    from ase.calculators.vasp import Vasp


    mol=ase.io.read('start.xyz')

    label='graphene'
  
    calc = Vasp(label='opt/%s'%(label),
                    prec='Accurate',
                    xc='PBE',pp='PBE', 
                    nsw=600,kpts=[6,6,1],
                    lreal=False,ibrion=2,ivdw=12,
                    isym=0,symprec=1.0e-5,
                    encut=315,ediff=0.00001,isif=2,
                    command= "your command")
    mol.calc=calc
    mol.calc.write_input(mol)
    mol=ase.io.read('opt/vasprun.xml')

    mol.get_potential_energy()
    ase.io.write('opt.json', mol)

    mol=ase.io.read('opt.json')

    label='graphene_freq'
    calc3 = Vasp(label='freq/%s'%(label),
                    prec='Accurate', ibrion=5,
                    xc='PBE',pp='PBE', 
                    ivdw=12,kpts=[6,6,1],
                    lreal=False,isym=0,symprec=1.0e-5,
                    encut=315,ediff=0.00001,isif=2,
                    command= "your command")

    mol.calc=calc3
    mol.get_potential_energy()
    modes=mol.calc.get_vibrations()
    modes.write('modes.json')


    mol2=ase.io.read('opt.json')
    cell=mol.get_cell()
    cell[0][0]-=0.1
    mol2.set_cell(cell)
    mol2.set_pbc([ True ,True,  True])
    label='graphenex-0_1'
    calc = Vasp(label='x-0_1/%s'%(label),
                prec='Accurate', 
                xc='PBE',pp='PBE',nsw=600,
                lreal=False,ibrion=2,ivdw=12,
                isym=0,symprec=1.0e-8,
                encut=315,ediff=0.00001,isif=2
                command="your command")
    mol2.calc=calc
    mol2.get_potential_energy()
    ase.io.write('x-0_1.json', mol2, format='json')
    j=Jedi(mol,mol2,modes)
    j.run()
    j.vmd_gen()

:download:`Starting geometry <graphene/start.xyz>`
:download:`Analysis output <graphene/analysis/jedi.txt>`
:download:`All data <graphene/graphene.zip>`

HCN
---

The HCN crystal is an interesting construct to examine bulk behavior. It consists of small molecules with strong intermolecular interactions. The standard Jedi analysis does not include those interactions.



.. code-block:: python

    from gpaw import GPAW , PW
    from ase.optimize import BFGS
    from ase.vibrations.vibrations import Vibrations
    import ase.io
    from ase.calculators.dftd3 import DFTD3

    from gpaw.analyse.vdwradii import vdWradii
    from ase.constraints import FixAtoms

    mol=ase.io.read('start.xyz')
    convergence={'energy': 0.00001}
    calc=DFTD3(dft=GPAW(xc='PBE',mode=PW(700),kpts=[3,2,2],convergence=convergence),damping='bj')
    mol.calc=calc

    opt=BFGS(mol)
    opt.run(fmax=0.05)

    calc=DFTD3(dft=GPAW(xc='PBE',mode=PW(700),kpts=[3,2,2],convergence=convergence,symmetry='off'),damping='bj')
    mol.calc=calc

    vib=Vibrations(mol)
    vib.run()
    vib.summary()
    modes=vib.get_vibrations()

    vib=Vibrations(mol,indices=[0,4,1,6,10,7])
    vib.run()
    vib.summary()
    partmodes=vib.get_vibrations()
    
    mol2=mol.copy()
    cell=mol2.get_cell()
    cell[2][2]-=0.1
    mol2.set_cell(cell)

    mol2.calc=calc

    dis=BFGS(mol2)
    dis.run(fmax=0.05)
    mol2.set_constraint()
    ase.io.write('dis.json',mol2)

    j=Jedi(mol,mol2,modes)

    
    j.run()
    j.vmd_gen()

    jpart=Jedi(mol,mol2,partmodes)
   

    jpart.partial_analysis(indices=[0,4,1,6,10,7])
    jpart.vmd_gen()

The visualization should look like following picture.

.. image:: hcn/all.pdf
    :width: 30%

.. image:: hcn/analysis/all/vmd/allcolorbar.pdf
    :width: 10%

.. image:: hcn/part.pdf
    :width: 30%

.. image:: hcn/analysis/part/vmd/allcolorbar.pdf
    :width: 10%

The existence of different values for chemical equivalent RIC is caused by the low accuracy which gives a low quality hessian.

To include the dipole interactions for this example, a modified version of the get_hbonds() function can be modified so that C atoms are seen as possible donors.

:download:`get_hbonds() <hcn/analysis/dipole.py>`

.. code-block:: python

    from dipole import get_hbonds

    j=Jedi(mol,mol2,modes)
    j.add_custom_bonds(get_hbonds(mol))
    
    j.run()
    j.vmd_gen()

    jpart=Jedi(mol,mol2,partmodes)
    j.add_custom_bonds(get_hbonds(mol))

    jpart.partial_analysis(indices=[0,4,1,6,10,7])
    jpart.vmd_gen()

With dipole interactions the visualization looks as follows

.. image:: hcn/alldipole.pdf
    :width: 30%

.. image:: hcn/analysis/alldipole/vmd/allcolorbar.pdf
    :width: 10%

.. image:: hcn/partdipole.pdf
    :width: 30%

.. image:: hcn/analysis/partdipole/vmd/allcolorbar.pdf
    :width: 10%

The outputs can be found here.

:download:`all <hcn/analysis/all/jedi.txt>`
:download:`part <hcn/analysis/part/jedi.txt>`
:download:`alldipole <hcn/analysis/alldipole/jedi.txt>`
:download:`partdipole <hcn/analysis/partdipole/jedi.txt>`
:download:`All data <hcn/hcn.zip>`
