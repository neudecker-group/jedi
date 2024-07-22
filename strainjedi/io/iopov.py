#!/usr/bin/env python
"""Simplified POV-RAY environment creator."""

import os
from pathlib import Path
import numpy as np
from ase.data.colors import jmol_colors
from ase.data import covalent_radii
from ase.data.vdw import vdw_radii


class POV:
    """Class to create .pov files to be executed with the pov-ray raytracer.
    Initialize with

    atoms : an ase atoms object

    Keyword arguments: (all distances in Angstroms)
    ------------------
    tex : a texture to use for the atoms, either a single value or a list
        of len(atoms), default = 'vmd'
    radii : atomic radii. if a single value is given, it is interpreted as
        a multiplier for the covalent radii in ase.data. if a list of
        len(atoms) is given, it is interpreted as individual atomic radii.
        default = 1.0
    colors : a list of len(atoms) of the colors, as (r,g,b). default is
        None which will use ASE standard colors
    cameratype : type of povray camera, default='perspective'
    cameralocation : location of camera as an (x,y,z) tuple,
        default = (0., 0., 20)
    look_at : where the camera is pointed at as an (x,y,z) tuple.
        default = (0., 0., 0.)
    camera_right_up : the right and up vectors that define the image
        boundaries. The right:up ratio will also define the aspect ratio
        (width:height) of the resulting image. These two vectors should
        generally be orthogonal -- generally on the x,y plane.
        default = [(-8.,0.,0.),(0.,6.,0.)]
    cameradirection : the initial direction vector of the camera before
        it is moved with look_at. This will also control the zoom, with
        higher values being more zoomed in. default = (0., 0., 10.)
    area_light : location and parameters of area light as [(x,y,x), color,
        width, height, Nlamps_x, Nlamps_y], default = [(20., 3., 40.),
        'White', .7, .7, 3, 3]
    background : background color, default = 'White'
    bondatoms : list of atoms to be bound together, as in
        [(index1, index2), ...], default = None
    bondradii : radii to use in drawing bonds, default = 0.1
    pixelwidth : width in pixels of the final image. Note that the height
        is set by the aspect ratio (controlled by carmera_right_up).
        default = 320
    clipplane : plane at which to clip atoms, for example "y, 0.00".
        default = None
    cell : unit/simulation cell of the system. Creates a box
        default = None
    """

    _default_settings = {
        'tex': 'vmd',
        'radii': 1.,
        'scale_radii': None,
        'atom_colors': None,
        'bond_colors': None,
        'alpha': True,
        'cameratype': 'perspective',
        'cameralocation': (0., 0., 20.),
        'look_at': (0., 0., 0.),
        'camera_right_up': [(-8., 0., 0.), (0., 6., 0.)],
        'cameradirection': (0., 0., 10.),
        'area_light': [(20., 3., 40.), 'White', .7, .7, 3, 3],
        'background': 'White',
        'bondatoms': None,
        'bondradius': .1,
        'aspectratio': None,
        'pixelwidth': 320,
        'clipplane': None,
        'cell': None,
    }

    def __init__(self, atoms, **kwargs):
        for k, v in self._default_settings.items():
            setattr(self, '_' + k, kwargs.pop(k, v))
        if len(kwargs) > 0:
            print(kwargs)
            raise TypeError('POV got one or more unexpected keywords.')
        self._atoms = atoms
        self._numbers = atoms.get_atomic_numbers()
        if self._atom_colors is None:
            self._atom_colors = jmol_colors[self._numbers]
        if self._bond_colors is None:
            self._bond_colors = np.array([(1.000, 1.000, 1.000)]*len(self._bondatoms))
        if (type(self._radii) is float) or (type(self._radii) is int):
            self._radii = covalent_radii[self._numbers] * self._radii
        if self._bondatoms is None:
            self._bondatoms = []
        if self._scale_radii is None:
           self._scale_radii=[0.5]*len(self._atoms)
        if self._aspectratio is None:
            self._aspectratio = (np.linalg.norm(self._camera_right_up[0]) /
                                 np.linalg.norm(self._camera_right_up[1]))

    def write(self, filename, label, run_povray=None):
        """Writes out the .pov file for ray-tracing and also an associated
        .ini file. If filename ends in ".png" it will run povray to turn it
        into a png file. If the filename ends in ".pov" it will not. This can
        be overridden with the keyword run_povray.
        """
        if isinstance(label, str):
            destination_dir = Path(label)
        elif isinstance(label, Path):
            destination_dir = label
        else:
            raise TypeError("Please specify the directory (label) to write vmd scripts to as Path or string")
        destination_dir.mkdir(parents=True, exist_ok=True)

        if filename.endswith('.png'):
            filename = filename[:-4] + '.pov'
            if run_povray is None:
                run_povray = True
        elif filename.endswith('.pov'):
            if run_povray is None:
                run_povray = False
        else:
            raise RuntimeError('filename must end in .pov or .png')
        self._filename = filename
        filebase = filename[:-4]
        # Write the .pov file.
        f = open(destination_dir / Path(filebase + '.pov'), 'w')

        def w(text):
            f.write(text + '\n')

        w('#include "colors.inc"')
        w('#include "finish.inc"')
        w('')
        w('global_settings {assumed_gamma 1 max_trace_level 6}')
        w('background {color %s}' % self._background)
        w('camera {%s' % 'perspective')
        w('  location <%.2f,%.2f,%.2f>' % tuple(self._cameralocation))
        camera_right_up = self._camera_right_up
        w('  right <%.2f,%.2f,%.2f> up <%.2f,%.2f,%.2f>' %
          (camera_right_up[0][0], camera_right_up[0][1],
           camera_right_up[0][2], camera_right_up[1][0],
           camera_right_up[1][1], camera_right_up[1][2]))
        w('  direction <%.2f,%.2f,%.2f>' % tuple(self._cameradirection))
        w('  look_at <%.2f,%.2f,%.2f>}' % tuple(self._look_at))
        w('light_source {<%.2f,%.2f,%.2f> color %s' %
          tuple(list(self._area_light[0]) + [self._area_light[1]]))
        w('  area_light <%.2f,0,0>, <0,%.2f,0>, %i, %i' %
          tuple(self._area_light[2:]))
        w('  adaptive 1 jitter}')
        w('')
        w('#declare simple = finish {phong 0.7}')
        w('#declare pale = finish {ambient .5 diffuse .85 roughness .001 specular 0.200 }')
        w('#declare intermediate = finish {ambient 0.3 diffuse 0.6 specular 0.60 roughness 0.04 }')
        w('#declare vmd = finish {ambient .0 diffuse .65 phong 0.1 phong_size 40. specular 0.500 }')
        w('#declare metal = finish {ambient 0.05 brilliance 2 diffuse 0.6 metallic specular 0.80 roughness 1/120 reflection 0.3 }')
        w('#declare chrome = finish {ambient 0.3 diffuse 0.7 reflection 0.15 brilliance 8 specular 0.8 roughness 0.1 }')
        w('#declare rubber = finish { ambient 0.2 diffuse 1 brilliance 1 phong 0.2 phong_size 20 specular 0 roughness 0.05 metallic 0 reflection { 0 0 fresnel on metallic 0 } conserve_energy }')
        w('#declare jmol = finish {ambient .2 diffuse .6 specular 1 roughness .001 metallic}')
        w('#declare ase2 = finish {ambient 0.05 brilliance 3 diffuse 0.6 metallic specular 0.70 roughness 0.04 reflection 0.15}')
        w('#declare ase3 = finish {ambient .15 brilliance 2 diffuse .6 metallic specular 1. roughness .001 reflection .0}')
        w('#declare ase4 = finish {ambient 0.05 brilliance 3 diffuse 0.6 specular 0.70 roughness 0.04 reflection 0.005}')
        w('#declare glas = finish {ambient .05 diffuse .3 specular 1. roughness .001}')
        w('#declare Rbond = %.3f;' % self._bondradius)
        w('#declare Rcell = %.3f;' % 0.015)
        w('')
        if self._clipplane is None:
            w('#macro atom(LOC, R, COL, FIN)')
            w('  sphere{LOC, R texture{pigment{COL} finish{FIN}}}')
            w('#end')
        else:
            w('#macro atom(LOC, R, COL, FIN)')
            w('  difference{')
            w('   sphere{LOC, R}')
            w('   plane{%s}' % self._clipplane)
            w('   texture{pigment{COL} finish{FIN}}')
            w('  }')
            w('#end')
        w('')

        if (type(self._tex)!=list) and (type(self._tex)!=np.ndarray) :
           self._tex=[self._tex] * len(self._atoms)
        for atom in self._atoms:
            w('atom(<%.2f,%.2f,%.2f>, %.2f, rgb <%.2f,%.2f,%.2f>, %s) // #%i'
              % (atom.x, atom.y, atom.z,
                 self._radii[atom.index]*self._scale_radii[atom.index], self._atom_colors[atom.index][0],
                 self._atom_colors[atom.index][1], self._atom_colors[atom.index][2],
                 self._tex[atom.index], atom.index))

        for i, bond in enumerate(self._bondatoms):
            pos0 = self._atoms[bond[0]].position.copy()
            pos1 = self._atoms[bond[1]].position.copy()
            color = self._bond_colors[i]
            w('cylinder {<%.2f,%.2f,%.2f>, <%.2f,%.2f,%.2f>, Rbond '
              'texture{pigment {rgb <%.2f,%.2f,%.2f>} finish{%s}}} '
              ' // # %i to %i' %
              (pos0[0], pos0[1], pos0[2],
               pos1[0], pos1[1], pos1[2],
               color[0], color[1], color[2], self._tex[bond[0]],
               bond[0], bond[1]))
        if self._cell is not None:
            a=np.array(self._cell[0])
            b=np.array(self._cell[1])
            c=np.array(self._cell[2])
            zero=np.zeros(3)
            w('cylinder {<%.2f,%.2f,%.2f>, <%.2f,%.2f,%.2f>, Rcell '
              'texture{pigment {rgb <%.2f,%.2f,%.2f>} finish{%s}}} '
              ' // # celledge' %
              (zero[0], zero[1], zero[2],
               a[0], a[1], a[2],
               0, 0, 0, self._tex[0]))
            w('cylinder {<%.2f,%.2f,%.2f>, <%.2f,%.2f,%.2f>, Rcell '
              'texture{pigment {rgb <%.2f,%.2f,%.2f>} finish{%s}}} '
              ' // # celledge' %
              (zero[0], zero[1], zero[2],
               b[0], b[1], b[2],
               0, 0, 0, self._tex[0]))
            w('cylinder {<%.2f,%.2f,%.2f>, <%.2f,%.2f,%.2f>, Rcell '
              'texture{pigment {rgb <%.2f,%.2f,%.2f>} finish{%s}}} '
              ' // # celledge' %
              (b[0], b[1], b[2],
               a[0]+b[0], a[1]+b[1], a[2]+b[2],
               0, 0, 0, self._tex[0]))
            w('cylinder {<%.2f,%.2f,%.2f>, <%.2f,%.2f,%.2f>, Rcell '
              'texture{pigment {rgb <%.2f,%.2f,%.2f>} finish{%s}}} '
              ' // # celledge' %
              (a[0], a[1], a[2],
               a[0]+b[0], a[1]+b[1], a[2]+b[2],
               0, 0, 0, self._tex[0]))
            
            w('cylinder {<%.2f,%.2f,%.2f>, <%.2f,%.2f,%.2f>, Rcell '
              'texture{pigment {rgb <%.2f,%.2f,%.2f>} finish{%s}}} '
              ' // # celledge' %
              (zero[0], zero[1], zero[2],
               c[0], c[1], c[2],
               0, 0, 0, self._tex[0]))
            w('cylinder {<%.2f,%.2f,%.2f>, <%.2f,%.2f,%.2f>, Rcell '
              'texture{pigment {rgb <%.2f,%.2f,%.2f>} finish{%s}}} '
              ' // # celledge' %
              (a[0], a[1], a[2],
               a[0]+c[0], a[1]+c[1], a[2]+c[2],
               0, 0, 0, self._tex[0]))
            w('cylinder {<%.2f,%.2f,%.2f>, <%.2f,%.2f,%.2f>, Rcell '
              'texture{pigment {rgb <%.2f,%.2f,%.2f>} finish{%s}}} '
              ' // # celledge' %
              (b[0], b[1], b[2],
               b[0]+c[0], b[1]+c[1], b[2]+c[2],
               0, 0, 0, self._tex[0]))
            w('cylinder {<%.2f,%.2f,%.2f>, <%.2f,%.2f,%.2f>, Rcell '
              'texture{pigment {rgb <%.2f,%.2f,%.2f>} finish{%s}}} '
              ' // # celledge' %
              (a[0]+b[0], a[1]+b[1], a[2]+b[2],
               a[0]+b[0]+c[0], a[1]+b[1]+c[1], a[2]+b[2]+c[2],
               0, 0, 0, self._tex[0]))
            
            w('cylinder {<%.2f,%.2f,%.2f>, <%.2f,%.2f,%.2f>, Rcell '
              'texture{pigment {rgb <%.2f,%.2f,%.2f>} finish{%s}}} '
              ' // # celledge' %
              (c[0]+zero[0], c[1]+zero[1], c[2]+zero[2],
               c[0]+a[0], c[1]+a[1], c[2]+a[2],
               0, 0, 0, self._tex[0]))
            w('cylinder {<%.2f,%.2f,%.2f>, <%.2f,%.2f,%.2f>, Rcell '
              'texture{pigment {rgb <%.2f,%.2f,%.2f>} finish{%s}}} '
              ' // # celledge' %
              (c[0]+zero[0], c[1]+zero[1], c[2]+zero[2],
               c[0]+b[0], c[1]+b[1], c[2]+b[2],
               0, 0, 0, self._tex[0]))
            w('cylinder {<%.2f,%.2f,%.2f>, <%.2f,%.2f,%.2f>, Rcell '
              'texture{pigment {rgb <%.2f,%.2f,%.2f>} finish{%s}}} '
              ' // # celledge' %
              (c[0]+b[0], c[1]+b[1], c[2]+b[2],
               c[0]+a[0]+b[0], c[1]+a[1]+b[1], c[2]+a[2]+b[2],
               0, 0, 0, self._tex[0]))
            w('cylinder {<%.2f,%.2f,%.2f>, <%.2f,%.2f,%.2f>, Rcell '
              'texture{pigment {rgb <%.2f,%.2f,%.2f>} finish{%s}}} '
              ' // # celledge' %
              (c[0]+a[0], c[1]+a[1], c[2]+a[2],
               c[0]+a[0]+b[0], c[1]+a[1]+b[1], c[2]+a[2]+b[2],
               0, 0, 0, self._tex[0]))
            
#            // -- Coming now: 12 cell edges --
#
#union {
#  cylinder { <-8.145095,-3.870409,3.729225>, <7.943105,-3.870409,3.729225>, 0.013669  material { M_BD4 } }	// #4: * -- *
#  cylinder { <-8.145095,-3.870409,3.729225>, <-8.145095,4.173691,3.729225>, 0.013669  material { M_BD4 } }	// #5: * -- *
#  cylinder { <-8.145095,-3.870409,3.729225>, <-8.145095,-3.870409,-4.314875>, 0.013669  material { M_BD4 } }	// #6: * -- *
#  cylinder { <7.943105,-3.870409,3.729225>, <7.943105,4.173691,3.729225>, 0.013669  material { M_BD4 } }	// #7: * -- *
#  cylinder { <7.943105,-3.870409,3.729225>, <7.943105,-3.870409,-4.314875>, 0.013669  material { M_BD4 } }	// #8: * -- *
#  cylinder { <-8.145095,4.173691,3.729225>, <7.943105,4.173691,3.729225>, 0.013669  material { M_BD4 } }	// #9: * -- *
#  cylinder { <-8.145095,4.173691,3.729225>, <-8.145095,4.173691,-4.314875>, 0.013669  material { M_BD4 } }	// #10: * -- *
#  cylinder { <7.943105,4.173691,3.729225>, <7.943105,4.173691,-4.314875>, 0.013669  material { M_BD4 } }	// #11: * -- *
#  cylinder { <-8.145095,-3.870409,-4.314875>, <7.943105,-3.870409,-4.314875>, 0.013669  material { M_BD4 } }	// #12: * -- *
#  cylinder { <-8.145095,-3.870409,-4.314875>, <-8.145095,4.173691,-4.314875>, 0.013669  material { M_BD4 } }	// #13: * -- *
#  cylinder { <7.943105,-3.870409,-4.314875>, <7.943105,4.173691,-4.314875>, 0.013669  material { M_BD4 } }	// #14: * -- *
#  cylinder { <-8.145095,4.173691,-4.314875>, <7.943105,4.173691,-4.314875>, 0.013669  material { M_BD4 } }	// #15: * -- *
#
#  rotate <0.00000,-0.00000,0.00000>
#}
        f.close()

        # Write the .ini file.
        f = open(destination_dir / Path(filebase + '.ini'), 'w')
        w('Input_File_Name=%s' % os.path.split(filename)[1])
        w('Output_to_File=True')
        w('Output_File_Type=N')
        if self._alpha==True:
           w('Output_Alpha=On')
        else:
           w('Output_Alpha=False')
        w('Width=%d' % self._pixelwidth)
        w('Height=%.0f' % (self._pixelwidth / self._aspectratio))
        w('Antialias=True')
        w('Antialias_Threshold=0.1')
        w('Display=False')
        w('Pause_When_Done=True')
        w('Verbose=False')

        f.close()
        if run_povray:
            self.raytrace(filename, destination_dir=destination_dir)

    def raytrace(self, filename=None, destination_dir=None):
        """Run povray on the generated file."""

        if not filename:
            filename = self._filename
        path = Path(filename)
        if destination_dir:
            path = destination_dir / filename

        if path.parent != Path(''):
            pwd = Path.cwd()
            os.chdir(path.parent)

        os.system(f'povray {path.stem}.ini')

        if path.parent != Path(''):
            os.chdir(pwd)
