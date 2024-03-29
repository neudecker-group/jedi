from setuptools import setup

setup(
    name='strainjedi',
    version='1.0.0',    
    description='The python distribution for the JEDI analysis',
    url='https://github.com/neudecker-group/jedi',
    author='Tim Neudecker, Sanna Benter, Henry Wang, Wilke Dononelli',
    author_email='neudecker@uni-bremen.de',
    license='MIT',
    packages=['strainjedi','strainjedi/io'],
    install_requires=['mpi4py>=2.0',
                      'numpy',  
                      'ase',
                      'pytest'                   
                      ],

    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        
        'Operating System :: POSIX :: Linux',       
        'Programming Language :: Python :: 3.8',
    ],
)