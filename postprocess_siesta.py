#!/usr/bin/env python3


#---------------------------------------------------------------------------------
# User setting parameters
#------------------------------------
user_precursor_energy_per_atom = 0.0 # [eV] The reference energy (precursor alone) when calculating relative energy.
#---------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
# --- Prepare environment and clean previous results ---
'''
# Install libraries
pip install ase==3.26.0 scipy==1.13.0 psutil==7.0.0 gpaw==25.7.0
pip install pymsym==0.3.4
pip install spglib==2.6.0

# Siesta Installation
sudo apt update
sudo apt -y install cmake gfortran build-essential libopenmpi-dev libopenblas-dev 
sudo apt -y install libhdf5-dev pkg-config libreadline-dev
cd $HOME
wget https://gitlab.com/siesta-project/siesta/-/releases/5.4.0/downloads/siesta-5.4.0.tar.gz
tar xvf siesta-5.4.0.tar.gz
cd siesta-5.4.0
cmake -S . -B _build -DSIESTA_WITH_FLOOK="OFF"
cmake --build _build -j 4
sudo cmake --install _build
echo 'export SIESTA_PP_PATH=$HOME/siesta-5.4.0/Pseudo/ThirdParty-Tools/ONCVPSP/nc-sr-05_pbe_standard_psml' >> ~/.bashrc  # path of pseudo-potentials
source ~/.bashrc

### Pseudo-potantials
# http://www.icmab.es/siesta
# 
# use *.psml (>= Siesta 5.4.0)
# https://www.pseudo-dojo.org/
# cd $HOME/siesta-5.4.0/Pseudo/ThirdParty-Tools/ONCVPSP$
# (set) nc-sr-05_pbe_standard_psml.tgz
# tar xvf nc-sr-05_pbe_standard_psml.tgz
# 
# not use *.psml, then need to convert (< Siesta 5.4.0)
# psml2psf Si.psml > Si.psf

# Usage
pyton3 mol2crystal_siesta.py
'''
#---------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
# --- Library imports and warning settings ---
import os
import glob
import shutil
import numpy as np
from ase.io import read, write
#from ase.spacegroup import crystal
#from scipy.spatial.distance import pdist
import subprocess
import psutil
import re
from ase.geometry import cellpar_to_cell
#from ase.neighborlist import NeighborList
#from ase.data import vdw_radii, atomic_numbers

# Siesta settings
from ase.calculators.siesta import Siesta
from ase.filters import UnitCellFilter
from ase.optimize import BFGS, LBFGS, FIRE
from ase.units import Ry, Ha

# Warning settings
import warnings
warnings.filterwarnings("ignore", message="scaled_positions .* are equivalent")
#---------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
# --- Clean old outputs and temporary folders ---
if (os.path.exists('valid_structures_postprocess_old')):
    shutil.rmtree( 'valid_structures_postprocess_old')

if (os.path.exists('valid_structures_postprocess')):
    os.rename(     'valid_structures_postprocess','valid_structures_postprocess_old')

if (os.path.exists('optimized_structures_vasp_old')):
    shutil.rmtree( 'optimized_structures_vasp_old')   

if (os.path.exists('optimized_structures_vasp')):
    os.rename(     'optimized_structures_vasp','optimized_structures_vasp_old')

dirs_to_remove = ['temp', 'gaff_pbc_temp', 'reaxff_temp', 'dftb_temp', 'xtb_temp', 'mopac_temp', 
  'qe_temp', 'abinit_temp', 'openmx_temp', 'gpaw_temp', 'siesta_temp', 'cp2k_temp', 'nwchem_temp', 'elk_temp']
for dir_name in dirs_to_remove:
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
#---------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
# --- Configure thread count for OpenMP/OpenMPI --
cpu_count = psutil.cpu_count(logical=False)
os.environ["OMP_NUM_THREADS"] = '1'             # use OpenMPI
#os.environ["OMP_NUM_THREADS"] = str(cpu_count) # use OpenMP 
#---------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
# Output directories
os.makedirs("optimized_structures_vasp", exist_ok=True)
os.makedirs("valid_structures_postprocess", exist_ok=True)
#---------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
# Siesta optimization using *.psml pseudo-potentials
def siesta_optimize(fname, precursor_energy_per_atom):
    try:
        temp_dir = "siesta_temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        atoms = read(fname)
        
        os.environ['SIESTA_COMMAND'] = f'mpirun -np {cpu_count} /usr/local/bin/siesta'

        # Siesta calculator
        calc = Siesta(
            label         = os.path.join(temp_dir, 'siesta_calc'),
            xc            = 'PBE',                # LDA, PBE, revPBE, RPBE, WC, PBEsol, BLYP, (Including vdW: DRSLL, LMKLL, KBM)
            mesh_cutoff   = 200*Ry,               # Mesh cutoff (affects calculation accuracy, typically 150-300 Ry)
            energy_shift  = 0.01*Ry,              # Energy shift to avoid overlap between atoms (typically 0.01 Ry)
            basis_set     = 'DZP',                # Basis function size (SZ, DZ, DZP, TZP, etc.) (typically DZP)
            kpts          = (1, 1, 1),            # k-point mesh ((1,1,1) for molecules and isolated systems)
            fdf_arguments = {
                'SpinPolarized': False,           # Spin depolarization (set to True for magnetic calculations)）
                'SpinOrbit': False,               # Usually, when using SOI, spin polarization is also enabled.
                
                'SCF.ConvergenceTolerance': str(1.0e-3*len(atoms)/Ha), # 1 meV/atom -> Ha unit
                'DM.Tolerance': '1.d-4',          # Tolerance of Density Matrix. 
                'DM.MixingWeight': 0.1,           # Density matrix mixing coefficient (affects convergence stability)
                'MaxSCFIterations': 100,          # Maximum number of SCF iterations
                'SolutionMethod': 'diagon',       # Diagonalization methods (diagon, OMM, OrderN, CheSS)
                
                'VDWCorrection': True,            # wdW correlation: DFT-D2 or DFT-D3
                'VDWFunctional': 'DFTD3',         # DFTD2, DFTD3
                'VDWScaling': 1.0,                # 0.75:DFTD2, 1.0:DFTD3
                
                'ElectronicTemperature': '300 K', # Electron temperature (K) -> Affects convergence in metallic systems
                'OccupationFunction': 'FD',       # Fermi Distribution (FD), Methfessel-Paxton (MP), Cold
                
                #'WriteForces': True,              # Output force (useful for optimization)
                #'WriteKpoints': True,             # Output k-point information
                #'WriteCoorXmol': True,            # output *.xmol file
                #'SaveHS': True,                   # Save Hamiltonian and overlap matrix
                
                #'LatticeConstant': '1.0 Ang',     # Scaling of lattice parameters (if necessary)
            },
            pseudo_path   = os.environ.get("SIESTA_PP_PATH", "./psf"), # path of Pseudo-potentials
        )

        atoms.calc = calc

        ucf = UnitCellFilter(atoms)
        '''
        opt = FIRE(ucf,
            logfile=os.path.join(temp_dir, 'opt.log'),
            trajectory=os.path.join(temp_dir, 'opt.traj'))
        '''
        opt = LBFGS(ucf,
            logfile=os.path.join(temp_dir, 'opt.log'),
            trajectory=os.path.join(temp_dir, 'opt.traj'))
        opt.run(fmax=0.5)

        # Save optimized structure
        opt_fname = fname.replace("valid_structures", "optimized_structures_vasp").replace("POSCAR", "OPT") + ".vasp"
        write(opt_fname, atoms, format='vasp')

        # Energy and density
        energy_value = atoms.get_potential_energy()
        num_atoms = len(atoms)
        energy_per_atom = energy_value / num_atoms
        relative_energy_per_atom = energy_per_atom - precursor_energy_per_atom

        total_mass_amu = sum(atoms.get_masses())
        total_mass_g = total_mass_amu * 1.66053906660e-24
        volume = atoms.get_volume()
        volume_cm3 = volume * 1e-24
        density = total_mass_g / volume_cm3 if volume_cm3 > 0 else 0

        print(f"Final energy per atom: {energy_per_atom:.6f} [eV/atom]")
        print(f"Final relative energy per atom: {relative_energy_per_atom:.6f} [eV/atom]")
        print(f"Number of atoms: {num_atoms}")
        print(f"Volume: {volume:.6f} [A3]")
        print(f"Density: {density:.3f} [g/cm^3]")

        with open("structure_vs_energy.txt", "a") as out:
            out.write(f"{opt_fname} {relative_energy_per_atom:.6f} {energy_per_atom:.6f} {density:.3f} {num_atoms} {volume:.6f}\n")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error optimizing {fname}: {e}")
#---------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
# --- Reference energy from original molecule ---
with open("structure_vs_energy.txt", "w") as f:
    print("# POSCAR file, Relative Energy [eV/atom], Total Energy [eV/atom], Density [g/cm^3], Number of atoms, Volume [A^3]", file=f)
#---------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
# --- Main ---
print(f"------------------------------------------------------")
print("# Calculate valid structures")
valid_files = []
directory = "valid_structures"
for fname in os.listdir(directory):
    filepath = os.path.join(directory, fname)
    try:
        print(f"Calculate file: {fname}")
        siesta_optimize(filepath, precursor_energy_per_atom=user_precursor_energy_per_atom)
        
        opt_fname_path  = f"optimized_structures_vasp/{fname}".replace("POSCAR", "OPT") + ".vasp"
        post_fname_path = f"valid_structures_postprocess/{fname}"
        shutil.copy(opt_fname_path, post_fname_path)
        print(f"------------------------------------------------------")
        continue
    except Exception:
        continue
#---------------------------------------------------------------------------------

print("Finished space group search and Siesta optimization.")
