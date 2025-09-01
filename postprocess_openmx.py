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
pip install ase==3.22.1 scipy==1.13.0 psutil==7.0.0
pip install pymsym==0.3.4
pip install spglib==2.6.0

# OpenMX v3.8.5
sudo apt update
sudo apt -y install openmx

# Usage
pyton3 mol2crystal_openmx.py
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

# OpenMX settings
from ase.units import Ha, Ry
from ase.calculators.openmx import OpenMX
from ase.optimize import BFGS, LBFGS, FIRE

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
# OpenMX optimization
def openmx_optimize(fname, precursor_energy_per_atom):
    try:
        temp_dir = "openmx_temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        atoms = read(fname)
        cwd = os.getcwd()

        # ----------------------------------------------------------------------------------------
        # OpenMX calculator settings
        # ASE, OpenMX manual: https://ase-lib.org/ase/calculators/openmx.html
        #   Note: tpl = (), lst = [], int = 0, flt = 0.0, str = '' or ""
        # ASE source code: https://ase-lib.org/_modules/ase/calculators/openmx/openmx.html#OpenMX
        # OpenMX v.3.8 manual: https://www.openmx-square.org/openmx_man3.8/openmx.html
        #   Keywords: https://www.openmx-square.org/openmx_man3.8/node24.html
        # ----------------------------------------------------------------------------------------
        # Note: I have confirmed that all of the following configurations work, so choose the method you prefer.
        # ----------------------------------------------------------------------------------------
        #os.environ["ASE_OPENMX_COMMAND"] = "/usr/bin/openmx"                        # OpenMP  (command is also acceptable)
        #os.environ["ASE_OPENMX_COMMAND"] = f"mpirun -np {cpu_coun} /usr/bin/openmx" # OpenMPI (command is also acceptable)
        #os.environ["OPENMX_DFT_DATA_PATH"] = "/usr/share/openmx/DFT_DATA13"         # Pseudopotential and basis function directories ('data_path' is also acceptable)
        # ----------------------------------------------------------------------------------------
        calc = OpenMX(
            # Calculator parameters
            #---------------------------------------
            command       = f'mpirun -np {cpu_count} /usr/bin/openmx', # OpenMX path (ASE_OPENMX_COMMAND environment variable is also acceptable)
            #---------------------------------------
            #command       = '/usr/bin/openmx',      # OpenMX path (ASE_OPENMX_COMMAND environment variable is also acceptable)
            #mpi={'processes': int(cpu_count), 'threads': int(os.getenv("OMP_NUM_THREADS", "1"))}, # Parallel execution settings (specified as a dictionary)
            #---------------------------------------
            data_path     = '/usr/share/openmx/DFT_DATA13',  # Pseudopotential path (OPENMX_DFT_DATA_PATH environment variable is also acceptable)
            #---------------------------------------
            label='openmx_calc',                    # Used for output file name and System.Name (default: openmx)
            restart       = None,                   # Restart settings (new calculation with None)
            debug         = False,                  # Debug Output
            nohup         = True,                   # Run in the background with nohup
            dft_data_dict = None,                   # Basis function settings for each atomic type (if necessary)
            
            #dft_data_dict = { # DFT-D2 settings (OpenMX ver.3.9)
            #    'scf.dftD': 'on',
            #    'version.dftD': '2',
            #    'DFTD.Unit': 'Ang',
            #    'DFTD.rcut_dftD': '100.0',
            #    'DFTD.cncut_dftD': '40',
            #    'DFTD.IntDirection': '1 1 1',
            #    'DFTD.scale6': '0.75'
            #},
            
            # Standard parameters
            xc            = 'GGA-PBE',              # Exchange-correlation functions (e.g., LDA, LSDA-CA, LSDA-PW, GGA-PBE)
            #energy_cutoff = 150*Ry,                # Energy cutoff [Ry] (Default GPAW values)
            convergence   = 1e-3 * len(atoms) / Ha, # SCF convergence condition (convert 1 meV/atom to Ha unit)
            kpts          = (1, 1, 1),              # k-point. 1x1x1
            eigensolver   = 'DC',                   # DC, Krylov, ON2, Cluster, Band
            spinpol       = 'OFF',                  # Non-spin polarized calculation (no magnetism)
            external      = [0.0,0.0,0.0],          # External Electric Field: default=0.0 0.0 0.0 (GV/m)
            mixer         = 'RMM-DIISK',            # Simple, GR-Pulay, RM-DIIS, Kerker, RMM-DIISK, RM-DIISH
            charge        = 0.0,                    # Total charge of the system (unspecified)
            maxiter       = 200,                    # Maximum number of SCF cycles
            smearing      = 300,                    # Smearing temperature (K) (Default vaule: 300 [K])
            
            # Density of States (DOS)
            dos_fileout   = False,                  # DOS output availability
            dos_erange    = (-25, 20),              # DOS energy range (eV)
            dos_kgrid     = (2, 2, 2),              # k-points for DOS (same as kpts if not specified)
        )

        os.chdir(temp_dir)
        try:
            atoms.calc = calc
            #opt = LBFGS(atoms)
            opt = FIRE(atoms)
            opt.run(fmax=0.5)
        finally:
            os.chdir(cwd)

        # Save optimized structure
        opt_fname = fname.replace("valid_structures", "optimized_structures_vasp").replace("POSCAR", "OPT") + ".vasp"
        write(opt_fname, atoms, format='vasp')

        # Energy and density calculations
        energy_per_atom = atoms.get_potential_energy() / len(atoms)
        relative_energy_per_atom = energy_per_atom - precursor_energy_per_atom

        total_mass_amu = sum(atoms.get_masses())
        total_mass_g = total_mass_amu * 1.66053906660e-24
        volume = atoms.get_volume()
        volume_cm3 = volume * 1e-24
        density = total_mass_g / volume_cm3 if volume_cm3 > 0 else 0

        print(f"Final energy per atom: {energy_per_atom:.6f} [eV/atom]")
        print(f"Final relative energy per atom: {relative_energy_per_atom:.6f} [eV/atom]")
        print(f"Number of atoms: {len(atoms)}")
        print(f"Volume: {volume:.6f} [A^3]")
        print(f"Density: {density:.3f} [g/cm^3]")

        with open("structure_vs_energy.txt", "a") as out:
            out.write(f"{opt_fname} {relative_energy_per_atom:.6f} {energy_per_atom:.6f} {density:.3f} {len(atoms)} {volume:.6f}\n")
        input()
    except Exception as e:
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
        openmx_optimize(filepath, precursor_energy_per_atom=user_precursor_energy_per_atom)
        
        opt_fname_path  = f"optimized_structures_vasp/{fname}".replace("POSCAR", "OPT") + ".vasp"
        post_fname_path = f"valid_structures_postprocess/{fname}"
        shutil.copy(opt_fname_path, post_fname_path)
        print(f"------------------------------------------------------")
        continue
    except Exception:
        continue
#---------------------------------------------------------------------------------

print("Finished space group search and Lammps (GAFF) optimization.")
