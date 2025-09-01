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

# xTB ver. 6.7.1
cd $HOME
wget https://github.com/grimme-lab/xtb/releases/download/v6.7.1/xtb-6.7.1-linux-x86_64.tar.xz
tar xvf xtb-6.7.1-linux-x86_64.tar.xz
echo 'export PATH=$PATH:$HOME/xtb-dist/bin' >> ~/.bashrc
source ~/.bashrc

# Usage
pyton3 mol2crystal_xtb.py
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

# xTB settings
from ase.units import Bohr

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
# xTB calculation
def xtb_optimize(fname, precursor_energy_per_atom):
    try:
        temp_dir = "xtb_temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        atoms = read(fname)
        original_cell = atoms.get_cell()

        # POSCAR file (VASP)
        temp_poscar = os.path.join(temp_dir, "input.poscar")
        write(temp_poscar, atoms, format="vasp")
        shutil.copy("xtb.inp", os.path.join(temp_dir, "xtb_temp.inp"))
        xtb_cmd = ["mpirun", "-np", str(cpu_count), "xtb", "input.poscar", "--periodic", "--opt", "--gfn", "1", "--input", "xtb_temp.inp"]
        with open(os.path.join(temp_dir, "xtb_output.log"), "w") as log_file:
            result = subprocess.run(xtb_cmd, cwd=temp_dir, stdout=log_file, stderr=subprocess.STDOUT)

        # save result regardless of convergence
        opt_out = os.path.join(temp_dir, "xtbopt.poscar")
        last_out = os.path.join(temp_dir, "xtblast.poscar")

        optimized = None
        source = None
        
        if os.path.exists(opt_out):
            optimized = read(opt_out, format='vasp')
            source = "xtbopt.poscar"
            print("Geometry optimization converged")
        elif os.path.exists(last_out):
            optimized = read(last_out, format='vasp')
            source = "xtblast.poscar"
            print("Note!!! Geometry optimization is not converged")
        else:
            print(f"[Error] No structure file found for {fname}")
            return

        # Cell conversion
        cell = optimized.get_cell() * Bohr
        optimized.set_cell(cell)

        opt_fname = fname.replace("valid_structures", "optimized_structures_vasp").replace("POSCAR", "OPT") + ".vasp"
        write(opt_fname, optimized, format='vasp')
        print(f"[{source}] Saved: {opt_fname}")
        
        # --- Extract only the last energy value ---
        log_path = os.path.join(temp_dir, "xtbopt.log")
        energy_value = None
        
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                for line in reversed(f.readlines()):
                    match = re.search(r"energy:\s*(-?\d+\.\d+)", line)
                    if match:
                        energy_value = float(match.group(1))
                        break
        
        if energy_value is not None:
            num_atoms = len(atoms) # or num_atoms = atoms.get_global_number_of_atoms()
            energy_per_atom = energy_value / num_atoms * 27.2114
            relative_energy_per_atom = energy_per_atom - precursor_energy_per_atom
        
            # --- density calculation ---
            total_mass_amu = sum(optimized.get_masses())
            total_mass_g = total_mass_amu * 1.66053906660e-24
            volume = optimized.get_volume()
            volume_cm3 = volume * 1e-24
            density = total_mass_g / volume_cm3 if volume_cm3 > 0 else 0
        
            print(f"Final energy per atom: {energy_per_atom:.6f} [eV/atom]")
            print(f"Final relative energy per atom: {relative_energy_per_atom:.6f} [eV/atom]")
            print(f"Number of atoms: {num_atoms}")
            print(f"Volume: {volume:.6f} [A3]")
            print(f"Density: {density:.3f} [g/cm^3]")
        
            with open("structure_vs_energy.txt", "a") as out:
                out.write(f"{opt_fname} {relative_energy_per_atom:.6f} {energy_per_atom:.6f} {density:.3f} {num_atoms} {volume:.6f}\n")
        else:
            print("Energy value not found in xtbopt.log.")
        
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
        xtb_optimize(filepath, precursor_energy_per_atom=user_precursor_energy_per_atom)
        
        opt_fname_path  = f"optimized_structures_vasp/{fname}".replace("POSCAR", "OPT") + ".vasp"
        post_fname_path = f"valid_structures_postprocess/{fname}"
        shutil.copy(opt_fname_path, post_fname_path)
        print(f"------------------------------------------------------")
        continue
    except Exception:
        continue
#---------------------------------------------------------------------------------

print("Finished space group search and xTB optimization.")
