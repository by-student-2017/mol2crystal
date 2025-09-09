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

# lammps (stable_22Jul2025)
cd $HOME
sudo apt -y install cmake gfortran gcc libopenmpi-dev
git clone -b stable https://github.com/lammps/lammps.git
cd lammps
mkdir build && cd build
cmake -D BUILD_MPI=yes -D BUILD_SHARED_LIBS=no -D PKG_KSPACE=yes -D PKG_MOLECULE=yes -D PKG_EXTRA-MOLECULE=yes -D PKG_USER-MISC=yes -D PKG_EXTRA-DUMP=yes -D PKG_REAXFF=yes -D PKG_QEQ=yes -D PKG_MC=yes -D PKG_EAM=yes -D PKG_MEAM=yes -D PKG_RIGID=yes -D PKG_USER-CG-CMM=yes ../cmake
make -j$(nproc)
sudo make install
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Usage
pyton3 mol2crystal_gaff_pbc.py
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

# Lammps settings
from ase.io.lammpsdata import read_lammps_data
from ase.io.lammpsdata import write_lammps_data
from ase.data import atomic_masses

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
# --- Configure thread count for OpenMP/OpenMPI ---
cpu_count = psutil.cpu_count(logical=False)
os.environ["OMP_NUM_THREADS"] = '1'             # OpenMPI
#os.environ["OMP_NUM_THREADS"] = str(cpu_count) # OpenMP 
#---------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
# Output directories
os.makedirs("optimized_structures_vasp", exist_ok=True)
os.makedirs("valid_structures_postprocess", exist_ok=True)
#---------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
# Lammps (Reaxff) optimization
def reaxff_optimize(fname, precursor_energy_per_atom):
    try:
        temp_dir = "reaxff_temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        log_file = os.path.join(temp_dir, "system.log")

        for src_path in glob.glob("ffield.reax*"):
            dst_path = os.path.join(temp_dir, os.path.basename(src_path))
            shutil.copy(src_path, dst_path)

        atoms = read(fname)
        atoms.set_initial_charges([0.0] * len(atoms))
        crystal_data_path = os.path.join(temp_dir, "crystal.data")
        write(crystal_data_path, atoms, format="lammps-data", units="real", atom_style="charge")

        # Generate LAMMPS input file
        symbols = atoms.get_chemical_symbols()
        unique_symbols = sorted(set(symbols), key=symbols.index)
        elem_string = " ".join(unique_symbols)
        
        with open("in_reaxff.lmp", "r") as f:
            lines = f.readlines()
        
        # unique_symbols is generated from atoms.get_chemical_symbols()
        with open(os.path.join(temp_dir, "in_reaxff_temp.lmp"), "w") as f:
            for line in lines:
                if line.strip().startswith("variable elem string"):
                    f.write(f'variable elem string "{elem_string}"\n')
                    # Automatic mass generation using ASE's atomic_masses
                    for i, elem in enumerate(unique_symbols, start=1):
                        mass = atomic_masses[atoms[atoms.get_chemical_symbols().index(elem)].number]
                        f.write(f"mass {i} {mass:.3f}\n")
                else:
                    f.write(line)
        
        '''
        symbol_to_type = {sym: i+1 for i, sym in enumerate(unique_symbols)}
        with open(os.path.join(temp_dir, "atom_types.txt"), "w") as f:
            for sym, typ in symbol_to_type.items():
                f.write(f"{typ} {sym}\n")
        '''
        
        '''
        cell_lengths = atoms.get_cell().lengths()
        rep_x = max(1, int(12.8 // cell_lengths[0]) + 1)
        rep_y = max(1, int(12.8 // cell_lengths[1]) + 1)
        rep_z = max(1, int(12.8 // cell_lengths[2]) + 1)
        
        # Modify replicate line in the copied LAMMPS input file
        input_path = os.path.join(temp_dir, "in_gaff_pbc_temp.lmp")
        with open(input_path, "r") as file:
            lines = file.readlines()
        
        with open(input_path, "w") as file:
            for line in lines:
                if line.strip().startswith("replicate"):
                    file.write(f"replicate {rep_x} {rep_y} {rep_z}\n")
                else:
                    file.write(line)
        print(f"Updated replicate line to: replicate {rep_x} {rep_y} {rep_z} for >= 12.8 x 12.8 x 12.8 cell size")
        '''
        
        # Run LAMMPS
        #cmd = f"mpirun -np {cpu_count} lmp -in in_gaff_pbc_temp.lmp | tee ./../log.lammps"
        cmd = f"mpirun -np {cpu_count} lmp -in in_reaxff_temp.lmp"
        with open(log_file, "a") as log:
            subprocess.run(cmd, shell=True, cwd=temp_dir, stdout=log, stderr=subprocess.STDOUT)

        log_path = os.path.join(temp_dir, "log.lammps")

        # Step 7: Extract energy from log.lammps
        with open(log_path, "r") as f:
            log_content = f.read()
        matches = re.findall(r"The total energy \(kcal/mol\)\s*=\s*(-?\d+\.\d+)", log_content, re.MULTILINE)
        if matches:
            energy = float(matches[-1]) * 0.043361254529175 # kcal/mol -> eV
        else:
            print(f"Warning: Energy not found for {filename}. Skipping this entry.")

        # Defining a file path
        md_npt_path = os.path.join(temp_dir, "md_npt.data")
        md_nvt_path = os.path.join(temp_dir, "md_nvt.data")
        
        # Loading the Structure
        if os.path.exists(md_npt_path):
            print("Using NPT-optimized structure.")
            atoms = read_lammps_data(md_npt_path, atom_style="charge")
            atoms.wrap()
        elif os.path.exists(md_nvt_path):
            print("Warning: NPT structure not found. Using NVT-optimized structure.")
            atoms = read_lammps_data(md_nvt_path, atom_style="charge")
            atoms.wrap()
        else:
            print("Warning: Neither NPT nor NVT structure found. Using original structure.")
            atoms = read(fname)

        opt_fname = fname.replace("valid_structures", "optimized_structures_vasp").replace("POSCAR", "OPT") + ".vasp"
        write(opt_fname, atoms, format='vasp')

        # Step 8: Compute properties
        num_atoms = len(atoms)
        energy_per_atom = energy / num_atoms if num_atoms > 0 else 0.0
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
        reaxff_optimize(filepath, precursor_energy_per_atom=user_precursor_energy_per_atom)
        
        opt_fname_path  = f"optimized_structures_vasp/{fname}".replace("POSCAR", "OPT") + ".vasp"
        post_fname_path = f"valid_structures_postprocess/{fname}"
        shutil.copy(opt_fname_path, post_fname_path)
        print(f"------------------------------------------------------")
        continue
    except Exception:
        continue
#---------------------------------------------------------------------------------

print("Finished space group search and Lammps (ReaxFF) optimization.")
