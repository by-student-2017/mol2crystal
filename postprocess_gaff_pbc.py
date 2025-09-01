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

# moltemplate + antechamber + mol22lt.pl
sudo apt update
sudo apt -y install dos2unix python3-pip libgfortran5 liblapack3
wget https://github.com/makoto-yoneya/makoto-yoneya.github.io/raw/master/LAMMPS-organics/install_moltemplate.sh
sh install_moltemplate.sh
wget https://github.com/makoto-yoneya/makoto-yoneya.github.io/raw/master/LAMMPS-organics/install_WSLmisc.sh
sh install_WSLmisc.sh

# lammps (stable_22Jul2025)
cd $HOME
sudo apt -y install cmake gfortran gcc libopenmpi-dev
git clone -b stable https://github.com/lammps/lammps.git
cd lammps
mkdir build && cd build
cmake -D BUILD_MPI=yes -D BUILD_SHARED_LIBS=no -D PKG_KSPACE=yes -D PKG_MOLECULE=yes -D PKG_EXTRA-MOLECULE=yes -D PKG_USER-MISC=yes -D PKG_EXTRA-DUMP=yes -D PKG_REAXFF=yes -D PKG_QEQ=yes -D PKG_MC=yes -D PKG_EAM=yes -D PKG_MEAM=yes -D PKG_RIGID=yes -D PKG_USER-CG-CMM=yes ../cmake
make -j$(nproc)
sudo make install

# libgfortran3 (use "-c bcc")
wget http://archive.ubuntu.com/ubuntu/pool/universe/g/gcc-6/gcc-6-base_6.4.0-17ubuntu1_amd64.deb
wget http://archive.ubuntu.com/ubuntu/pool/universe/g/gcc-6/libgfortran3_6.4.0-17ubuntu1_amd64.deb
sudo dpkg -i gcc-6-base_6.4.0-17ubuntu1_amd64.deb
sudo dpkg -i libgfortran3_6.4.0-17ubuntu1_amd64.deb

## remove methods (If you are concerned about other addictions)
# sudo apt -y purge libgfortran3
# sudo apt -y purge gcc-6-base
# sudo apt autoremove

# Usage
pyton3 postprocess_gaff_pbc.py
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
from ase.geometry import wrap_positions

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
# Lammps (GAFF) optimization
def gaff_pbc_optimize(fname, precursor_energy_per_atom):
    try:
        temp_dir = "gaff_pbc_temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        log_file = os.path.join(temp_dir, "system.log")

        # Step 1: Convert POSCAR to PDB
        mol = read('molecular_files/precursor.mol')
        precursor_pdb_path = os.path.join(temp_dir, "precursor.pdb")
        write(precursor_pdb_path, mol, format="proteindatabank")

        atoms = read(fname)
        atoms = atoms.copy()
        atoms.set_positions(atoms.get_positions(wrap=False))
        #crystal_pdb_path = os.path.join(temp_dir, "crystal.pdb") # The molecule was broken in PDB. I changed it to XYZ.
        #write(crystal_pdb_path, atoms, format="proteindatabank") # The molecule was broken in PDB. I changed it to XYZ.
        crystal_xyz_path = os.path.join(temp_dir, "crystal.xyz")
        write(crystal_xyz_path, atoms, format="xyz")

        moleculars = int(len(atoms)/len(mol))

        # Step 2: Run antechamber to generate mol2
        cmd = f"antechamber -i precursor.pdb -fi pdb -o precursor.mol2 -fo mol2 -at gaff -c gas -s 2"
        with open(log_file, "w") as log:
            subprocess.run(cmd, shell=True, cwd=temp_dir, stdout=log, stderr=subprocess.STDOUT)

        # Step 3: Run mol22lt.pl to generate .lt file
        with open(log_file, "w") as log:
            subprocess.run(f"mol22lt.pl < precursor.mol2 > precursor.lt", 
            shell=True, cwd=temp_dir, stdout=log, stderr=subprocess.STDOUT)

        # Step 4: Create system.lt
        system_lt_path = os.path.join(temp_dir, "system.lt")
        with open(system_lt_path, "w") as f:
            f.write("import \"gaff.lt\"\n")
            f.write("import \"precursor.lt\"\n")
            f.write(f"crystal = new MOL [{moleculars}]\n")

        # Step 5: Run moltemplate.sh with PDB
        #cmd = f"moltemplate.sh -atomstyle full -pdb crystal.pdb system.lt" # The molecule was broken in PDB. I changed it to XYZ.
        # Step 5: Run moltemplate.sh with XYZ
        cmd = f"moltemplate.sh -atomstyle full -xyz crystal.xyz system.lt"
        with open(log_file, "a") as log:
            subprocess.run(cmd, shell=True, cwd=temp_dir, stdout=log, stderr=subprocess.STDOUT)

        # Step 5.5: Copy LAMMPS input file
        shutil.copy("in_gaff_pbc.lmp", os.path.join(temp_dir, "in_gaff_pbc_temp.lmp"))
        
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
        
        # Step 6: Run LAMMPS
        #cmd = f"mpirun -np {cpu_count} lmp -in in_gaff_pbc_temp.lmp | tee ./../log.lammps"
        cmd = f"mpirun -np {cpu_count} lmp -in in_gaff_pbc_temp.lmp"
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
            print("Energy value not found in LAMMPS output.")
            energy = 0.0

        # Defining a file path
        md_npt_path = os.path.join(temp_dir, "md_npt.data")
        md_nvt_path = os.path.join(temp_dir, "md_nvt.data")
        
        # Loading the Structure
        if os.path.exists(md_npt_path):
            print("Using NPT-optimized structure.")
            atoms = read_lammps_data(md_npt_path, atom_style="full")
            atoms.wrap()
        elif os.path.exists(md_nvt_path):
            print("Warning: NPT structure not found. Using NVT-optimized structure.")
            atoms = read_lammps_data(md_nvt_path, atom_style="full")
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
        gaff_pbc_optimize(filepath, precursor_energy_per_atom=user_precursor_energy_per_atom)
        
        opt_fname_path  = f"optimized_structures_vasp/{fname}".replace("POSCAR", "OPT") + ".vasp"
        post_fname_path = f"valid_structures_postprocess/{fname}"
        shutil.copy(opt_fname_path, post_fname_path)
        print(f"------------------------------------------------------")
        continue
    except Exception:
        continue
#---------------------------------------------------------------------------------

print("Finished space group search and Lammps (GAFF) optimization.")
