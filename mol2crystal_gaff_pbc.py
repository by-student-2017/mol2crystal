#!/usr/bin/env python3

### Install libraries
# pip install ase==3.22.1 scipy==1.13.0 psutil==7.0.0

### moltemplate + antechamber + mol22lt.pl
# sudo apt update
# sudo apt -y install dos2unix python3-pip libgfortran5 liblapack3
# wget https://github.com/makoto-yoneya/makoto-yoneya.github.io/raw/master/LAMMPS-organics/install_moltemplate.sh
# sh install_moltemplate.sh
# wget https://github.com/makoto-yoneya/makoto-yoneya.github.io/raw/master/LAMMPS-organics/install_WSLmisc.sh
# sh install_WSLmisc.sh

### lammps (stable_22Jul2025)
# cd $HOME
# sudo apt -y install cmake gfortran gcc libopenmpi-dev
# git clone -b stable https://github.com/lammps/lammps.git
# cd lammps
# mkdir build && cd build
# cmake -D BUILD_MPI=yes -D BUILD_SHARED_LIBS=no -D PKG_KSPACE=yes -D PKG_MOLECULE=yes -D PKG_EXTRA-MOLECULE=yes -D PKG_USER-MISC=yes -D PKG_EXTRA-DUMP=yes -D PKG_REAXFF=yes -D PKG_QEQ=yes -D PKG_MC=yes -D PKG_EAM=yes -D PKG_RIGID=yes -D PKG_USER-CG-CMM=yes ../cmake
# make -j$(nproc)
# sudo make install

### libgfortran3 (use "-c bcc")
# wget http://archive.ubuntu.com/ubuntu/pool/universe/g/gcc-6/gcc-6-base_6.4.0-17ubuntu1_amd64.deb
# wget http://archive.ubuntu.com/ubuntu/pool/universe/g/gcc-6/libgfortran3_6.4.0-17ubuntu1_amd64.deb
# sudo dpkg -i gcc-6-base_6.4.0-17ubuntu1_amd64.deb
# sudo dpkg -i libgfortran3_6.4.0-17ubuntu1_amd64.deb
#
### remove methods (If you are concerned about other addictions)
# sudo apt -y purge libgfortran3
# sudo apt -y purge gcc-6-base
# sudo apt autoremove

### Usage
# pyton3 mol2crystal_gaff_pbc.py

import os
import shutil
import numpy as np
from ase.io import read, write
from ase.spacegroup import crystal
from scipy.spatial.distance import pdist
import subprocess
import psutil
import re

from ase.io.lammpsdata import read_lammps_data
from ase.geometry import wrap_positions

import warnings
warnings.filterwarnings("ignore", message="scaled_positions .* are equivalent")

if (os.path.exists('valid_structures_old')):
    shutil.rmtree( 'valid_structures_old')   

if (os.path.exists('valid_structures')):
    os.rename(     'valid_structures','valid_structures_old')

if (os.path.exists('optimized_structures_vasp_old')):
    shutil.rmtree( 'optimized_structures_vasp_old')   

if (os.path.exists('optimized_structures_vasp')):
    os.rename(     'optimized_structures_vasp','optimized_structures_vasp_old')

dirs_to_remove = ['temp', 'cp2k_temp', 'dftb_temp', 'gaff_temp', 'gaff_pbc_temp', 'gpaw_temp', 'mopac_temp', 'siesta_temp', 'xtb_temp']
for dir_name in dirs_to_remove:
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)

cpu_count = psutil.cpu_count(logical=False)
os.environ["OMP_NUM_THREADS"] = '1'             # OpenMPI
#os.environ["OMP_NUM_THREADS"] = str(cpu_count) # OpenMP 

print(f"------------------------------------------------------")
print("# Read molecule")
mol = read('molecular_files/precursor.mol')
symbols = mol.get_chemical_symbols()
positions = mol.get_positions()
com = mol.get_center_of_mass()
mol.translate(-com)

print("# Bounding box and cell")
min_pos = positions.min(axis=0)
max_pos = positions.max(axis=0)
extent = max_pos - min_pos
extent[extent < 1.0] = 1.0  # avoid zero-length cell
margin = 6.0 # >= vdW radius
#cellpar = list(extent + margin) + [90, 90, 90]
#cell = np.array([[cellpar[0], 0, 0], [0, cellpar[1], 0], [0, 0, cellpar[2]]])
max_extent = extent.max() + 2 * margin  # Ensure margin on both sides
cellpar = [max_extent, max_extent, max_extent, 90, 90, 90]
cell = np.array([[max_extent, 0, 0],
                 [0, max_extent, 0],
                 [0, 0, max_extent]])
inv_cell = np.linalg.inv(cell)
mol.translate(0.5 * max_extent) # Move to the center of the cell
print("Cell parameters (a, b, c, alpha, beta, gamma):", cellpar)
print("Cell matrix:\n", cell)

# Output directories
os.makedirs("valid_structures", exist_ok=True)
os.makedirs("optimized_structures_vasp", exist_ok=True)

# Check for atomic overlap
def has_overlap(atoms, min_threshold=0.1, max_threshold=0.93):
    dists = pdist(atoms.get_positions())
    return np.any((dists > min_threshold) & (dists < max_threshold))

# Rotation
def rotate_molecule(positions, theta, phi):
    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])
    Ry = np.array([
        [np.cos(phi), 0, np.sin(phi)],
        [0,           1, 0],
        [-np.sin(phi), 0, np.cos(phi)]
    ])
    return positions @ Rz.T @ Ry.T

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
        crystal_pdb_path = os.path.join(temp_dir, "crystal.pdb")
        write(crystal_pdb_path, atoms, format="proteindatabank")

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

        # Step 5: Run moltemplate.sh
        cmd = f"moltemplate.sh -atomstyle full -pdb crystal.pdb system.lt"
        with open(log_file, "a") as log:
            subprocess.run(cmd, shell=True, cwd=temp_dir, stdout=log, stderr=subprocess.STDOUT)

        # Step 5.5: Copy LAMMPS input file
        shutil.copy("in_gaff_pbc.lmp", os.path.join(temp_dir, "in_gaff_pbc_temp.lmp"))

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

        optimized_xyz = os.path.join(temp_dir, "md.data")
        if os.path.exists(optimized_xyz):
            atoms = read_lammps_data(optimized_xyz, atom_style="full")
            atoms.wrap()
        else:
            print("Warning: Optimized structure file not found. Using original structure.")
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
            out.write(f"{fname} {relative_energy_per_atom:.6f} {energy_per_atom:.6f} {density:.3f} {num_atoms} {volume:.6f}\n")

    except Exception as e:
        print(f"Error optimizing {fname}: {e}")

# Reference energy from original molecule
with open("structure_vs_energy.txt", "w") as f:
    print("# POSCAR file, Relative Energy [eV/atom], Total Energy [eV/atom], Density [g/cm^3], Number of atoms, Volume [A^3]", file=f)

print(f"------------------------------------------------------")
print("# Generate valid structures")
nmesh = 3 # 0 - 45 degrees divided into nmesh
print(f"0 - 45 degrees divided into",nmesh)
print(f"------------------------------------------------------")
valid_files = []
for i, theta in enumerate(np.linspace(0, np.pi/4, nmesh)):
    for j, phi in enumerate(np.linspace(0, np.pi/4, nmesh)):
        print("theta", theta, ", phi", phi, ", space group: 2 - 230")
        print(f"------------------------------------------------------")
        rotated_positions = rotate_molecule(positions, theta, phi)
        shifted_positions = rotated_positions - rotated_positions.min(axis=0)
        fractional_positions = np.dot(shifted_positions, inv_cell)

        for sg in range(2, 231):
            try:
                crystal_structure = crystal(symbols=symbols,
                                            basis=fractional_positions,
                                            spacegroup=sg,
                                            cellpar=cellpar,
                                            pbc=True)
                if not has_overlap(crystal_structure):
                    fname = f"valid_structures/POSCAR_theta_{i}_phi_{j}_sg_{sg}"
                    write(fname, crystal_structure, format='vasp')
                    valid_files.append(fname)
                    gaff_pbc_optimize(fname, precursor_energy_per_atom=0.0)
                    print(f"Success: theta={i}, phi={j}, space group {sg}")
                    print(f"------------------------------------------------------")
            except Exception:
                continue

print("Finished checking space groups. valid structures written.")
