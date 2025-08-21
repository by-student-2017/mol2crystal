#!/usr/bin/env python3

# Install libraries
# pip install ase==3.22.1 scipy==1.13.0 psutil==7.0.0
# cd $HOME
# wget https://github.com/grimme-lab/xtb/releases/download/v6.7.1/xtb-6.7.1-linux-x86_64.tar.xz
# tar xvf xtb-6.7.1-linux-x86_64.tar.xz
# echo 'export PATH=$PATH:$HOME/xtb-dist/bin' >> ~/.bashrc

# Usage
# pyton3 mol2crystal_xtb.py

import os
import numpy as np
from ase.io import read, write
from ase.spacegroup import crystal
from scipy.spatial.distance import pdist
import subprocess
import shutil
import psutil
import uuid
import re

# xTB optimization function
import glob

import warnings
warnings.filterwarnings("ignore", message="scaled_positions .* are equivalent")

# Parallerization for xTB calculation
cpu_count = psutil.cpu_count(logical=False)
os.environ["OMP_NUM_THREADS"] = str(cpu_count)

print("# Read molecule")
mol = read('molecular_files/precursor.mol')
symbols = mol.get_chemical_symbols()
positions = mol.get_positions()

print("# Bounding box and cell")
min_pos = positions.min(axis=0)
max_pos = positions.max(axis=0)
extent = max_pos - min_pos
margin = 5.0
cellpar = list(extent + margin) + [90, 90, 90]
cell = np.array([[cellpar[0], 0, 0], [0, cellpar[1], 0], [0, 0, cellpar[2]]])
inv_cell = np.linalg.inv(cell)
print("Cell parameters (a, b, c, alpha, beta, gamma):", cellpar)
print("Cell matrix:\n", cell)

# Output directories
os.makedirs("valid_structures", exist_ok=True)
os.makedirs("optimized_structures_vasp", exist_ok=True)

# Overlap check
def has_overlap(atoms, threshold=0.9):
    dists = pdist(atoms.get_positions())
    return np.any(dists < threshold)

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


def xtb_optimize(fname):
    try:
        temp_dir = "xtb_temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        atoms = read(fname)
        original_cell = atoms.get_cell()
        
        # old version. read xyz file and run xtb
        #temp_xyz = os.path.join(temp_dir, "input.xyz")
        #write(temp_xyz, atoms, format='extxyz')
        #xtb_cmd = ["xtb", "input.xyz", "--opt", "--gfn", "1"]
        
        # Copy POSCAR to temp_dir
        temp_poscar = os.path.join(temp_dir, "POSCAR")
        write(temp_poscar, atoms, format='vasp')

        # run xtb
        xtb_cmd = ["xtb", "POSCAR", "--opt", "--gfn", "1"]
        with open(os.path.join(temp_dir, "xtb_output.log"), "w") as log_file:
            result = subprocess.run(xtb_cmd, cwd=temp_dir, stdout=log_file, stderr=subprocess.STDOUT)

        # save result regardless of convergence
        opt_xyz = os.path.join(temp_dir, "xtbopt.xyz")
        last_out = os.path.join(temp_dir, "xtblast.xyz")
        
        optimized = None
        source = None
        
        if os.path.exists(opt_xyz):
            optimized = read(opt_xyz, format='xyz')
            source = "xtbopt.xyz"
            print("Geometry optimization converged")
        elif os.path.exists(last_out):
            optimized = read(last_out, format='xyz')
            source = "xtblast.xyz"
            print("Note!!! Geometry optimization is not converged")
        else:
            print(f"[Error] No structure file found for {fname}")
            return
        
        optimized.set_cell(original_cell)
        optimized.set_pbc(True)
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
            num_atoms = optimized.get_number_of_atoms()
            energy_per_atom = energy_value / num_atoms * 27.2114
        
            # --- density calculation ---
            total_mass_amu = sum(optimized.get_masses())
            total_mass_g = total_mass_amu * 1.66053906660e-24
            volume = optimized.get_volume()
            volume_cm3 = volume * 1e-24
            density = total_mass_g / volume_cm3 if volume_cm3 > 0 else 0
        
            print(f"Final energy per atom: {energy_per_atom:.6f} [eV/atom]")
            print(f"Number of atoms: {num_atoms}")
            print(f"Volume: {volume:.6f} [A3]")
            print(f"Density: {density:.3f} [g/cm^3]")
        
            with open("optimized_structures_vasp/structure_vs_energy.txt", "a") as out:
                out.write(f"{fname} {energy_per_atom:.6f} {density:.3f} {num_atoms} {volume:.6f} \n")
        else:
            print("Energy value not found in xtbopt.log.")
        
    except Exception as e:
        print(f"Error optimizing {fname}: {e}")


print("# Generate valid structures")
valid_files = []
for i, theta in enumerate(np.linspace(0, np.pi/4, 3)):
    for j, phi in enumerate(np.linspace(0, np.pi/4, 3)):
        print("theta", theta, ", phi", phi, ", space group: 2 - 230")
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
                    xtb_optimize(fname)
                    print(f"Success: theta={i}, phi={j}, space group {sg}")
            except Exception:
                continue

# delete old files
#temp_dir = "xtb_temp"
#shutil.rmtree(temp_dir)

print("Finished space group search and xTB optimization.")
