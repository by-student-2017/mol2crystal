#!/usr/bin/env python3

### Install libraries
# pip install ase==3.22.1 scipy==1.13.0 psutil==7.0.0

### OpenBable
# sudo apt update
# sudo apt install openbabel
# sudo apt install libopenbabel-dev

### Usage
# pyton3 mol2crystal_uff.py

import os
import shutil
import numpy as np
from ase.io import read, write
from ase.spacegroup import crystal
from scipy.spatial.distance import pdist
import subprocess
import psutil
import re

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
#os.environ["OMP_NUM_THREADS"] = '1'           # use OpenMPI
os.environ["OMP_NUM_THREADS"] = str(cpu_count) # use OpenMP 

print(f"------------------------------------------------------")
print("# Read molecule")
mol = read('molecular_files/precursor.mol')
symbols = mol.get_chemical_symbols()
positions = mol.get_positions()
#com = mol.get_center_of_mass()             # Center of mass (center of gravity)
com = np.mean(mol.get_positions(), axis=0)  # Volume center (geometric center)
mol.translate(-com)

print("# Bounding box and cell")
min_pos = positions.min(axis=0)
max_pos = positions.max(axis=0)
extent = max_pos - min_pos
extent[extent < 1.0] = 1.0  # avoid zero-length cell
margin = 2.0 # >= vdW radius (1.55 - 3.43)
margin = margin * 1.5 # Intermolecular arrangement: 1.2 - 1.5, Sparse placement (e.g., porous materials): 1.6 - 2.0
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

# OpenBabel (GAFF or UFF) calculation
def obenergy_calc(fname, precursor_energy_per_atom):
    temp_dir = "temp"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    atoms = read(fname)
    atoms.set_pbc(False)

    temp_xyz = os.path.join(temp_dir, "input.xyz")
    write(temp_xyz, atoms, format='xyz')

    # Run Open Babel to calculate energy: GAFF, UFF, MMFF94, MMFF94s, Ghemical
    obenergy_cmd = ["obenergy", "-ff", "GAFF", temp_xyz]
    try:
        result = subprocess.run(obenergy_cmd, capture_output=True, text=True, check=True)
        output = result.stdout
        match = re.search(r"TOTAL ENERGY =\s*(-?\d+\.\d+)", output)
        if match:
            energy = float(match.group(1))
        else:
            print("Energy value not found in Open Babel output.")
            energy = 0.0
    except subprocess.CalledProcessError as e:
        print("Error running Open Babel:", e)
        energy = 0.0

    num_atoms = len(atoms) # or num_atoms = atoms.get_global_number_of_atoms()
    energy_per_atom = energy * 0.0103636 / num_atoms if num_atoms > 0 else 0.0
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

# Reference energy from original molecule
temp_mol = os.path.join('molecular_files/precursor.mol')
write("precursor.xyz", mol, format="xyz")
obenergy_cmd = ["obenergy", "-ff", "GAFF", "precursor.xyz"]
result = subprocess.run(obenergy_cmd, capture_output=True, text=True, check=True)
output = result.stdout
match = re.search(r"TOTAL ENERGY =\s*(-?\d+\.\d+)", output)
if match:
    precursor_energy_per_atom = float(match.group(1)) / len(mol)
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
                    obenergy_calc(fname, precursor_energy_per_atom)
                    print(f"Success: theta={i}, phi={j}, space group {sg}")
                    print(f"------------------------------------------------------")
            except Exception:
                continue

print("Finished checking space groups. valid structures written.")
