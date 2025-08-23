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

print("# The margin of one molecule setting")
margin = 2.0 # >= vdW radius (1.55 - 3.43)
margin = margin * 1.5 # Intermolecular arrangement: 1.2 - 1.5, Sparse placement (e.g., porous materials): 1.6 - 2.0
print(f"Space around the molecule",margin, "[A]")

print("# Rotation angle setting")
nmesh = 7 # 0 - 90 degrees divided into nmesh
print(f"0 - 90 degrees divided into",nmesh)

# Output directories
os.makedirs("valid_structures", exist_ok=True)
os.makedirs("optimized_structures_vasp", exist_ok=True)

# Check for atomic overlap
# Old version (Simple method: This is simple but not bad.)
'''
def has_overlap(atoms, min_threshold=0.1, max_threshold=0.93):
    dists = pdist(atoms.get_positions())
    return np.any((dists > min_threshold) & (dists < max_threshold))
'''
# New version
atomic_radii = {
     "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96,  "B": 0.84,  "C": 0.76,  "N": 0.71,  "O": 0.66,  "F": 0.57, "Ne": 0.58,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11,  "P": 1.07,  "S": 1.05, "Cl": 1.02, "Ar": 1.06,
     "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60,  "V": 1.53, "Cr": 1.39, "Mn": 1.39, "Fe": 1.32, "Co": 1.26, "Ni": 1.24,
    "Cu": 1.32, "Zn": 1.22, "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20, "Kr": 1.16,
    "Rb": 2.20, "Sr": 1.95,  "Y": 1.90, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54, "Tc": 1.47, "Ru": 1.46, "Rh": 1.42, "Pd": 1.39,
    "Ag": 1.45, "Cd": 1.44, "In": 1.42, "Sn": 1.39, "Sb": 1.39, "Te": 1.38,  "I": 1.39, "Xe": 1.40
}
def has_overlap(atoms, atomic_radii, scale=0.90):
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()
    
    natoms = len(positions)
    for i in range(natoms):
        for j in range(i + 1, natoms):
            # Shortest distance (considering periodic boundaries)
            dist = atoms.get_distance(i, j, mic=True)
            r_i = atomic_radii.get(symbols[i], 0.7)
            r_j = atomic_radii.get(symbols[j], 0.7)
            threshold = scale * (r_i + r_j)
            if symbols[i] == "H" and symbols[j] == "H":
                threshold = scale * 2.0 # >= (r_i + r_j): Shortest H-H distance in an H2O molecule (1.51)
            if dist < threshold:
                return True
    return False

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
valid_files = []
for i, theta in enumerate(np.linspace(0, np.pi/2, nmesh)):
    for j, phi in enumerate(np.linspace(0, np.pi/2, nmesh)):
        print(f"------------------------------------------------------")
        print("theta", theta, ", phi", phi, ", space group: 2 - 230")
        print("# Rotation and Bounding box and cell")
        rotated_positions = rotate_molecule(positions, theta, phi)
        
        # Reevaluate bounding box
        min_pos = rotated_positions.min(axis=0)
        max_pos = rotated_positions.max(axis=0)
        extent = max_pos - min_pos
        
        cellpar = list(extent + 2 * margin) + [90, 90, 90]
        cell = np.array([[cellpar[0], 0, 0],
                         [0, cellpar[1], 0],
                         [0, 0, cellpar[2]]])
        inv_cell = np.linalg.inv(cell)
        
        shifted_positions = rotated_positions - rotated_positions.min(axis=0)
        fractional_positions = np.dot(shifted_positions, inv_cell)
        
        print("Cell parameters (a, b, c, alpha, beta, gamma):", cellpar)
        print("Cell matrix:\n", cell)
        print(f"------------------------------------------------------")
        
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
                else:
                    print("Space group under investigation:",sg)
                    print("Not adopted because the interatomic distance is too close.")
                    print(f"------------------------------------------------------")
            except Exception:
                continue

print("Finished checking space groups. valid structures written.")
