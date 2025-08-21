#!/usr/bin/env python3

# Install libraries
# pip install ase==3.22.1 scipy==1.13.0 psutil==7.0.0
#sudo apt update
#sudo apt install openbabel
#sudo apt install libopenbabel-dev

# Usage
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
extent[extent < 1.0] = 1.0  # avoid zero-length cell
margin = 15.0
cellpar = list(extent + margin) + [90, 90, 90]
cell = np.array([[cellpar[0], 0, 0], [0, cellpar[1], 0], [0, 0, cellpar[2]]])
inv_cell = np.linalg.inv(cell)
print("Cell parameters (a, b, c, alpha, beta, gamma):", cellpar)
print("Cell matrix:\n", cell)

os.makedirs("valid_structures", exist_ok=True)
#os.makedirs("optimized_structures_vasp", exist_ok=True)


def has_overlap(atoms, threshold=0.85):
    dists = pdist(atoms.get_positions())
    return np.any(dists < threshold)


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
    xtb_cmd = ["obenergy", "-ff", "GAFF", temp_xyz]
    try:
        result = subprocess.run(xtb_cmd, capture_output=True, text=True, check=True)
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

    num_atoms = len(mol) # or num_atoms = mol.get_global_number_of_atoms()
    energy_per_atom = energy * 0.0103636 / num_atoms if num_atoms > 0 else 0.0
    relative_energy_per_atom = energy_per_atom - precursor_energy_per_atom

    total_mass_amu = sum(atoms.get_masses())
    total_mass_g = total_mass_amu * 1.66053906660e-24
    volume = atoms.get_volume()
    volume_cm3 = volume * 1e-24
    density_val = total_mass_g / volume_cm3 if volume_cm3 > 0 else 0

    print(f"Final energy per atom: {energy_per_atom:.6f} [eV/atom]")
    print(f"Final relative energy per atom: {relative_energy_per_atom:.6f} [eV/atom]")
    print(f"Number of atoms: {num_atoms}")
    print(f"Volume: {volume:.6f} [A3]")
    print(f"Density: {density_val:.3f} [g/cm^3]")
    print(f"------------------------------------------------------")

    with open("structure_vs_energy.txt", "a") as out:
        out.write(f"{fname} {relative_energy_per_atom:.6f} {energy_per_atom:.6f} {density_val:.3f} {num_atoms} {volume:.6f}\n")

# Reference energy from original molecule
temp_mol = os.path.join('molecular_files/precursor.mol')
write("precursor.xyz", mol, format="xyz")
xtb_cmd = ["obenergy", "-ff", "GAFF", "precursor.xyz"]
result = subprocess.run(xtb_cmd, capture_output=True, text=True, check=True)
output = result.stdout
match = re.search(r"TOTAL ENERGY =\s*(-?\d+\.\d+)", output)
if match:
    precursor_energy_per_atom = float(match.group(1)) / len(mol)
with open("structure_vs_energy.txt", "w") as f:
    print("# POSCAR file, Relative Energy [eV/atom], Total Energy [eV/atom], Density [g/cm^3], Number of atoms, Volume [A^3]", file=f)

nmesh = 3
print("# Generate valid structures")
valid_files = []
for i, theta in enumerate(np.linspace(0, np.pi/4, nmesh)):
    for j, phi in enumerate(np.linspace(0, np.pi/4, nmesh)):
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
                    obenergy_calc(fname, precursor_energy_per_atom)
                    print(f"Success: theta={i}, phi={j}, space group {sg}")
            except Exception:
                continue

print("Finished checking space groups. valid structures written.")
