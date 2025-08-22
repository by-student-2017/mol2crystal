#!/usr/bin/env python3

# Install libraries
# pip install ase==3.22.1 scipy==1.13.0 psutil==7.0.0
# wget https://github.com/openmopac/mopac/releases/download/v23.1.2/mopac-23.1.2-linux.tar.gz
# tar xvf mopac-23.1.2-linux.tar.gz
# echo 'export PATH=$PATH:$HOME/mopac-23.1.2-linux/bin' >> ~/.bashrc
# source ~/.bashrc

# Usage
# pyton3 mol2crystal_mopac.py

import os
import glob
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

dirs_to_remove = ['temp', 'xtb_temp', 'dftb_temp', 'gpaw_temp']
for dir_name in dirs_to_remove:
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)

cpu_count = psutil.cpu_count(logical=False)
os.environ["OMP_NUM_THREADS"] = '1'

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
margin = 3.0
#cellpar = list(extent + margin) + [90, 90, 90]
#cell = np.array([[cellpar[0], 0, 0], [0, cellpar[1], 0], [0, 0, cellpar[2]]])
max_extent = extent.max() + margin
cellpar = [max_extent, max_extent, max_extent, 90, 90, 90]
cell = np.array([[max_extent, 0, 0],
                 [0, max_extent, 0],
                 [0, 0, max_extent]])
inv_cell = np.linalg.inv(cell)
print("Cell parameters (a, b, c, alpha, beta, gamma):", cellpar)
print("Cell matrix:\n", cell)

os.makedirs("valid_structures", exist_ok=True)
os.makedirs("optimized_structures_vasp", exist_ok=True)


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


import os
import shutil
import numpy as np
import re
import subprocess
from ase.io import read, write

def mopac_optimize(fname, precursor_energy_per_atom):
    try:
        temp_dir = "mopac_temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        atoms = read(fname)
        xyz_path = os.path.join(temp_dir, "input.xyz")
        write(xyz_path, atoms, format='xyz')

        # Create MOPAC input file
        mopac_input_path = os.path.join(temp_dir, "input.dat")
        with open(xyz_path, "r") as xyz_file:
            xyz_content = xyz_file.read()

        with open(mopac_input_path, "w") as f:
            f.write("PM7 XYZ SCFCRT=1.D-6 MOZYME EF GNORM=1.0 DEBUG STRESS PRESSURE=1.0E5\nNumber of atoms: ")
            f.write("PM7 XYZ SCFCRT=1.D-6 MOZYME\nNumber of atoms: ")
            f.write(xyz_content)
            cell = atoms.get_cell()
            f.write(f"Tv {cell[0][0]:22.15f} {cell[0][1]:22.15f} {cell[0][2]:22.15f}\n")
            f.write(f"Tv {cell[1][0]:22.15f} {cell[1][1]:22.15f} {cell[1][2]:22.15f}\n")
            f.write(f"Tv {cell[2][0]:22.15f} {cell[2][1]:22.15f} {cell[2][2]:22.15f}\n")

        # Run MOPAC
        log_path = os.path.join(temp_dir, "input.out")
        arc_path = os.path.join(temp_dir, "input.arc")
        with open(log_path, "w") as out:
            subprocess.run(["mpirun", "-np", str(cpu_count), "mopac", "input.dat"], cwd=temp_dir, stdout=out, stderr=subprocess.STDOUT)

        # Parse output for final energy
        energy_value = None
        if os.path.exists(arc_path):
            with open(log_path, "r") as f:
                for line in f:
                    match = re.search(r"HEAT OF FORMATION\s+=\s+(-?\d+\.\d+)", line)
                    if match:
                        energy_value = float(match.group(1))
                        break

        # Save optimized structure
        if os.path.exists(arc_path):
            optimized = read(arc_path)
            opt_fname = fname.replace("valid_structures", "optimized_structures_vasp").replace("POSCAR", "OPT") + ".vasp"
            write(opt_fname, optimized, format='vasp')

            if energy_value is not None:
                num_atoms = len(optimized)
                energy_per_atom = energy_value / num_atoms
                relative_energy_per_atom = energy_per_atom - precursor_energy_per_atom

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
                print(f"------------------------------------------------------")

                with open("structure_vs_energy.txt", "a") as out:
                    out.write(f"{fname} {relative_energy_per_atom:.6f} {energy_per_atom:.6f} {density:.3f} {num_atoms} {volume:.6f}\n")
            else:
                print("Energy value not found in MOPAC output.")
        else:
            print(f"[Error] No optimized structure file found for {fname}")

    except Exception as e:
        print(f"Error optimizing {fname}: {e}")

# Reference energy from original molecule
with open("structure_vs_energy.txt", "w") as f:
    print("# POSCAR file, Relative Energy [eV/atom], Total Energy [eV/atom], Density [g/cm^3], Number of atoms, Volume [A^3]", file=f)

nmesh = 3 # 0 - 45 degree devided nmesh
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
                    mopac_optimize(fname, precursor_energy_per_atom=0.0)
                    print(f"Success: theta={i}, phi={j}, space group {sg}")
            except Exception:
                continue

print("Finished space group search and DFTB+ optimization.")
