#!/usr/bin/env python3

# Install libraries
# pip install ase==3.22.1 scipy==1.13.0 psutil==7.0.0
# sudo apt -y install cp2k

# Usage
# pyton3 mol2crystal_cp2k.py

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

dirs_to_remove = ['temp', 'xtb_temp', 'dftb_temp', 'cp2k_temp', 'gpaw_temp']
for dir_name in dirs_to_remove:
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)

cpu_count = psutil.cpu_count(logical=False)
os.environ["OMP_NUM_THREADS"] = '1'

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

# CP2K optimization using GPW
def cp2k_optimize(fname, precursor_energy_per_atom):
    try:
        temp_dir = "cp2k_temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        atoms = read(fname)
        structure_xyz = os.path.join(temp_dir, "structure.xyz")
        write(structure_xyz, atoms, format='xyz')

        atoms.translate(-atoms.get_center_of_mass())
        cell = atoms.get_cell()
        a_line = f"      A  {cell[0][0]:.9f}  {cell[0][1]:.9f}  {cell[0][2]:.9f}\n"
        b_line = f"      B  {cell[1][0]:.9f}  {cell[1][1]:.9f}  {cell[1][2]:.9f}\n"
        c_line = f"      C  {cell[2][0]:.9f}  {cell[2][1]:.9f}  {cell[2][2]:.9f}\n"

        # Pseudopotential map of transition metals (with q values ​​as an example)
        transition_potentials = {
            "Sc": "GTH-PBE-q11", "Ti": "GTH-PBE-q12", "V": "GTH-PBE-q13",
            "Cr": "GTH-PBE-q14", "Mn": "GTH-PBE-q15", "Fe": "GTH-PBE-q16",
            "Co": "GTH-PBE-q17", "Ni": "GTH-PBE-q18", "Cu": "GTH-PBE-q19",
            "Zn": "GTH-PBE-q20"
        }

        kind_sections = ""
        unique_elements = sorted(set(atoms.get_chemical_symbols()))
        for elem in unique_elements:
            basis_set = "DZVP-MOLOPT-GTH"
            potential = transition_potentials.get(elem, "GTH-PBE")

            # If not present in BASIS_MOLOPT file, fallback to SR-GTH
            basis_file_path = os.path.join("data", "BASIS_MOLOPT")
            if os.path.exists(basis_file_path):
                with open(basis_file_path, "r") as bf:
                    if basis_set not in bf.read():
                        basis_set = "DZVP-MOLOPT-SR-GTH"

            kind_sections += f"    &KIND {elem}\n"
            kind_sections += f"      BASIS_SET {basis_set}\n"
            kind_sections += f"      POTENTIAL {potential}\n"
            kind_sections += f"    &END KIND\n"

        input_inp_path = "cp2k.inp"
        output_inp_path = os.path.join(temp_dir, "cp2k.inp")

        if os.path.exists(input_inp_path):
            with open(input_inp_path, "r") as f:
                lines = f.readlines()

            with open(output_inp_path, "w") as f:
                for line in lines:
                    if line.strip().startswith("A "):
                        f.write(a_line)
                    elif line.strip().startswith("B "):
                        f.write(b_line)
                    elif line.strip().startswith("C "):
                        f.write(c_line)
                    elif line.strip() == "&END TOPOLOGY":
                        f.write(line)
                        f.write(kind_sections)
                    else:
                        f.write(line)
            print("Modified cp2k.inp and saved to cp2k_temp.")
        else:
            print("cp2k.inp does not exist.")
            return

        if not shutil.which("cp2k.popt"):
            print("cp2k.popt not found in PATH.")
            return

        subprocess.run(["mpirun", "-np", str(cpu_count), "cp2k.popt", "-i", "cp2k.inp", "-o", "cp2k.out"], cwd=temp_dir, check=True)

        out_path = os.path.join(temp_dir, "cp2k.out")
        energy_value = None
        with open(out_path, "r") as f:
            for line in reversed(f.readlines()):
                if "ENERGY| Total FORCE_EVAL" in line:
                    parts = line.split()
                    energy_value = float(parts[-1])
                    break

        optimized_xyz = os.path.join(temp_dir, "cp2k_calc-pos-1.xyz")
        if os.path.exists(optimized_xyz):
            atoms = read(optimized_xyz)  # Replace with the optimized structure
        else:
            print("Warning: Optimized structure file not found. Using original structure.")

        opt_fname = fname.replace("valid_structures", "optimized_structures_vasp").replace("POSCAR", "OPT") + ".vasp"
        write(opt_fname, atoms, format='vasp')

        if energy_value is not None:
            num_atoms = len(atoms)
            energy_per_atom = energy_value / num_atoms * 27.2114
            relative_energy_per_atom = energy_per_atom - precursor_energy_per_atom
            
            # --- density calculation ---
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
                    cp2k_optimize(fname, precursor_energy_per_atom=0.0)
                    print(f"Success: theta={i}, phi={j}, space group {sg}")
                    print(f"------------------------------------------------------")
            except Exception:
                continue

print("Finished space group search and DFTB+ optimization.")
