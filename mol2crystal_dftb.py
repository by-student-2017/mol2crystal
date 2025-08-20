#!/usr/bin/env python3

# Install libraries
# pip install ase==3.22.1 scipy==1.13.0 psutil==7.0.0
# cd $HOME
# wget https://github.com/dftbplus/dftbplus/releases/download/24.1/dftbplus-24.1.x86_64-linux.tar.xz
# tar -xvf dftbplus-24.1.x86_64-linux.tar.xz
# echo 'export PATH=$PATH:$HOME/dftbplus-24.1.x86_64-linux/bin' >> ~/.bashrc

# Usage
# pyton3 mol2crystal_dftb.py

import os
import glob
import shutil
import numpy as np
from ase.io import read, write
from ase.spacegroup import crystal
from scipy.spatial.distance import pdist
import subprocess
import psutil

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
margin = 5.0
cellpar = list(extent + margin) + [90, 90, 90]
cell = np.array([[cellpar[0], 0, 0], [0, cellpar[1], 0], [0, 0, cellpar[2]]])
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


def dftb_optimize(fname):
    try:
        temp_dir = "dftb_temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        atoms = read(fname)
        vasp_path = os.path.join(temp_dir, "POSCAR")
        write(vasp_path, atoms, format='vasp')

        # Copy input file
        shutil.copy("dftb_in.hsd", os.path.join(temp_dir, "dftb_in.hsd"))

        # Run DFTB+
        log_path = os.path.join(temp_dir, "dftb_out.log")
        err_path = os.path.join(temp_dir, "dftb_err.log")
        with open(log_path, "w") as out, open(err_path, "w") as err:
            subprocess.run(["dftb+"], cwd=temp_dir, stdout=out, stderr=err)
        
        # Save result regardless of convergence
        geo_end = os.path.join(temp_dir, "geo_end.gen")
        geom_out = os.path.join(temp_dir, "geom.out.gen")
        
        if os.path.exists(geo_end):
            optimized = read(geo_end)
            source = "geo_end.gen"
            print("Geometry optimization converged")
        elif os.path.exists(geom_out):
            optimized = read(geom_out)
            source = "geom.out.gen"
            print("Note!!! Geometry optimization is not converged")
        else:
            print(f"[Error] No structure file found for {fname}")
            return
        
        opt_fname = fname.replace("valid_structures", "optimized_structures_vasp").replace("POSCAR", "OPT") + ".vasp"
        write(opt_fname, optimized, format='vasp')

    except Exception as e:
        print(f"Error optimizing {fname}: {e}")

# Keep temp_dir for debugging
# shutil.rmtree(dftb_temp)

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
                    dftb_optimize(fname)
                    print(f"Success: theta={i}, phi={j}, space group {sg}")
            except Exception:
                continue

print("Finished space group search and DFTB+ optimization.")
