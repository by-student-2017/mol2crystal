#!/usr/bin/env python3

# Install libraries
# pip install ase==3.26.0 scipy==1.13.0 psutil==7.0.0 gpaw==25.7.0
# pip install "numpy<2.0"

# Usage
# pyton3 mol2crystal_gpaw.py

import os
import numpy as np
from ase.io import read, write
from ase.spacegroup import crystal
from scipy.spatial.distance import pdist
import shutil
import psutil

from gpaw import GPAW, PW
from ase.constraints import UnitCellFilter
from ase.optimize import LBFGS

import warnings
warnings.filterwarnings("ignore", message="scaled_positions .* are equivalent")

# Set CPU threads
cpu_count = psutil.cpu_count(logical=False)
os.environ["OMP_NUM_THREADS"] = str(cpu_count)

# Read molecule
print("# Read molecule")
mol = read('molecular_files/precursor.mol')
symbols = mol.get_chemical_symbols()
positions = mol.get_positions()

# Define cell
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

# Create output directories
os.makedirs("valid_structures", exist_ok=True)
os.makedirs("optimized_structures_vasp", exist_ok=True)

# Check for atomic overlap
def has_overlap(atoms, threshold=0.85):
    dists = pdist(atoms.get_positions())
    return np.any(dists < threshold)

# Rotate molecule
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

# GPAW optimization
def gpaw_optimize(fname):
    try:
        temp_dir = "gpaw_temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        atoms = read(fname)

        # Output GPAW calculation to temporary directory
        calc = GPAW(xc='PBE', kpts=(1, 1, 1), mode=PW(400),
                    txt=os.path.join(temp_dir, 'gpaw_out.txt'))
        atoms.calc = calc

        ucf = UnitCellFilter(atoms)
        opt = LBFGS(ucf,
                    logfile=os.path.join(temp_dir, 'opt.log'),
                    trajectory=os.path.join(temp_dir, 'opt.traj'))
        opt.run(fmax=0.05)

        # Save the final structure
        opt_fname = fname.replace("valid_structures", "optimized_structures_vasp").replace("POSCAR", "OPT") + ".vasp"
        write(opt_fname, atoms, format='vasp')
        print(f"[GPAW] Saved: {opt_fname}")

        # Delete the temporary directory
        shutil.rmtree(temp_dir)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error optimizing {fname}: {e}")


# Generate structures
print("# Generate valid structures")
valid_files = []
for i, theta in enumerate(np.linspace(0, np.pi/4, 3)):
    for j, phi in enumerate(np.linspace(0, np.pi/4, 3)):
        print(f"theta={theta:.2f}, phi={phi:.2f}, space group: 2 - 230")
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
                    gpaw_optimize(fname)
                    print(f"Success: theta={i}, phi={j}, space group {sg}")
            except Exception:
                continue

print("Finished space group search and GPAW optimization.")
