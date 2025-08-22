#!/usr/bin/env python3

### Install libraries
# pip install ase==3.26.0 scipy==1.13.0 psutil==7.0.0 gpaw==25.7.0

### Siesta Installation
# sudo apt update
# sudo apt -y install cmake gfortran build-essential libopenmpi-dev libopenblas-dev 
# sudo apt -y install libhdf5-dev pkg-config libreadline-dev
# cd $HOME
# wget https://gitlab.com/siesta-project/siesta/-/releases/5.4.0/downloads/siesta-5.4.0.tar.gz
# tar xvf siesta-5.4.0.tar.gz
# cd siesta-5.4.0
# cmake -S . -B _build -DSIESTA_WITH_FLOOK="OFF"
# cmake --build _build -j 4
# sudo cmake --install _build
# echo 'export SIESTA_PP_PATH=$HOME/siesta-5.4.0/Pseudo/ThirdParty-Tools/ONCVPSP/nc-sr-05_pbe_standard_psml' >> ~/.bashrc  # path of pseudo-potentials
# source ~/.bashrc

### Pseudo-potantials
# http://www.icmab.es/siesta
# 
# use *.psml (>= Siesta 5.4.0)
# https://www.pseudo-dojo.org/
# cd $HOME/siesta-5.4.0/Pseudo/ThirdParty-Tools/ONCVPSP$
# (set) nc-sr-05_pbe_standard_psml.tgz
# tar xvf nc-sr-05_pbe_standard_psml.tgz
# 
# not use *.psml, then need to convert (< Siesta 5.4.0)
# psml2psf Si.psml > Si.psf

### Usage
# pyton3 mol2crystal_siesta.py

import os
import numpy as np
from ase.io import read, write
from ase.spacegroup import crystal
from scipy.spatial.distance import pdist
import shutil
import psutil

from ase.calculators.siesta import Siesta
from ase.filters import UnitCellFilter
from ase.optimize import BFGS
from ase.units import Ry

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

# Set CPU threads
cpu_count = psutil.cpu_count(logical=False)
os.environ["OMP_NUM_THREADS"] = str(cpu_count)

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
margin = 3.0 # near vdW radius
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

# Siesta optimization using *.psml pseudo-potentials
def siesta_optimize(fname, precursor_energy_per_atom):
    try:
        temp_dir = "siesta_temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        atoms = read(fname)

        # Siesta calculator
        calc = Siesta(
            label=os.path.join(temp_dir, 'siesta_calc'),
            xc='PBE',
            mesh_cutoff=200 * Ry,
            energy_shift=0.01 * Ry,
            basis_set='DZP',
            kpts=(1, 1, 1),
            fdf_arguments={
                'DM.MixingWeight': 0.1,
                'MaxSCFIterations': 100,
                'SolutionMethod': 'diagon',
                'WriteCoorXmol': True
            },
            pseudo_path=os.environ.get("SIESTA_PP_PATH", "./psf") # path of Pseudo-potentials
        )

        atoms.calc = calc

        ucf = UnitCellFilter(atoms)
        opt = BFGS(ucf,
            logfile=os.path.join(temp_dir, 'opt.log'),
            trajectory=os.path.join(temp_dir, 'opt.traj'))
        opt.run(fmax=0.05)

        # Save optimized structure
        opt_fname = fname.replace("valid_structures", "optimized_structures_vasp").replace("POSCAR", "OPT") + ".vasp"
        write(opt_fname, atoms, format='vasp')

        # Energy and density
        energy_value = atoms.get_potential_energy()
        num_atoms = len(atoms)
        energy_per_atom = energy_value / num_atoms
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
        import traceback
        traceback.print_exc()
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
        print(f"theta={theta:.2f}, phi={phi:.2f}, space group: 2 - 230")
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
                    siesta_optimize(fname, precursor_energy_per_atom=0.0)
                    print(f"Success: theta={i}, phi={j}, space group {sg}")
                    print(f"------------------------------------------------------")
            except Exception:
                continue

print("Finished space group search and GPAW optimization.")
