#!/usr/bin/env python3

### Install libraries + GPAW ver. 25.7.0
# pip install ase==3.26.0 scipy==1.13.0 psutil==7.0.0 gpaw==25.7.0
# pip install "numpy<2.0"

### Usage
# pyton3 mol2crystal_gpaw.py

import os
import numpy as np
from ase.io import read, write
from ase.spacegroup import crystal
from scipy.spatial.distance import pdist
import shutil
import psutil

from gpaw import GPAW, PW
from ase.filters import UnitCellFilter
from ase.optimize import LBFGS

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

# Set CPU threads
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
vdw_radii = {
     "H": 1.20, "He": 1.40, "Li": 1.82, "Be": 1.53,  "B": 1.92,  "C": 1.70,  "N": 1.55,  "O": 1.52,  "F": 1.47, "Ne": 1.54,
    "Na": 2.27, "Mg": 1.73, "Al": 1.84, "Si": 2.10,  "P": 1.80,  "S": 1.80, "Cl": 1.75, "Ar": 1.88,  "K": 2.75, "Ca": 2.31,
    "Sc": 2.11, "Ti": 2.00,  "V": 2.00, "Cr": 2.00, "Mn": 2.00, "Fe": 2.00, "Co": 2.00, "Ni": 1.63, "Cu": 1.40, "Zn": 1.39,
    "Ga": 1.87, "Ge": 2.11, "As": 1.85, "Se": 1.90, "Br": 1.85, "Kr": 2.02, "Rb": 3.03, "Sr": 2.49,  "Y": 2.00, "Zr": 2.00,
    "Nb": 2.00, "Mo": 2.00, "Tc": 2.00, "Ru": 2.00, "Rh": 2.00, "Pd": 1.63, "Ag": 1.72, "Cd": 1.58, "In": 1.93, "Sn": 2.17,
    "Sb": 2.06, "Te": 2.06,  "I": 1.98, "Xe": 2.16, "Cs": 3.43, "Ba": 2.68, "La": 2.00, "Ce": 2.00, "Pr": 2.00, "Nd": 2.00,
    "Pm": 2.00, "Sm": 2.00, "Eu": 2.00, "Gd": 2.00, "Tb": 2.00, "Dy": 2.00, "Ho": 2.00, "Er": 2.00, "Tm": 2.00, "Yb": 2.00,
    "Lu": 2.00, "Hf": 2.00, "Ta": 2.00,  "W": 2.00, "Re": 2.00, "Os": 2.00, "Ir": 2.00, "Pt": 1.75, "Au": 1.66, "Hg": 1.55,
    "Tl": 1.96, "Pb": 2.02, "Bi": 2.07, "Po": 2.00, "At": 2.00, "Rn": 2.20, "Fr": 2.00, "Ra": 2.00, "Ac": 2.00, "Th": 2.00,
    "Pa": 2.00,  "U": 1.96, "Np": 1.90, "Pu": 1.87, "XX": 2.00, "Am": 2.00, "Cm": 1.52, "Bm": 2.00
}
margin = 1.7 # >= vdW radius (H:1.20 - Cs:3.43)
margin = margin * 1.5 # Intermolecular arrangement: 1.2 - 1.5, Sparse placement (e.g., porous materials): 1.6 - 2.0
print(f"Space around the molecule",margin, "[A]")

print("# Rotation angle setting")
nmesh = 3 # 45 - 90 degrees divided into nmesh
print(f"45 - 90 degrees divided into",nmesh)

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
covalent_radii = {
     "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96,  "B": 0.84,  "C": 0.76,  "N": 0.71,  "O": 0.66,  "F": 0.57, "Ne": 0.58,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11,  "P": 1.07,  "S": 1.05, "Cl": 1.02, "Ar": 1.06,  "K": 2.03, "Ca": 1.76,
    "Sc": 1.70, "Ti": 1.60,  "V": 1.53, "Cr": 1.39, "Mn": 1.39, "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22,
    "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20, "Kr": 1.16, "Rb": 2.20, "Sr": 1.95,  "Y": 1.90, "Zr": 1.75,
    "Nb": 1.64, "Mo": 1.54, "Tc": 1.47, "Ru": 1.46, "Rh": 1.42, "Pd": 1.39, "Ag": 1.45, "Cd": 1.44, "In": 1.42, "Sn": 1.39,
    "Sb": 1.39, "Te": 1.38,  "I": 1.39, "Xe": 1.40, "Cs": 2.44, "Ba": 2.15, "La": 2.07, "Ce": 2.04, "Pr": 2.03, "Nd": 2.01,
    "Pm": 1.99, "Sm": 1.98, "Eu": 1.98, "Gd": 1.96, "Tb": 1.94, "Dy": 1.92, "Ho": 1.92, "Er": 1.89, "Tm": 1.90, "Yb": 1.87,
    "Lu": 1.87, "Hf": 1.75, "Ta": 1.70,  "W": 1.62, "Re": 1.51, "Os": 1.44, "Ir": 1.41, "Pt": 1.36, "Au": 1.36, "Hg": 1.32,
    "Tl": 1.45, "Pb": 1.46, "Bi": 1.48, "Po": 1.40, "At": 1.50, "Rn": 1.50, "Fr": 2.60, "Ra": 2.21, "Ac": 2.15, "Th": 2.06,
    "Pa": 2.00,  "U": 1.96, "Np": 1.90, "Pu": 1.87, "XX": 2.00, "Am": 1.39, "Cm": 0.66, "Bm": 1.39
}
def has_overlap(atoms, covalent_radii, scale=0.90):
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()
    
    natoms = len(positions)
    for i in range(natoms):
        for j in range(i + 1, natoms):
            # Shortest distance (considering periodic boundaries)
            dist = atoms.get_distance(i, j, mic=True)
            r_i = covalent_radii.get(symbols[i], 0.7)
            r_j = covalent_radii.get(symbols[j], 0.7)
            threshold = scale * (r_i + r_j)
            if symbols[i] == "H" and symbols[j] == "H":
                threshold = scale * 1.50 # >= (r_i + r_j): Shortest H-H distance in an H2O molecule (1.51)
            if dist < threshold:
                return True
    return False

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
def gpaw_optimize(fname, precursor_energy_per_atom):
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
        
        # --- Extract only the last energy value ---
        #energy_value = atoms.get_potential_energy()
        log_path = os.path.join(temp_dir, "gpaw_out.txt")
        energy_value = None
        
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                for line in reversed(f.readlines()):
                    match = re.search(r"iter:\s+\d+\s+\d{2}:\d{2}:\d{2}\s+(-?\d+\.\d+)", line)
                    if match:
                        energy_value = float(match.group(1))
                        break
        
        if energy_value is not None:
            num_atoms = len(atoms) # or num_atoms = atoms.get_global_number_of_atoms()
            energy_per_atom = energy_value / num_atoms * 27.2114
            relative_energy_per_atom = energy_per_atom - precursor_energy_per_atom
            
            # --- density calculation ---
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
            
            with open("structure_vs_energy.txt", "a") as out:
                out.write(f"{fname} {relative_energy_per_atom:.6f} {energy_per_atom:.6f} {density:.3f} {num_atoms} {volume:.6f}\n")
        else:
            print("Energy value not found in xtbopt.log.")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error optimizing {fname}: {e}")

# Reference energy from original molecule
with open("structure_vs_energy.txt", "w") as f:
    print("# POSCAR file, Relative Energy [eV/atom], Total Energy [eV/atom], Density [g/cm^3], Number of atoms, Volume [A^3]", file=f)

def adjust_cellpar_by_spacegroup(sg, cellpar):
    adjusted_cellpar = cellpar.copy()

    # Monoclinic (3–15)
    if sg in {4, 11, 12, 18}:
        adjusted_cellpar[1] *= 2
    elif sg in {5, 8, 12, 13}:
        adjusted_cellpar[0] *= 2 / np.sqrt(2)
        adjusted_cellpar[1] *= 2 / np.sqrt(2)
    elif sg in {7, 9, 11, 13, 15}:
        adjusted_cellpar[2] *= 2
    elif sg in {14}:
        adjusted_cellpar[1] *= 2 / np.sqrt(2)
        adjusted_cellpar[2] *= 2 / np.sqrt(2)

    # Orthorhombic (16–74)
    elif sg in {28, 51}: # a
        adjusted_cellpar[0] *= 2
    elif sg in {62, 74}: # b
        adjusted_cellpar[1] *= 2
    elif sg in {17, 20, 26, 27, 33, 36, 37, 45, 49, 60, 63, 66, 72}: # c
        adjusted_cellpar[2] *= 2
    elif sg in {67}: # a,b
        adjusted_cellpar[0] *= 2
        adjusted_cellpar[1] *= 2
    elif sg in {29, 53, 54}: # a,c
        adjusted_cellpar[0] *= 2
        adjusted_cellpar[2] *= 2
    elif sg in {39, 57}: # b,c
        adjusted_cellpar[1] *= 2
        adjusted_cellpar[2] *= 2
    elif sg in {24, 73}: # a,b,c
        adjusted_cellpar[0] *= 2
        adjusted_cellpar[1] *= 2
        adjusted_cellpar[2] *= 2
    elif sg in {18, 21, 32, 35, 50, 55, 59, 65}: # ab plane
        adjusted_cellpar[0] *= 2 / np.sqrt(2)
        adjusted_cellpar[1] *= 2 / np.sqrt(2)
    elif sg in {30, 38}: # bc plane
        adjusted_cellpar[1] *= 2 / np.sqrt(2)
        adjusted_cellpar[2] *= 2 / np.sqrt(2)
    elif sg in {31}: # ac plane
        adjusted_cellpar[0] *= 2 / np.sqrt(2)
        adjusted_cellpar[2] *= 2 / np.sqrt(2)
    elif sg in {40, 46, 52}:
        adjusted_cellpar[1] *= 2
        adjusted_cellpar[1] *= 2 / np.sqrt(2)
        adjusted_cellpar[2] *= 2 / np.sqrt(2)
    elif sg in {19, 22, 41, 42, 56, 61, 64, 68, 69}: # FCC
        adjusted_cellpar[0] *= 2 / np.sqrt(2)
        adjusted_cellpar[1] *= 2 / np.sqrt(2)
        adjusted_cellpar[2] *= 2 / np.sqrt(2)
    elif sg in {43, 70}: # Diamond
        adjusted_cellpar[0] *= 4 / np.sqrt(2)
        adjusted_cellpar[1] *= 4 / np.sqrt(2)
        adjusted_cellpar[2] *= 4 / np.sqrt(2)
    elif sg in {23, 34, 44, 48, 58, 71}: # BCC
        adjusted_cellpar[0] *= 2 / np.sqrt(3)
        adjusted_cellpar[1] *= 2 / np.sqrt(3)
        adjusted_cellpar[2] *= 2 / np.sqrt(3)

    # Tetragonal (75–142)
    elif sg in {80}: # b
        adjusted_cellpar[1] *= 2
    elif sg in {77, 84, 93, 101, 103, 105, 108, 112, 120, 124, 131, 132, 135}: # c
        adjusted_cellpar[2] *= 2
    elif sg in {76, 78, 91, 95}: # c
        adjusted_cellpar[2] *= 4
    elif sg in {79, 82, 86, 87, 88, 94, 97, 98, 102, 104, 107, 109, 114, 118, 119, 121, 122, 126, 128, 134, 136, 137, 139, 141}: # BCC
        adjusted_cellpar[0] *= 2 / np.sqrt(3)
        adjusted_cellpar[1] *= 2 / np.sqrt(3)
        adjusted_cellpar[2] *= 2 / np.sqrt(3)
    elif sg in {85, 90, 100, 113, 117, 125, 127, 129}: # ab plane
        adjusted_cellpar[0] *= 2 / np.sqrt(2)
        adjusted_cellpar[1] *= 2 / np.sqrt(2)
    elif sg in {92, 96, 106, 109, 130, 133, 138, 140, 142}: # c, ab plane
        adjusted_cellpar[2] *= 2
        adjusted_cellpar[0] *= 2 / np.sqrt(2)
        adjusted_cellpar[1] *= 2 / np.sqrt(2)
    elif sg in {123, 139}:
        factor = 1.5
        adjusted_cellpar[:3] = [x * factor for x in cellpar[:3]]

    # Trigonal (143–167)
    elif sg in {143, 147, 149, 150, 156, 157, 162, 164}:
        adjusted_cellpar[5] = 120     # gamma = 120 degree
    elif sg in {158, 159, 163, 165}:
        adjusted_cellpar[2] *= 2
        adjusted_cellpar[5] = 120     # gamma = 120 degree
    elif sg in {144, 145, 151, 152, 153, 154}:
        adjusted_cellpar[2] *= 4
        adjusted_cellpar[5] = 120     # gamma = 120 degree
    elif sg in {146, 148, 155, 160, 161, 166, 167}:
        adjusted_cellpar[0] *= 3      # = 4 * sqrt(3)/2 * sqrt(3)/2
        adjusted_cellpar[1] *= 3      # = 4 * sqrt(3)/2 * sqrt(3)/2
        adjusted_cellpar[2] *= 4      
        adjusted_cellpar[5] = 120     # gamma = 120 degree

    # Hexagonal (168-194)
    elif sg in {168, 174, 175, 177, 183, 187, 189, 191}:
        adjusted_cellpar[5] = 120     # gamma = 120 degree
    elif sg in {173, 176, 182, 184, 185, 186, 188, 190, 192, 193, 194}:
        adjusted_cellpar[2] *= 2
        adjusted_cellpar[5] = 120     # gamma = 120 degree
    elif sg in {171, 172, 180, 181}:
        adjusted_cellpar[2] *= 4
        adjusted_cellpar[5] = 120     # gamma = 120 degree
    elif sg in {169, 170, 178, 179}:
        adjusted_cellpar[2] *= 6
        adjusted_cellpar[5] = 120     # gamma = 120 degree
    elif sg in {146, 148, 155, 160, 161, 166, 167}:
        adjusted_cellpar[0] *= 3      # = 4 * sqrt(3)/2 * sqrt(3)/2
        adjusted_cellpar[1] *= 3      # = 4 * sqrt(3)/2 * sqrt(3)/2
        adjusted_cellpar[2] *= 4      
        adjusted_cellpar[5] = 120     # gamma = 120 degree

    # Cubic (195–230)
    elif sg in {196, 198, 202, 205, 209, 212, 216, 225}: # FCC
        adjusted_cellpar[0] *= 2 / np.sqrt(2)
        adjusted_cellpar[1] *= 2 / np.sqrt(2)
        adjusted_cellpar[2] *= 2 / np.sqrt(2)
    elif sg in {203, 210, 212, 213, 214, 220, 227, 228, 230}: # Diamond
        adjusted_cellpar[0] *= 4 / np.sqrt(2)
        adjusted_cellpar[1] *= 4 / np.sqrt(2)
        adjusted_cellpar[2] *= 4 / np.sqrt(2)
    elif sg in {197, 201, 204, 208, 211, 217, 218, 222, 223, 224, 229}: # BCC
        adjusted_cellpar[0] *= 2 / np.sqrt(3)
        adjusted_cellpar[1] *= 2 / np.sqrt(3)
        adjusted_cellpar[2] *= 2 / np.sqrt(3)
    elif sg in {199, 206, 219, 226}:
        adjusted_cellpar[0] *= 2
        adjusted_cellpar[1] *= 2
        adjusted_cellpar[2] *= 2

    return adjusted_cellpar

#---------------------------------------------------------------------------------
# Placed at the geometric center
center = positions.mean(axis=0)
centered_positions = positions - center

# Find the atom farthest from the geometric center
distances = np.linalg.norm(centered_positions, axis=1)
farthest_index = np.argmax(distances)
principal_axis = centered_positions[farthest_index]

# Define the target direction (theta=45, phi=45)
theta = np.pi / 4
phi = np.pi / 4
target_direction = np.array([
    np.sin(theta) * np.cos(phi),
    np.sin(theta) * np.sin(phi),
    np.cos(theta)
])

# rotate principal_axis to target_direction
def rotation_matrix_from_vectors(vec1, vec2):
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    cross = np.cross(vec1, vec2)
    dot = np.dot(vec1, vec2)
    if np.isclose(dot, -1.0):
        return -np.eye(3)
    elif np.isclose(dot, 1.0):
        return np.eye(3)
    skew = np.array([
        [0, -cross[2], cross[1]],
        [cross[2], 0, -cross[0]],
        [-cross[1], cross[0], 0]
    ])
    R = np.eye(3) + skew + np.dot(skew, skew) * ((1 - dot) / (np.linalg.norm(cross) ** 2))
    return R

rotation_matrix = rotation_matrix_from_vectors(principal_axis, target_direction)
rotated_positions = centered_positions.dot(rotation_matrix.T)
#---------------------------------------------------------------------------------

print(f"------------------------------------------------------")
print("# Generate valid structures")
valid_files = []
for i, theta in enumerate(np.linspace(0, np.pi/2, nmesh)):
    for j, phi in enumerate(np.linspace(0, np.pi/2, nmesh)):
        print(f"------------------------------------------------------")
        print("theta", theta, ", phi", phi, ", space group: 2 - 230")
        print("# Rotation and Bounding box and cell")
        
        # Rotate molecule first
        rotated_positions = rotate_molecule(positions, theta, phi)
        
        # Reevaluate bounding box separately for x, y, z
        min_x, min_y, min_z = rotated_positions.min(axis=0)
        max_x, max_y, max_z = rotated_positions.max(axis=0)
        extent_x = max_x - min_x
        extent_y = max_y - min_y
        extent_z = max_z - min_z
        
        # Define cell parameters with margin
        cell_x = extent_x + margin
        cell_y = extent_y + margin
        cell_z = extent_z + margin
        cellpar = [cell_x, cell_y, cell_z, 90, 90, 90]
        
        for sg in range(1, 231):
            if sg >= 195:
                print(f"Skipping space group {sg} (too symmetric for molecular crystals)")
                continue
            
            # In structures with planes, intersections occur, so users can exclude them.
            # Users can also exclude unnecessarily large structures.
            excluded_spacegroups = [16, 27, 36, 43, 45, 49, 50, 54, 70, 72]
            if sg in excluded_spacegroups:
                print(f"Skipping space group {sg} (known issue or too symmetric for molecules)")
                continue
            
            try:
                adjusted_cellpar = adjust_cellpar_by_spacegroup(sg, cellpar)
                print("Cell parameters (a, b, c, alpha, beta, gamma):", adjusted_cellpar)
                
                # Reconstructed cell matrix
                cell = adjusted_cellpar
                print("Cell matrix:\n", cell)
                
                cell = cellpar_to_cell(adjusted_cellpar)
                inv_cell = np.linalg.inv(cell)

                # Shift the numerator
                min_corner = rotated_positions.min(axis=0)
                shift_vector = -min_corner + margin/2
                shifted_positions = rotated_positions + shift_vector

                # Transform the molecule based on adjusted_cellpar
                fractional_positions = np.dot(shifted_positions, inv_cell)

                crystal_structure = crystal(symbols=symbols,
                                            basis=fractional_positions,
                                            spacegroup=sg,
                                            cellpar=adjusted_cellpar,
                                            pbc=True)
                atoms = crystal_structure
                n_molecules = len(atoms) / len(mol) # # Estimation of the number of molecules
                print(f"Space group under investigation: {sg}")
                print(f"number of molecules ({n_molecules:.2f}) in the unit cell.")
                if len(atoms) == len(mol):
                    print(f"Not adopted because single molecule only.")
                elif n_molecules > 100: # Exclude if there are too many molecules (e.g., more than 100 molecules)
                    print(f"Not adopted because too many molecules ({n_molecules:.2f}) in the unit cell.")
                elif not has_overlap(crystal_structure, covalent_radii):
                    fname = f"valid_structures/POSCAR_theta_{i}_phi_{j}_sg_{sg}"
                    write(fname, crystal_structure, format='vasp')
                    valid_files.append(fname)
                    gpaw_optimize(fname, precursor_energy_per_atom=0.0)
                    print(f"Success: theta={i}, phi={j}, space group {sg}")
                else:
                    print("Not adopted because the interatomic distance is too close.")
                print(f"------------------------------------------------------")
                continue
            except Exception:
                continue

print("Finished space group search and GPAW optimization.")
