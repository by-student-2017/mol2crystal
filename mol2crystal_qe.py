#!/usr/bin/env python3


#---------------------------------------------------------------------------------
# User setting parameters
#------------------------------------
user_margin = 1.70                   # >= vdW radius (H:1.20 - Cs:3.43)
user_margin_scale = 1.2              # Intermolecular arrangement: 1.2 - 1.5, Sparse placement (e.g., porous materials): 1.6 - 2.0
user_nmesh = 2                       # 0 - 45 degrees divided into nmesh
user_overlap_scale = 0.90            # threshold = scale * (r_i + r_j), covalent_radii: r_i and r_j
user_included_spacegroups = [34,230] # Include certain space groups from consideration  (high priority)
user_excluded_spacegroups = [1,2,70] # Exclude certain space groups from consideration  (low  priority)
user_skipping_spacegroups = 231      # Omit if space group >= user_skipping_spacegroups (low priority):
user_max_depth = 1                   # Neighborhood and top-level search. Number of recursions to find candidates.
user_skipping_n_molecules = 100      # Skip large molecular systems (>= user_skipping_n_molecules) (high priority)
user_primitive_cell_output = 1       # 0:No, 1:Yes (using spglib==2.6.0)
user_precursor_energy_per_atom = 0.0 # [eV] The reference energy (precursor alone) when calculating relative energy.
#---------------------------------------------------------------------------------
# Note(user_skipping_spacegroups): Since the space group ranges from 1 to 230, specifying 231 means that all are taken into consideration.


#---------------------------------------------------------------------------------
# --- Prepare environment and clean previous results ---
'''
# Install libraries
pip install ase==3.22.1 scipy==1.13.0 psutil==7.0.0
pip install pymsym==0.3.4
pip install spglib==2.6.0

# QE v.6.7MaX
sudo apt update
sudo apt -y install quantum-espresso

## Pseudo-potantials
# QE pslibrary: https://pseudopotentials.quantum-espresso.org/legacy_tables/ps-library
# TEHOS: https://theos-wiki.epfl.ch/Main/Pseudopotentials
# pslibrary: https://dalcorso.github.io/pslibrary/PP_list.html
# SSSP: https://www.materialscloud.org/discover/sssp/table/efficiency
# (Set the pseudopotential in the pseudodirectory.)

# Usage
pyton3 mol2crystal_qe.py
'''
#---------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
# --- Library imports and warning settings ---
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
from ase.geometry import cellpar_to_cell
from ase.neighborlist import NeighborList
#from ase.data import vdw_radii, atomic_numbers

# Point group analysis in space groups
import pymsym

# QE settings
from ase.calculators.espresso import Espresso, EspressoProfile
from ase.optimize import BFGS, LBFGS, FIRE

# Warning settings
import warnings
warnings.filterwarnings("ignore", message="scaled_positions .* are equivalent")
#---------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
# --- Clean old outputs and temporary folders ---
if (os.path.exists('valid_structures_old')):
    shutil.rmtree( 'valid_structures_old')   

if (os.path.exists('valid_structures')):
    os.rename(     'valid_structures','valid_structures_old')

if (os.path.exists('optimized_structures_vasp_old')):
    shutil.rmtree( 'optimized_structures_vasp_old')   

if (os.path.exists('optimized_structures_vasp')):
    os.rename(     'optimized_structures_vasp','optimized_structures_vasp_old')

dirs_to_remove = ['temp', 'gaff_pbc_temp', 'reaxff_temp', 'dftb_temp', 'xtb_temp', 'mopac_temp', 
  'qe_temp', 'abinit_temp', 'openmx_temp', 'gpaw_temp', 'siesta_temp', 'cp2k_temp', 'nwchem_temp', 'elk_temp']
for dir_name in dirs_to_remove:
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
#---------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
# --- Configure thread count for OpenMP/OpenMPI ---
cpu_count = psutil.cpu_count(logical=False)
os.environ["OMP_NUM_THREADS"] = '1'             # use OpenMPI
#os.environ["OMP_NUM_THREADS"] = str(cpu_count) # use OpenMP 
#---------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
# --- Load and center molecule, set margin and rotation parameters ---
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
#margin = 1.70 # >= vdW radius (H:1.20 - Cs:3.43)
#margin = margin * 1.2 # Intermolecular arrangement: 1.2 - 1.5, Sparse placement (e.g., porous materials): 1.6 - 2.0
margin = user_margin
margin = margin * user_margin_scale
print(f"Space around the molecule",margin, "[A]")

print("# Rotation angle setting")
#nmesh = 3 # 0 - 45 degrees divided into nmesh
nmesh = user_nmesh
print(f"0 - 45 degrees divided into",nmesh)

# Output directories
os.makedirs("valid_structures", exist_ok=True)
os.makedirs("optimized_structures_vasp", exist_ok=True)
#---------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
# --- Check for atomic overlap ---
# Old version (Simple method: This is simple but not bad.)
'''
def has_overlap(atoms, min_threshold=0.1, max_threshold=0.93):
    dists = pdist(atoms.get_positions())
    return np.any((dists > min_threshold) & (dists < max_threshold))
'''
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
'''
# New version 1: More detailed checks than the Simple version. Order(N^2) method.
def has_overlap(atoms, covalent_radii, scale):
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
'''
# New version 2: Faster than New version 1. Order(N) methods (linked-cell method)
def has_overlap_neighborlist(atoms, covalent_radii, scale):
    symbols = atoms.get_chemical_symbols()
    radii = [covalent_radii.get(sym, 0.7) * scale for sym in symbols]
    cutoffs = [r * 2 for r in radii]  # NeighborList expects diameter

    nl = NeighborList(cutoffs=cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)

    for i in range(len(atoms)):
        indices, offsets = nl.get_neighbors(i)
        for j, offset in zip(indices, offsets):
            dist = atoms.get_distance(i, j, mic=True)
            r_i = covalent_radii.get(symbols[i], 0.7)
            r_j = covalent_radii.get(symbols[j], 0.7)
            threshold = scale * (r_i + r_j)
            if symbols[i] == "H" and symbols[j] == "H":
                threshold = scale * 1.50
            if dist < threshold:
                return True
    return False
#---------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
# --- Rotate molecule ---
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
#---------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
# QE optimization
def qe_optimize(fname, precursor_energy_per_atom):
    try:
        temp_dir = "qe_temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        atoms = read(fname)

        # Get current directory
        cwd = os.getcwd()
        
        # Directory containing UPF files
        pseudo_dir = os.path.join(cwd, "pseudo")
        
        elements = set(atoms.get_chemical_symbols())
        pseudo_dict = {}
        for elem in elements:
            for pseudo_filename in os.listdir(pseudo_dir):
                if pseudo_filename.startswith(elem) and pseudo_filename.endswith(".UPF"):
                    pseudo_dict[elem] = pseudo_filename
                    break

        # List to store extracted ecutwfc values
        max_ecutwfc = 0
        for elem in elements:
            upf_file = pseudo_dict[elem]
            upf_path = os.path.join(pseudo_dir, upf_file)
            with open(upf_path, "r") as f:
                content = f.read()
                match_wfc = re.search(r"cutoff for wavefunctions:\s*([\d.]+)", content)
                match_rho = re.search(r"cutoff for charge density:\s*([\d.]+)", content)
                if match_wfc and match_rho:
                    ecutwfc_local = float(match_wfc.group(1))
                    ecutrho_local = float(match_rho.group(1))
                    if ecutwfc_local > max_ecutwfc:
                        max_ecutwfc = ecutwfc_local
                        ecutrho = ecutrho_local
                    break
        # Fallback in case no values were found
        if max_ecutwfc == 0:
            max_ecutwfc = 40
            ecutrho = 4 * max_ecutwfc

        input_data = {
            'control': {
                'calculation': 'scf',
                'restart_mode': 'from_scratch',
                'outdir': './',
                'disk_io': 'low',
                'tstress': True,
                'tprnfor': True,
                'etot_conv_thr': 1.0e-4*len(atoms),
                'forc_conv_thr': 1.0e-3*len(atoms),
            },
            'system': {
                'ecutwfc': max_ecutwfc,
                'ecutrho': ecutrho,
                'vdw_corr': 'DFT-D', # DFT-D, DFT-D3, MBD, XDM, (TS)
            },
            'electrons': {
                'conv_thr': 1.0e-3/13.6058*len(atoms),
            },
            'ions': {
                'ion_dynamics': 'bfgs',
            },
            'cell': {
                'cell_dynamics': 'bfgs',
                'cell_dofree': 'all',
                'press': 1.0e-3,
                #'press_conv_thr': 0.5,
            }
        }
        
        profile = EspressoProfile(
            command=f'mpirun -n {cpu_count} /usr/bin/pw.x',
            pseudo_dir=pseudo_dir,
        )
        
        calc = Espresso(
            profile=profile,
            pseudopotentials=pseudo_dict,
            input_data=input_data,
        )
        
        os.chdir(temp_dir)
        try:
            #atoms.set_calculator(calc)
            atoms.calc = calc
            opt = LBFGS(atoms)
            #opt = FIRE(atoms)
            opt.run(fmax=0.5)
        finally:
            os.chdir(cwd)

        '''
        print("Number of atoms:", len(atoms))
        print("Atomic positions:", atoms.get_positions())
        print("Cell parameters:", atoms.get_cell())
        print("Path:", opt_fname)
        '''

        # Preservation of structure after optimization
        opt_fname = fname.replace("valid_structures", "optimized_structures_vasp").replace("POSCAR", "OPT") + ".vasp"
        write(opt_fname, atoms, format='vasp')

        # Energy and density calculations
        energy_per_atom = atoms.get_potential_energy() / len(atoms)
        relative_energy_per_atom = energy_per_atom - precursor_energy_per_atom

        total_mass_amu = sum(atoms.get_masses())
        total_mass_g = total_mass_amu * 1.66053906660e-24
        volume = atoms.get_volume()
        volume_cm3 = volume * 1e-24
        density = total_mass_g / volume_cm3 if volume_cm3 > 0 else 0

        print(f"Final energy per atom: {energy_per_atom:.6f} [eV/atom]")
        print(f"Final relative energy per atom: {relative_energy_per_atom:.6f} [eV/atom]")
        print(f"Number of atoms: {len(atoms)}")
        print(f"Volume: {volume:.6f} [A3]")
        print(f"Density: {density:.3f} [g/cm^3]")

        with open("structure_vs_energy.txt", "a") as out:
            out.write(f"{opt_fname} {relative_energy_per_atom:.6f} {energy_per_atom:.6f} {density:.3f} {len(atoms)} {volume:.6f}\n")

    except Exception as e:
        print(f"Error optimizing {fname}: {e}")
#---------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
# --- Reference energy from original molecule ---
with open("structure_vs_energy.txt", "w") as f:
    print("# POSCAR file, Relative Energy [eV/atom], Total Energy [eV/atom], Density [g/cm^3], Number of atoms, Volume [A^3]", file=f)
#---------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
# --- Adjust unit cell parameters based on space group symmetry ---
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


#---------------------------------------------------------------------------------
# --- Adjust unit cell parameters based on space group symmetry ---

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


#---------------------------------------------------------------------------------
# --- Analyze point group and derive candidate space groups ---
print(f"------------------------------------------------------")
print(f"# Point group symmetry for 'precursor.mol'")
symbols = mol.get_chemical_symbols()
mol_atomic_numbers = mol.get_atomic_numbers()
positions = mol.get_positions()

# Get point group
pg = pymsym.get_point_group(mol_atomic_numbers, positions)
print(f"point group: {pg}")

# Get Symmetry Number
sn = pymsym.get_symmetry_number(mol_atomic_numbers, positions)
print(f"symmetry number: {sn}")

print("------------------------------------------------------")
print("# Applicable space groups (based on point group symmetry)")

# Groups and corresponding space groups (with international numbers)
point_group_to_space_groups = {
     "C1": list(range(1, 2)),     # P1
     "Ci": list(range(2, 3)),     # P-1
     "C2": list(range(3, 6)),     # P2, P21, C2
     "Cs": list(range(6, 10)),    # Pm, Pc, Cm, Cc
    "C2h": list(range(10, 16)),   # P2/m, P21/m, C2/m, C2/c, P2/c, P21/c
     "D2": list(range(16, 25)),   # P222, P212121, etc.
    "C2v": list(range(25, 47)),   # Pmm2, Pmc21, etc.
    "D2h": list(range(47, 75)),   # Pmmm, Pnma, etc.
     "C4": list(range(75, 81)),   # P4, P41, etc.
     "S4": list(range(81, 83)),   # P-4, P41
    "C4h": list(range(83, 89)),   # P4/m, P42/m, etc.
     "D4": list(range(89, 99)),   # P422, P4212, etc.
    "C4v": list(range(99, 111)),  # P4mm, P42mc, etc.
    "D2d": list(range(111, 123)), # P-42m, P-421m, etc.
    "D4h": list(range(123, 143)), # P4/mmm, P4/nmm, etc.
     "C3": list(range(143, 147)), # P3, P31, P32
    "C3i": list(range(147, 149)), # R-3, P-3
     "D3": list(range(149, 156)), # P312, P321, etc.
    "C3v": list(range(156, 162)), # P3m1, P31m, etc.
    "D3d": list(range(162, 168)), # R-3m, P-3m1, etc.
     "C6": list(range(168, 174)), # P6, P61, etc.
    "C3h": [174],                 # P-6
    "C6h": list(range(175, 177)), # P6/m, P62/m
     "D6": list(range(177, 183)), # P622, P6122, etc.
    "C6v": list(range(183, 187)), # P6mm, P6cc, etc.
    "D3h": list(range(187, 191)), # P-6m2, P-6c2, etc.
    "D6h": list(range(191, 195)), # P6/mmm, P6/mcc, etc.
      "T": list(range(195, 200)), # P23, F23, etc.
     "Th": list(range(200, 207)), # P213, Pa3, etc.
      "O": list(range(207, 215)), # P432, F432, etc.
     "Td": list(range(215, 221)), # P-43m, F-43m, etc.
     "Oh": list(range(221, 231))  # Pm-3m, Fm-3m, etc.
}

# Dictionary defining strict inclusion relationships between point groups.
# Each key is a point group, and its value is a list of point groups that are strictly included (i.e., subgroups).
# The list includes the group itself and all of its subgroups in descending symmetry.
related_point_groups_strict = {
     "C1": ["C1"],                       # Identity only (no symmetry)
     "Ci": ["Ci", "C1"],                 # Inversion symmetry
     "Cs": ["Cs", "C1"],                 # Mirror plane symmetry
     "C2": ["C2", "Ci", "Cs", "C1"],     # Two-fold rotation axis
    "C2h": ["C2h", "C2"],                # C2 + horizontal mirror plane
    "C2v": ["C2v", "C2", "Cs"],          # C2 + vertical mirror planes
     "D2": ["D2", "C2h", "C2v"],         # Three perpendicular C2 axes
    "D2h": ["D2h", "D2"],                # D2 + inversion + mirror planes
     "C4": ["C4", "C2"],                 # Four-fold rotation axis
     "S4": ["S4", "C2", "Ci"],           # Four-fold improper rotation
    "C4h": ["C4h", "C4", "C2h"],         # C4 + horizontal mirror plane
    "C4v": ["C4v", "C4", "C2v"],         # C4 + vertical mirror planes
     "D4": ["D4", "D2", "C4"],           # D2 + C4 axis
    "D4h": ["D4h", "D4", "C4h", "D2h"],  # D4 + mirror planes + inversion
    "D2d": ["D2d", "D2", "S4"],          # D2 + diagonal mirror planes
     "C3": ["C3", "C1"],                 # Three-fold rotation axis
    "C3i": ["C3i", "C3", "Ci"],          # C3 + inversion
    "C3v": ["C3v", "C3", "Cs"],          # C3 + vertical mirror planes
    "C3h": ["C3h", "C3", "Cs"],          # C3 + horizontal mirror plane
     "D3": ["D3", "C3"],                 # Three C2 axes perpendicular to C3
    "D3d": ["D3d", "D3", "C3i"],         # D3 + mirror planes + inversion
    "D3h": ["D3h", "D3", "C3h"],         # D3 + horizontal mirror plane
     "C6": ["C6", "C3"],                 # Six-fold rotation axis
    "C6h": ["C6h", "C6", "C3i"],         # C6 + horizontal mirror plane
    "C6v": ["C6v", "C6", "C3v"],         # C6 + vertical mirror planes
     "D6": ["D6", "C6", "D3"],           # D3 + C6 axis
    "D6h": ["D6h", "D6", "C6h", "D3d"],  # D6 + mirror planes + inversion
      "T": ["T", "D2"],                  # Tetrahedral rotation symmetry
     "Th": ["Th", "T", "D2h"],           # Tetrahedral + inversion
      "O": ["O", "T"],                   # Octahedral rotation symmetry
     "Td": ["Td", "T"],                  # Tetrahedral + mirror planes
     "Oh": ["Oh", "O", "Th"]             # Full octahedral symmetry
}

# Dictionary defining physical inclusion relationships between point groups.
# Each key is a point group, and its value is a list of physically related higher-symmetry point groups (supergroups).
# These represent one-level physical symmetry extensions, not strict group-theoretical subgroups.
# "Nearby" groups are also included when they are structurally or physically similar,
# even if not direct supergroups in the strict group-theoretical sense.
related_point_groups_physical = {
     "C1": ["Ci", "Cs", "C2"],           # C1 can be extended to inversion, mirror, or two-fold rotation
     "Ci": ["C2h", "D2h"],               # Inversion symmetry can be extended to C2h or full orthorhombic D2h
     "Cs": ["C2v", "D2d"],               # Mirror symmetry can be extended to vertical mirror systems or diagonal D2d
     "C2": ["C2h", "C2v", "D2"],         # Two-fold rotation can be extended to mirror or dihedral systems
    "C2h": ["D2h", "C4h"],               # C2h can be extended to full orthorhombic or tetragonal with horizontal mirror
    "C2v": ["D2", "D2h", "C4v", "D4h"],  # C2v can be extended to dihedral or tetragonal systems
     "D2": ["D2h", "D4"],                # D2 can be extended to full orthorhombic or tetragonal dihedral
    "D2h": ["D4h"],                      # D2h can be extended to full tetragonal symmetry
     "C4": ["C4h", "C4v"],               # C4 can be extended to horizontal or vertical mirror systems
     "S4": ["D2d", "D4h"],               # Improper rotation can be extended to diagonal or full tetragonal
    "C4h": ["D4h"],                      # C4h can be extended to full tetragonal symmetry
    "C4v": ["D4h", "Td", "D2d", "C2v"],  # C4v can be extended to full tetragonal symmetry (+ "D2d", "C2v")
     "D4": ["D4h"],                      # D4 can be extended to full tetragonal symmetry
    "D4h": ["Oh"],                       # D4h is already a full tetragonal group
    "D2d": ["D4h", "C4v", "S4", "Cs"],   # D2d can be extended to full tetragonal symmetry (+ "C4v", "S4", "Cs")
     "C3": ["C3i", "C3v", "C3h", "T", "C2", "C4"],  # C3 can be extended to inversion, vertical or horizontal mirror systems (+ "C2", "C4")
    "C3i": ["D3d", "C6h"],               # C3i can be extended to dihedral or hexagonal systems
    "C3v": ["D3h", "C6v", "Td", "C2v", "Cs"], # C3v can be extended to dihedral or hexagonal systems (+ "C2v", "Cs")
    "C3h": ["D3h"],                      # C3h can be extended to full dihedral symmetry
     "D3": ["D3d", "D3h", "T", "C3v"],   # D3 can be extended to full dihedral systems (+ "C3v")
    "D3d": ["D6h"],                      # D3d can be extended to full hexagonal symmetry
    "D3h": ["D6h"],                      # D3h can be extended to full hexagonal symmetry
     "C6": ["C6h", "C6v"],               # C6 can be extended to horizontal or vertical mirror systems
    "C6h": ["D6h"],                      # C6h can be extended to full hexagonal symmetry
    "C6v": ["D6h", "C3v", "C6h"],        # C6v can be extended to full hexagonal symmetry (+ "C3v", "C6h")
     "D6": ["D6h"],                      # D6 can be extended to full hexagonal symmetry
    "D6h": [],                           # D6h is already a full hexagonal group
      "T": ["Th", "O", "Td"],            # Tetrahedral rotation can be extended to inversion, octahedral, or mirror systems
     "Th": ["Oh"],                       # Tetrahedral + inversion can be extended to full octahedral symmetry
      "O": ["Oh"],                       # Octahedral rotation can be extended to full octahedral symmetry
     "Td": ["Oh"],                       # Tetrahedral + mirror can be extended to full octahedral symmetry
     "Oh": []                            # Oh is the highest cubic symmetry group
}

# Recursive function to get all subgroups (strict inclusion)
def get_all_subgroups(group, relation_dict):
    subgroups = set()
    def recurse(g):
        if g in subgroups:
            return
        subgroups.add(g)
        for sg in relation_dict.get(g, []):
            recurse(sg)
    recurse(group)
    return subgroups

# Recursive function to expand physical supergroups up to a given depth
def expand_physical_supergroups(base_dict, max_depth):
    if max_depth == 0:
        return {}
    expanded_dict = {}
    for group in base_dict:
        visited = set()
        current_level = set(base_dict.get(group, []))
        all_supergroups = set(current_level)
        for _ in range(max_depth - 1):
            next_level = set()
            for g in current_level:
                next_level.update(base_dict.get(g, []))
            next_level -= all_supergroups
            all_supergroups.update(next_level)
            current_level = next_level
        expanded_dict[group] = sorted(all_supergroups)
    return expanded_dict

# Step 1: Get all strictly included subgroups of the input point group
strict_groups = get_all_subgroups(pg, related_point_groups_strict)

# Step 2: Collect space groups corresponding to the strict subgroups
strict_sgs = set()
for g in strict_groups:
    strict_sgs.update(point_group_to_space_groups.get(g, []))

# Step 3: Expand physical supergroups of the input point group up to the specified depth
physical_supergroups = expand_physical_supergroups(related_point_groups_physical, max_depth=user_max_depth)
supergroups = physical_supergroups.get(pg, [])

# Step 4 & 5: Collect space groups directly corresponding to supergroups and their subgroups
expanded_sgs = set()
direct_supergroup_sgs = set()

for sg in supergroups:
    # Step 4: Direct space groups of supergroups
    direct_supergroup_sgs.update(point_group_to_space_groups.get(sg, []))
    
    # Step 5: Subgroups of supergroups
    subgroups = get_all_subgroups(sg, related_point_groups_strict)
    for g in subgroups:
        expanded_sgs.update(point_group_to_space_groups.get(g, []))

# Step 6: Combine space groups from strict subgroups and expanded supergroups
combined_sgs = sorted(strict_sgs.union(expanded_sgs))

# Output results
print(f"Strictly matched space groups (strict subgroups of '{pg}'):")
print(sorted(strict_sgs))

print(f"\nSpace groups directly corresponding to physical supergroups of '{pg}' (up to {user_max_depth}-level):")
print(sorted(direct_supergroup_sgs))

print(f"\nCombined applicable space groups (strict subgroups of '{pg}' + subgroups of its physical supergroups):")
space_groups = combined_sgs
print(space_groups)

print("------------------------------------------------------")
print(f"# Space group filtering conditions (Forced User Settings)")
print(f"Include (high priority): {user_included_spacegroups}")
print(f"Exclude (low priority): {user_excluded_spacegroups}")
print(f"Omit if >= {user_skipping_spacegroups} (low priority)")
print("------------------------------------------------------")
#---------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
# --- Find primitive cell after generating crystal structure ---
def get_primitive_cell(atoms):
    lattice = atoms.cell
    positions = atoms.get_scaled_positions()
    numbers = atoms.get_atomic_numbers()
    cell = (lattice, positions, numbers)
    
    primitive = spglib.find_primitive(cell)
    if primitive is None:
        return atoms
    
    lattice, positions, numbers = primitive
    primitive_atoms = Atoms(numbers=numbers, cell=lattice, scaled_positions=positions, pbc=True)
    return primitive_atoms
#---------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
# --- Main ---
print(f"------------------------------------------------------")
print("# Generate valid structures")
valid_files = []
for i, theta in enumerate(np.linspace(0, np.pi/4, nmesh)):
    for j, phi in enumerate(np.linspace(0, np.pi/4, nmesh)):
        print(f"------------------------------------------------------")
        print("theta", theta, ", phi", phi, ", space group: 1 - 230")
        print("# Rotation and Bounding box and cell")
        
        # Rotate molecule first
        rotated_positions = rotate_molecule(rotated_positions, theta, phi)
        
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
        
        print("Cell parameters (a, b, c, alpha, beta, gamma):", cellpar)
        cell = cellpar_to_cell(cellpar)
        print("Cell matrix:\n", cell)
        print(f"------------------------------------------------------")
        
        # Loop through all space groups (1–230) to check applicability
        for sg in range(1, 231):
            if sg not in space_groups:
                #print(f"Skipping space group {sg} (incompatible with point group '{pg}')")
                continue
            # Space group filter (high symmetry/known problem exclusion)
            excluded_spacegroups = user_excluded_spacegroups
            if sg in excluded_spacegroups:
                print(f"Skipping space group {sg} (known issue or too symmetric for molecules)")
                print(f"------------------------------------------------------")
                continue
            if sg >= user_skipping_spacegroups:
                print(f"Skipping space group {sg} (too symmetric for molecular crystals)")
                print(f"------------------------------------------------------")
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
                elif n_molecules > user_skipping_n_molecules: # Exclude if there are too many molecules
                    print(f"Not adopted because too many molecules ({n_molecules:.2f}) in the unit cell.")
                #elif not has_overlap(atoms, min_threshold=0.1, max_threshold=0.93): # For Simple
                #elif not has_overlap(crystal_structure, covalent_radii, scale=0.90): # New version 1
                elif not has_overlap_neighborlist(crystal_structure, covalent_radii, scale=user_overlap_scale): # For New version 2
                    fname = f"valid_structures/POSCAR_theta_{i}_phi_{j}_sg_{sg}"
                    write(fname, crystal_structure, format='vasp')
                    valid_files.append(fname)
                    qe_optimize(fname, precursor_energy_per_atom=user_precursor_energy_per_atom)
                    print(f"Success: theta={i}, phi={j}, space group {sg}")
                else:
                    print("Not adopted because the interatomic distance is too close.")
                print(f"------------------------------------------------------")
                continue
            except Exception:
                continue
#---------------------------------------------------------------------------------

print("Finished space group search and QE optimization.")
