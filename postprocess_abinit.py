#!/usr/bin/env python3


#---------------------------------------------------------------------------------
# User setting parameters
#------------------------------------
user_precursor_energy_per_atom = 0.0 # [eV] The reference energy (precursor alone) when calculating relative energy.
#---------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
# --- Prepare environment and clean previous results ---
'''
# Install libraries
pip install ase==3.22.1 scipy==1.13.0 psutil==7.0.0
pip install pymsym==0.3.4
pip install spglib==2.6.0

# Abinit v.9.6.2
sudo apt update
sudo apt -y install abinit

# Usage
pyton3 mol2crystal_abinit.py
'''
#---------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
# --- Library imports and warning settings ---
import os
import glob
import shutil
import numpy as np
from ase.io import read, write
#from ase.spacegroup import crystal
#from scipy.spatial.distance import pdist
import subprocess
import psutil
import re
from ase.geometry import cellpar_to_cell
#from ase.neighborlist import NeighborList
#from ase.data import vdw_radii, atomic_numbers

# Fro Abinit settings
from ase.data import atomic_numbers
from ase.geometry import cell_to_cellpar
from ase import Atoms

# Warning settings
import warnings
warnings.filterwarnings("ignore", message="scaled_positions .* are equivalent")
#---------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
# --- Clean old outputs and temporary folders ---
if (os.path.exists('valid_structures_postprocess_old')):
    shutil.rmtree( 'valid_structures_postprocess_old')

if (os.path.exists('valid_structures_postprocess')):
    os.rename(     'valid_structures_postprocess','valid_structures_postprocess_old')

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
# --- Configure thread count for OpenMP/OpenMPI --
cpu_count = psutil.cpu_count(logical=False)
os.environ["OMP_NUM_THREADS"] = '1'             # use OpenMPI
#os.environ["OMP_NUM_THREADS"] = str(cpu_count) # use OpenMP 
#---------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
# Output directories
os.makedirs("optimized_structures_vasp", exist_ok=True)
os.makedirs("valid_structures_postprocess", exist_ok=True)
#---------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
# Abinit
def compress_typat_simple(typat):
    count_dict = {}
    for t in typat:
        count_dict[t] = count_dict.get(t, 0) + 1
    return " ".join(f"{count_dict[t]}*{t}" for t in sorted(count_dict))

# Abinit optimization
def abinit_optimize(fname, precursor_energy_per_atom):
    try:
        temp_dir = "abinit_temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        atoms = read(fname)
        cwd = os.getcwd()
        pseudo_dir = os.path.join(cwd, "pseudo")

        symbols = atoms.get_chemical_symbols()
        unique_elements = sorted(set(symbols), key=symbols.index)
        pseudo_dict = {}

        # Find and copy pseudopotential files to temp_dir
        for elem in unique_elements:
            for pseudo_filename in os.listdir(pseudo_dir):
                if pseudo_filename.startswith(elem) and pseudo_filename.endswith(".xml"):
                    pseudo_dict[elem] = pseudo_filename
                    src_path = os.path.join(pseudo_dir, pseudo_filename)
                    dst_path = os.path.join(temp_dir, pseudo_filename)
                    shutil.copy(src_path, dst_path)
                    break

        # acell in Bohr
        cell = atoms.get_cell()
        cell_lengths = cell_to_cellpar(cell)[:3]
        acell_bohr = [length * 1.8897259886 for length in cell_lengths]

        # rprim as normalized vectors
        rprim = [cell[i] / np.linalg.norm(cell[i]) for i in range(3)]
        
        ntypat = len(unique_elements)
        znucl = [atomic_numbers[el] for el in unique_elements]
        natom = len(atoms)
        typat = [unique_elements.index(sym) + 1 for sym in symbols]
        positions = atoms.get_scaled_positions()
        
        volume_angstrom3 = atoms.get_volume()
        volume_bohr3 = volume_angstrom3 * (1.8897259886 ** 3)

        with open("opt.abi", "r") as f:
            lines = f.readlines()

        with open(os.path.join(temp_dir, "opt_temp.abi"), "w") as f:
            for line in lines:
                # Separation: Keywords and comments
                parts = line.split("#", 1)
                keyword = parts[0].strip()
                comment = f"  # {parts[1].strip()}" if len(parts) > 1 else ""
                if keyword == "pseudos":
                    indent_length = 11
                    max_line_length = 60
                    separator = ","
                    separator_with_newline = ",\n"
                    f.write('pseudos = "')
                    first_pseudo = pseudo_dict[unique_elements[0]]
                    f.write(first_pseudo)
                    line_length = len(first_pseudo)
                    for i, elem in enumerate(unique_elements[1:], start=1):
                        pseudo = pseudo_dict[elem]
                        if line_length + len(separator) + len(pseudo) > max_line_length:
                            f.write(separator_with_newline + ' ' * indent_length)
                            line_length = indent_length
                        else:
                            f.write(separator + ' ')
                            line_length += len(separator) + 1
                        f.write(pseudo)
                        line_length += len(pseudo)
                    f.write('"\n')
                elif keyword == "acell":
                    f.write("acell " + " ".join(f"{x:.6f}" for x in acell_bohr) + comment + "\n")
                elif keyword == "rprim":
                    f.write("rprim" + comment + "\n")
                    for vec in rprim:
                        f.write("  " + " ".join(f"{x:.6f}" for x in vec) + "\n")
                elif keyword == "ntypat":
                    f.write(f"ntypat {ntypat}{comment}\n")
                elif keyword == "znucl":
                    f.write("znucl " + " ".join(str(z) for z in znucl) + comment + "\n")
                elif keyword == "natom":
                    f.write(f"natom {natom}{comment}\n")
                elif keyword == "typat":
                    compressed_typat = compress_typat_simple(typat)
                    f.write(f"typat {compressed_typat}{comment}\n")
                elif keyword == "xred":
                    f.write("xred" + comment + "\n")
                    for pos in positions:
                        f.write("  " + "  ".join(f"{x:.6f}" for x in pos) + "\n")
                elif keyword == "toldfe":
                    toldfe_value = 1e-3 * len(atoms) / 27.2114
                    f.write(f"toldfe {toldfe_value:.6e} {comment}\n")
                else:
                    f.write(line)

        log_file = os.path.join(temp_dir,"abinit_run.log")

        # Run Abinit
        cmd = f"mpirun -np {cpu_count} /usr/bin/abinit opt_temp.abi"
        with open(log_file, "a") as log:
            subprocess.run(cmd, shell=True, cwd=temp_dir, stdout=log, stderr=subprocess.STDOUT)

        output_path = os.path.join(temp_dir, "opt_temp.abo")
        bohr_to_angstrom = 0.529177
        
        # Simulated content of opt_temp.abo (normally read from file)
        with open(output_path, "r") as f:
            content = f.read()
        
        # Extract acell
        acell_match = re.search(r"Scale of Primitive Cell \(acell\) \[bohr\]\n\s*(\S+)\s+(\S+)\s+(\S+)", content)
        acell = [float(acell_match.group(i)) * bohr_to_angstrom for i in range(1, 4)]
        
        # Extract rprimd
        rprimd_match = re.findall(r"Real space primitive translations \(rprimd\) \[bohr\]\n((?:\s*[-+Ee0-9.]+\s+[-+Ee0-9.]+\s+[-+Ee0-9.]+\n){3})", content)
        rprimd = []
        if rprimd_match:
            for line in rprimd_match[0].strip().split("\n"):
                rprimd.append([float(x) * bohr_to_angstrom for x in line.split()])
        
        # Extract xred
        xred_matches = re.findall(r"Reduced coordinates \(xred\)\n((?:\s*[-+Ee0-9.]+\s+[-+Ee0-9.]+\s+[-+Ee0-9.]+\n)+)", content)
        xred = []
        if xred_matches:
            for line in xred_matches[-1].strip().split("\n"):
                xred.append([float(x) for x in line.split()])
        
        # Construct cell matrix
        acell = [1.0, 1.0, 1.0]
        cell = np.dot(np.diag(acell), rprimd)
        
        # Create Atoms object
        atoms = Atoms(symbols=symbols, cell=cell, scaled_positions=xred, pbc=True)
        
        # Extract total energy
        etotal_match = re.search(r"Total energy \(etotal\) \[Ha\]=\s*(-?\d+\.\d+E?[+-]?\d*)", content)
        etotal = float(etotal_match.group(1)) * 27.2114 if etotal_match else 0.0  # Ha to eV
        energy = etotal
        
        opt_fname = fname.replace("valid_structures", "optimized_structures_vasp").replace("POSCAR", "OPT") + ".vasp"
        write(opt_fname, atoms, format='vasp')
        
        # Compute properties
        num_atoms = len(atoms)
        energy_per_atom = energy / num_atoms if num_atoms > 0 else 0.0
        relative_energy_per_atom = energy_per_atom - precursor_energy_per_atom

        total_mass_amu = sum(atoms.get_masses())
        total_mass_g = total_mass_amu * 1.66053906660e-24
        volume = atoms.get_volume()
        volume_cm3 = volume * 1e-24
        density = total_mass_g / volume_cm3 if volume_cm3 > 0 else 0

        print(f"Final energy per atom: {energy_per_atom:.6f} [eV/atom]")
        print(f"Final relative energy per atom: {relative_energy_per_atom:.6f} [eV/atom]")
        print(f"Number of atoms: {num_atoms}")
        print(f"Volume: {volume:.6f} [A^3]")
        print(f"Density: {density:.3f} [g/cm^3]")

        with open("structure_vs_energy.txt", "a") as out:
            out.write(f"{opt_fname} {relative_energy_per_atom:.6f} {energy_per_atom:.6f} {density:.3f} {num_atoms} {volume:.6f}\n")

    except Exception as e:
        print(f"Error optimizing {fname}: {e}")
#---------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
# --- Reference energy from original molecule ---
with open("structure_vs_energy.txt", "w") as f:
    print("# POSCAR file, Relative Energy [eV/atom], Total Energy [eV/atom], Density [g/cm^3], Number of atoms, Volume [A^3]", file=f)
#---------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
# --- Main ---
print(f"------------------------------------------------------")
print("# Calculate valid structures")
valid_files = []
directory = "valid_structures"
for fname in os.listdir(directory):
    filepath = os.path.join(directory, fname)
    try:
        print(f"Calculate file: {fname}")
        abinit_optimize(filepath, precursor_energy_per_atom=user_precursor_energy_per_atom)
        
        opt_fname_path  = f"optimized_structures_vasp/{fname}".replace("POSCAR", "OPT") + ".vasp"
        post_fname_path = f"valid_structures_postprocess/{fname}"
        shutil.copy(opt_fname_path, post_fname_path)
        print(f"------------------------------------------------------")
        continue
    except Exception:
        continue
#---------------------------------------------------------------------------------

print("Finished space group search and Lammps (GAFF) optimization.")
