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

# CP2k ver.9.1
sudo apt -y install cp2k

# Usage
pyton3 mol2crystal_cp2k.py
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
        cp2k_optimize(filepath, precursor_energy_per_atom=user_precursor_energy_per_atom)
        
        opt_fname_path  = f"optimized_structures_vasp/{fname}".replace("POSCAR", "OPT") + ".vasp"
        post_fname_path = f"valid_structures_postprocess/{fname}"
        shutil.copy(opt_fname_path, post_fname_path)
        print(f"------------------------------------------------------")
        continue
    except Exception:
        continue
#---------------------------------------------------------------------------------

print("Finished space group search and Lammps (GAFF) optimization.")
