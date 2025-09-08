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
#from ase.spacegroup import crystal
#from scipy.spatial.distance import pdist
import subprocess
import psutil
import re
from ase.geometry import cellpar_to_cell
#from ase.neighborlist import NeighborList
#from ase.data import vdw_radii, atomic_numbers

# QE settings
from ase.calculators.espresso import Espresso
from ase.optimize import BFGS, LBFGS, FIRE

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
        
        calc = Espresso(
            command=f'mpirun -n {cpu_count} /usr/bin/pw.x',
            pseudo_dir=pseudo_dir,
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
# --- Main ---
print(f"------------------------------------------------------")
print("# Calculate valid structures")
valid_files = []
directory = "valid_structures"
for fname in os.listdir(directory):
    filepath = os.path.join(directory, fname)
    try:
        print(f"Calculate file: {fname}")
        qe_optimize(filepath, precursor_energy_per_atom=user_precursor_energy_per_atom)
        
        opt_fname_path  = f"optimized_structures_vasp/{fname}".replace("POSCAR", "OPT") + ".vasp"
        post_fname_path = f"valid_structures_postprocess/{fname}"
        shutil.copy(opt_fname_path, post_fname_path)
        print(f"------------------------------------------------------")
        continue
    except Exception:
        continue
#---------------------------------------------------------------------------------

print("Finished space group search and QE optimization.")
