#!/usr/bin/env python3


#---------------------------------------------------------------------------------
# User setting parameters
#------------------------------------
user_precursor_energy_per_atom = 0.0 # [eV] The reference energy (precursor alone) when calculating relative energy.
#---------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
# --- Prepare environment and clean previous results ---
'''
# Install libraries + GPAW ver. 25.7.0
pip install ase==3.26.0 scipy==1.13.0 psutil==7.0.0 gpaw==25.7.0
pip install "numpy<2.0"
pip install pymsym==0.3.4
pip install spglib==2.6.0

# DFTD-D3
pip install dftd3==1.2.1
cd $HOME
git clone https://github.com/loriab/dftd3.git
cd dftd3
make
sudo cp dftd3 /usr/local/bin/

# Usage
pyton3 mol2crystal_gpaw.py
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

# GPAW settings
from gpaw import GPAW, PW, Davidson
from ase.filters import UnitCellFilter
from ase.optimize import BFGS, LBFGS, FIRE
from gpaw import Mixer

# DFTD-D3 settings
from ase.calculators.dftd3 import DFTD3

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
# GPAW optimization
def gpaw_optimize(fname, precursor_energy_per_atom):
    try:
        temp_dir = "gpaw_temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        atoms = read(fname)

        # Output GPAW calculation to temporary directory
        gpaw_calc = GPAW(xc          = 'PBE',             # PBE, RPBE, PBEsol, PW91, BLYP, LDA, CA, TPSS, (LibXC: SCAN, HSE06, B3LYP)
                         kpts        = (1, 1, 1),         # k-point mesh (gamma-point only), suitable for molecules and isolated systems
                         mode        = PW(300),           # The plane wave basis is used, with a cutoff energy of 300 eV.
                         spinpol     = False,             # Spin-unpolarized calculations (applicable to closed-shell and non-magnetic systems)
                         occupations = {
                            'name': 'fermi-dirac',        # 'marzari-vanderbilt', 'methfessel-paxton', 'tetrahedron-method', 'improved-tetrahedron-method'
                            'width': 0.05},               # The electron smearing is set by the Fermi distribution. The electron temperature is 0.05 eV.
                         convergence = {
                            'energy': 1e-3 * len(atoms),  # Total energy convergence criterion (eV)
                            'density': 1e-4,              # Electron density convergence criteria
                            'eigenstates': 1e-8           # Convergence of eigenstates (if necessary)
                         },
                         charge      = 0,                 # Charge of the entire system, assuming a neutral system (e.g., molecule, surface)
                         symmetry    = {'point_group': True}, # Enable point group symmetry, which contributes to improved computational efficiency
                         maxiter     = 200,               # Maximum number of SCF iterations, increase if no convergence occurs
                         eigensolver = 'cg',              # cg: Conjugate gradient method (Fast and stable), dav: Davidson method (Good performance in most cases)
                         mixer       = Mixer(beta=0.05,   # Reducing the mixing coefficient reduces vibration.
                                             nmaxold=5,   # Reducing the number of histories stabilizes the
                                             weight=50.0  # Increase the value to emphasize the short wavelength component and suppress charge sloshing.
                                             ),
                         parallel    = {'domain': 1, 'band': 1}, # Parallel calculation settings. Valid when using MPI.
                         txt         = os.path.join(temp_dir, 'gpaw_out.txt'), # Destination for saving output files. Calculation logs are recorded.
                         )
        # old = DFT-D2
        calc = DFTD3(dft=gpaw_calc, xc='pbe', old=True, command='/usr/local/bin/dftd3')
        atoms.calc = calc

        ucf = UnitCellFilter(atoms)
        opt = FIRE(ucf, 
            logfile=os.path.join(temp_dir, 'opt.log'),
            trajectory=os.path.join(temp_dir, 'opt.traj'))
        '''
        opt = LBFGS(ucf,
            logfile=os.path.join(temp_dir, 'opt.log'),
            trajectory=os.path.join(temp_dir, 'opt.traj'))
        '''
        opt.run(fmax=0.5)

        # Save the final structure
        opt_fname = fname.replace("valid_structures", "optimized_structures_vasp").replace("POSCAR", "OPT") + ".vasp"
        write(opt_fname, atoms, format='vasp')
        input()
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
                out.write(f"{opt_fname} {relative_energy_per_atom:.6f} {energy_per_atom:.6f} {density:.3f} {num_atoms} {volume:.6f}\n")
        else:
            print("Energy value not found in xtbopt.log.")

    except Exception as e:
        import traceback
        traceback.print_exc()
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
        gpaw_optimize(filepath, precursor_energy_per_atom=user_precursor_energy_per_atom)
        
        opt_fname_path  = f"optimized_structures_vasp/{fname}".replace("POSCAR", "OPT") + ".vasp"
        post_fname_path = f"valid_structures_postprocess/{fname}"
        shutil.copy(opt_fname_path, post_fname_path)
        print(f"------------------------------------------------------")
        continue
    except Exception:
        continue
#---------------------------------------------------------------------------------

print("Finished space group search and GPAW optimization.")
