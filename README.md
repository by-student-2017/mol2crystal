# mol2crystal

## Feature
- A crystal structure is created by applying a space group to a single precursor. The output is in VASP's POSCAR format.
- The system applies related space groups from the point group of a single precursor. A crystal structure with no atomic overlap is proposed for all subgroups related to the user-specified supergroup.
- The molecule is set as the geometric center, and then the atom position farthest from the geometric center is set to theta 45 degrees and phi 45 degrees. From there, the calculation is performed by dividing theta 0 - 45 and phi 0 - 45.
- Crystal structures created by ASE are output as primitive cells using spglib, which contributes to reducing the cost of calculating the total energy and geometry optimization later. The output can be expanded using VESTA.
- The generated crystal structure can then be used to obtain energy using the various quantum chemistry calculation codes listed below to create a density-energy diagram. From the resulting diagram, crystal structures with high density and low energy are selected as candidates.

## Selection of evaluation method
- Classical molecular dynamics (GAFF and ReaxFF in Lammps) have been developed. GAFF can perform in-cell geometry optimization, but the environment setup is complicated. ReaxFF is quite limited in the elements it can handle, but is expected to be a promising candidate search method.
- Empirical quantum chemical calculations (MOPAC, xTB, DFTB+) have also been developed. MOPAC is not currently recommended because it does not have a cell optimization function. xTB and DFTB+ are expected to be promising candidate search methods.
- First-principles calculation codes (QE, Abinit, OpenMX, GPAW, Siesta, CP2k, NWChem, Elk, etc.) are also available, but are not recommended due to their high computational cost. They may run without problems on medium- to large-scale computers.
- Not yet developed: NWChem

## Table 1. Comparison of Evaluation Methods
| Code      | version | Method  | OPT (pbc) | Element  | Accuracy | Cost      | Recommendaiton |
| --------- | ------- | ------- | --------- | -------- | -------- | --------- | -------------- |
| Lammps    |22Jul2025| GAFF    | Yes       | Possible | Low      | Low       | High           |
| Lammps    |22Jul2025| ReaxFF  | Yes       | limited  | Low      | Low       | Middle         |
|           |         |         |           |          |          |           |                |
| DFTB+     | 24.1    |GFN1-xTB | Yes       | Possible | Middle   | Middle    | High           |
| xTB       | 6.7.1   |GFN1-xTB | Yes       | Possible | Middle   | Middle    | Middle         |
| MOPAC     | 23.1.2  |PM6-D3H4 | No        | Possible | Middle   | Middle    | Low            |
|           |         |         |           |          |          |           |                |
| QE        | 6.7MaX  | DFT+vdW | Yes       | Possible | High     | High      | High           |
| Abinit    | 9.6.2   | DFT+vdW | Yes       | Possible | High     | High      | Low            |
| OpenMX    |3.8.5 (3.9)|DFT (+vdW)| Yes       | Possible | High     | High      | Middle         |
| GPAW      | 25.7.0  | DFT+vdW | Yes       | Possible | High     | High      | Middle         |
| Siesta    | 5.4.0   | DFT+vdW | Yes       | Possible | High     | High      | High           |
| CP2k      | 9.1     | DFT+vdW | Yes       | Possible | High     | Very high | Low            |
| NWChem    | 7.0.2   | DFT+vdW | Yes       | Possible | High     | Very high | Low            |
| Elk       | 7.2.42  | DFT     | Yes       | Possible | High     | Very high | Low            |

Table 2. vdW correation (This is necessary to consider intermolecular interactions more accurately.)
| Method | Note |
| ------ | ---- |
| GAFF   | Dispersion forces are approximated by an empirical Lennard-Jones potential (empirically adjusted values ​​are used, rather than theoretical corrections as in DFT-D). |
| ReaxFF | It is approximated by an empirically adjusted force field rather than a theoretical dispersion correction like Grimme (ReaxFF does not explicitly introduce dispersion forces (vdW) in the same way as DFT-D, but it does include Lennard-Jones type non-bonded interactions and has a distance-dependent potential between molecules).　|
|GFN-xTB | incorporates a term based on Grimme's D3 dispersion correction into the Hamiltonian.　|
|PM6-D3H4| D3 (Grimme dispersion correction) is taken into account. H4 (hydrogen bond correction) is a correction term to handle hydrogen bond energy more accurately (especially important for biomolecules and water clusters). |

| Code | Note |
| ------ | ---- |
| QE     | DFT-D, DFT-D3, MBD, and XDM are available. TS requires a special library (libvdwxc). |
| Abinit | DFT-D2, DFT-D3, and DFT-D3(BJ) are available. vdW-WF1, vdW-WF2, and vdW-QHO-WF) use Wannier functions, so they require the user to check and adjust parameters, such as fitting the Wannier functions, and are not easy to use. |
| OpenMX | DFT-D2 and DFT-D3 are supported from version 3.9 onwards. OpenMX version 3.9 or later is required. |
| GPAW   | DFT-D2, and DFT-D3 are available. Use the dftd3 code. |
| Siesta | DFT-D2, and DFT-D3 are available. |
|    | DFT-D2, DFT-D3, and DFT-D3(BJ) are available. Edit cp2k.inp to select the method you want. |
| NWChem | DFT-D2, and DFT-D3 are available. |

---

## Insatall main python code 
- For Ubuntu 22.04 LTS (pandas==1.3.5)
```
cd $HOME
git clone https://github.com/by-student-2017/mol2crystal.git
cd mol2crystal

sudo apt update
sudo apt -y install python3-pip python3-tk

pip install ase==3.22.1 scipy==1.13.0 psutil==7.0.0
pip install "numpy<2.0"
pip install pymsym==0.3.4
pip install spglib==2.6.0
pip install pandas
```
- For Ubuntu 24.04 LTS (pandas==2.3.2)
```
cd $HOME
sudo apt update
sudo apt -y install python3-pip python3-venv python3-tk

git clone https://github.com/by-student-2017/mol2crystal.git
cd mol2crystal

python3 -m venv ~/mol2crystal/venv
source ~/mol2crystal/venv/bin/activate

pip install ase==3.22.1 scipy==1.13.0 psutil==7.0.0
pip install "numpy<2.0"
pip install pymsym==0.3.4
pip install spglib==2.6.0
pip install pandas
```
- After the above operation, the following command is required first every time on Ubuntu 24.04 LTS:
```
deactivate
source ~/mol2crystal/venv/bin/activate
```

### classic MD
- Lammps version (ReaxFF)
```
# lammps (Installation: 2025/Aug/22)
cd $HOME
sudo apt -y install cmake gfortran gcc libopenmpi-dev
git clone -b stable https://github.com/lammps/lammps.git
cd lammps
mkdir build && cd build
cmake -D BUILD_MPI=yes -D BUILD_SHARED_LIBS=no -D PKG_KSPACE=yes -D PKG_MOLECULE=yes -D PKG_EXTRA-MOLECULE=yes -D PKG_USER-MISC=yes -D PKG_EXTRA-DUMP=yes -D PKG_REAXFF=yes -D PKG_QEQ=yes -D PKG_MC=yes -D PKG_EAM=yes -D PKG_MEAM=yes -D PKG_RIGID=yes -D PKG_USER-CG-CMM=yes ../cmake
make -j$(nproc)
sudo make install
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

- Lammps version (GAFF) (Ubuntu 22.04 LTS) (More robust than Ubuntu 24.04)
```
# lammps + moltemplate + antechamber + mol22lt.pl (Ref. [2])
cd $HOME
sudo apt update
sudo apt -y install dos2unix python3-pip libgfortran5 liblapack3 libtext-template-perl
wget https://github.com/makoto-yoneya/makoto-yoneya.github.io/raw/master/LAMMPS-organics/install_moltemplate.sh
sh install_moltemplate.sh
wget https://github.com/makoto-yoneya/makoto-yoneya.github.io/raw/master/LAMMPS-organics/install_WSLmisc.sh
sh install_WSLmisc.sh
```
- Lammps version (GAFF) (Ubuntu 24.04 LTS)
```
sudo apt install libtext-template-perl
python3 -m pip install --upgrade pip
python3 -m venv ~/mol2crystal/venv
source ~/mol2crystal/venv/bin/activate

# lammps + moltemplate + antechamber + mol22lt.pl (Ref. [2])
cd $HOME
sudo apt update
sudo apt -y install dos2unix python3-pip libgfortran5 liblapack3 libtext-template-perl
wget https://github.com/makoto-yoneya/makoto-yoneya.github.io/raw/master/LAMMPS-organics/install_moltemplate.sh
sh install_moltemplate.sh
wget https://github.com/makoto-yoneya/makoto-yoneya.github.io/raw/master/LAMMPS-organics/install_WSLmisc.sh
sh install_WSLmisc.sh

pip install moltemplate==2.22.4
chmod +x ~/bin/mol22lt.pl
cd ~/opt/moltemplate
pip install ./
sudo cp ~/opt/moltemplate/moltemplate/scripts/moltemplate.sh ~/.local/bin/
sudo chmod +x ~/.local/bin/moltemplate.sh
```
- Lammps version (GAFF): The procedure is the same as for the Lammps version (ReaxFF), so it is not necessary if it is already installed.
```
# lammps (Installation: 2025/Aug/22)
cd $HOME
sudo apt -y install cmake gfortran gcc libopenmpi-dev
git clone -b stable https://github.com/lammps/lammps.git
cd lammps
mkdir build && cd build
cmake -D BUILD_MPI=yes -D BUILD_SHARED_LIBS=no -D PKG_KSPACE=yes -D PKG_MOLECULE=yes -D PKG_EXTRA-MOLECULE=yes -D PKG_USER-MISC=yes -D PKG_EXTRA-DUMP=yes -D PKG_REAXFF=yes -D PKG_QEQ=yes -D PKG_MC=yes -D PKG_EAM=yes -D PKG_MEAM=yes -D PKG_RIGID=yes -D PKG_USER-CG-CMM=yes ../cmake
make -j$(nproc)
sudo make install
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### Semi-empirical quantum chemical calculations
- xTB version
```
### xTB ver. 6.7.1
cd $HOME
wget https://github.com/grimme-lab/xtb/releases/download/v6.7.1/xtb-6.7.1-linux-x86_64.tar.xz
tar xvf xtb-6.7.1-linux-x86_64.tar.xz
echo 'export PATH=$PATH:$HOME/xtb-dist/bin' >> ~/.bashrc
source ~/.bashrc
```
- DFTB+ version
```
### DFTB+ ver. 24.1
cd $HOME
wget https://github.com/dftbplus/dftbplus/releases/download/24.1/dftbplus-24.1.x86_64-linux.tar.xz
tar -xvf dftbplus-24.1.x86_64-linux.tar.xz
echo 'export PATH=$PATH:$HOME/dftbplus-24.1.x86_64-linux/bin' >> ~/.bashrc
source ~/.bashrc
```
- MOPAC version (Not recommended: Cells cannot be optimized)
```
### MOPAC ver. 23.1.2
wget https://github.com/openmopac/mopac/releases/download/v23.1.2/mopac-23.1.2-linux.tar.gz
tar xvf mopac-23.1.2-linux.tar.gz
echo 'export PATH=$PATH:$HOME/mopac-23.1.2-linux/bin' >> ~/.bashrc
source ~/.bashrc
```

### First-principles calculation (band calculation)
- QE version
```
# QE v.6.7MaX (Ubuntu 22.04 LTS), 6.7 (Ubuntu 24.04 LTS)　
sudo apt update
sudo apt -y install quantum-espresso

### Pseudo-potantials
# QE pslibrary: https://pseudopotentials.quantum-espresso.org/legacy_tables/ps-library
# TEHOS: https://theos-wiki.epfl.ch/Main/Pseudopotentials
# pslibrary: https://dalcorso.github.io/pslibrary/PP_list.html
# SSSP: https://www.materialscloud.org/discover/sssp/table/efficiency
# (Set the pseudopotential in the pseudodirectory.)

```
- Abinit
```
### Abinit v.9.6.2 (Ubuntu 22.04 LTS), 9.10.4 (Ubuntu 24.04 LTS)
sudo apt update
sudo apt -y install abinit
```
- OpenMX version
```
### OpenMX v3.8.5 (Ubuntu 22.04 LTS)
sudo apt update
sudo apt -y install openmx
```
- GPAW version
```
### Install libraries + GPAW ver. 25.7.0
pip install gpaw==25.7.0

### DFTD-D3
pip install dftd3==1.2.1
cd $HOME
git clone https://github.com/loriab/dftd3.git
cd dftd3
make
sudo cp dftd3 /usr/local/bin/
```
- SIESTA version
```
### Siesta Installation
sudo apt update
sudo apt -y install cmake gfortran build-essential libopenmpi-dev libopenblas-dev 
sudo apt -y install libhdf5-dev pkg-config libreadline-dev
cd $HOME
wget https://gitlab.com/siesta-project/siesta/-/releases/5.4.0/downloads/siesta-5.4.0.tar.gz
tar xvf siesta-5.4.0.tar.gz
cd siesta-5.4.0
cmake -S . -B _build -DSIESTA_WITH_FLOOK="OFF"
cmake --build _build -j 4
sudo cmake --install _build
echo 'export SIESTA_PP_PATH=$HOME/siesta-5.4.0/Pseudo/ThirdParty-Tools/ONCVPSP/nc-sr-05_pbe_standard_psml' >> ~/.bashrc  # path of pseudo-potentials
source ~/.bashrc

### Pseudo-potantials
# https://www.pseudo-dojo.org/
cd $HOME/siesta-5.4.0/Pseudo/ThirdParty-Tools/ONCVPSP$
# (set) nc-sr-05_pbe_standard_psml.tgz
tar xvf nc-sr-05_pbe_standard_psml.tgz
```
- CP2k version
```
### CP2k ver.9.1 (Ubuntu 22.04 LTS)
sudo apt -y install cp2k
```
- CP2k ver.2025.2 version (Ubuntu 24.04 LTS)
```
sudo apt update
sudo apt install -y cmake gfortran mpich libopenblas-dev liblapack-dev libscalapack-openmpi-dev libfftw3-dev libxc-dev libint-dev unzip

$HOME
git clone https://github.com/spack/spack.git
source spack/share/spack/setup-env.sh
#spack install cp2k@2025.2 +mpi ^libint ^libxc ^fftw ^scalapack
spack install cp2k@2025.2 +mpi ^libint ^libxc ^fftw
spack location -i cp2k
spack load cp2k
cp2k.psmp --version

mkdir -p $HOME/cp2k-data
cd $HOME/cp2k-data
wget https://raw.githubusercontent.com/cp2k/cp2k/master/data/BASIS_MOLOPT
wget https://raw.githubusercontent.com/cp2k/cp2k/master/data/GTH_POTENTIALS
wget https://raw.githubusercontent.com/cp2k/cp2k/master/data/dftd3.dat

sudo mkdir -p /usr/share/cp2k
sudo cp dftd3.dat BASIS_MOLOPT GTH_POTENTIALS /usr/share/cp2k/

echo 'export CP2K_DATA_DIR=/usr/share/cp2k' >> ~/.bashrc
source ~/.bashrc
```
- NWChem version
```
### NWChem v7.0.2 (Ubuntu 22.04 LTS), 7.2.2 (Ubuntu 24.04 LTS)
sudo apt update
sudo apt -y install nwchem-openmpi
```
- Elk
```
### Elk v7.2.42 (Ubuntu 22.04 LTS), 9.2.12 (Ubuntu 24.04 LTS)
sudo apt update
sudo apt -y install elk-lapw
```

---

## Usage
1. Draw a molecule with the free version of ChemSketch and output it in mol format.
2. Add hydrogens using Avogadro and save the molecule in mol format. Name the molecule "precursor.mol."
3. Put the mol file in molecular_files.
4. Make sure the Python code and molecular_files are in the same directory.
5. Execute the following command:
- mol2crystal.py: Apply space group to molecule. Since no other calculations are performed, interatomic and cell optimizations are not performed. A separate code is required for geometry optimization.
```
pyton3 mol2crystal.py
```
- Post-processing python3 code is provided, so it is recommended to use it after deleting unnecessary structures using valid_structures. For example, to use Lammp's GAFF, use the following:
```
python3 postprocess_gaff_pbc.py
```
- The data is recorded in “structure_vs_energy.txt” immediately after the calculation, so you can plot the results with the following command even if the entire calculation has not yet finished.
```
python3 plot.py
```
- It is also possible to plot using Gnuplot. If Gnuplot is installed using "sudo apt -y install gnuplot", you can get a figure with the following command.
```
gnuplot plot.gpl
```
- For the Windows version, you can also run it by simply double-clicking "plot.gpl".
- I've also provided Python code for selecting data, from which you can get candidates. I recommend removing "_selected" from valid_structures_selected and recalculating with a more accurate method. For example, try using DFTB+ or gaff_pbc QE.
```
python3 select_data.py
```

### classic MD
- Lammps version (ReaxFF): Cells can also be optimized. The prediction accuracy is not too bad either. Note that there is no combination of potentials for all elements (https://github.com/by-student-2017/lammps_education_reaxff_win/tree/master/potentials).
```
pyton3 mol2crystal_reaxff.py
```
- Lammps version (GAFF): Cells can also be optimized. The prediction accuracy is not too bad either.
```
pyton3 mol2crystal_gaff_pbc.py
```

### Semi-empirical quantum chemical calculations
- xTB version: Intermediate accuracy and computational cost between classical MD and first-principles calculations.
```
pyton3 mol2crystal_xtb.py
```
- DFTB+ version: Intermediate accuracy and computational cost between classical MD and first-principles calculations.
```
pyton3 mol2crystal_dftb.py
```
- MOPAC version: Geometry optimization was not performed. Note that this code outputs energies relative to the precursor energy. Cell optimization is not possible. (Not recommended: Cells cannot be optimized)
```
pyton3 mol2crystal_mopac.py
```

### First-principles calculation (band calculation)
- QE version: High accuracy due to first-principles calculation, but high calculation cost.
```
pyton3 mol2crystal_qe.py
```
- Abinit
```
pyton3 mol2crystal_abinit.py
```
- OpenMX version
```
pyton3 mol2crystal_openmx.py
```
- GPAW version: High accuracy due to first-principles calculation, but high calculation cost.
```
pyton3 mol2crystal_gpaw.py
```
- SIESTA version
```
pyton3 mol2crystal_siesta.py
```
- CP2k version
```
pyton3 mol2crystal_cp2k.py
```
- NWChem version
```
pyton3 mol2crystal_nwchem.py
```
- Elk version: This method provides very high accuracy, but it is computationally expensive and requires a lot of memory. (I do not recommend this method for calculations.)
```
pyton3 mol2crystal_elk.py
```

### plot
- matplotlib version
```
python3 plot.py
```
- gnuplot version
```
gnuplot plot.gpl
```

### select output data
- I've also provided Python code for selecting data, from which you can get candidates. I recommend removing "_selected" from valid_structures_top and recalculating with a more accurate method. For example, try using DFTB+ or gaff_pbc QE.
```
python3 select_data.py
```

## How to change the calculation conditions
- The calculation conditions can be easily changed at the beginning of the Python code. The corresponding part is as follows.
```
user_margin = 1.70                   # >= vdW radius (H:1.20 - Cs:3.43)
user_margin_scale = 1.2              # Intermolecular arrangement: 1.2 - 1.5, Sparse placement (e.g., porous materials): 1.6 - 2.0
user_nmesh = 3                       # 0 - 45 degrees divided into nmesh
user_overlap_scale = 0.80            # threshold = scale * (r_i + r_j), covalent_radii: r_i and r_j
#user_included_spacegroups = [3,230]  # Include certain space groups from consideration  (high priority)
user_included_spacegroups = list(range(3, 231)) # Includes all space group numbers 3 to 230
user_excluded_spacegroups = [1,2]    # Exclude certain space groups from consideration  (low  priority)
user_skipping_spacegroups = 231      # Omit if space group >= user_skipping_spacegroups (low priority)
user_max_depth = 1                   # Neighborhood and top-level search. Number of recursions to find candidates.
user_skipping_n_molecules = 100      # Skip large molecular systems (>= user_skipping_n_molecules) (high priority)
user_primitive_cell_output = 1       # 0:No, 1:Yes (using spglib==2.6.0)
user_precursor_energy_per_atom = 0.0 # [eV] The reference energy (precursor alone) when calculating relative energy.
```
- The parameters that you need to change in particular are "user_nmesh" and "user_max_depth". You can change the other parameters as you get used to it.

## Usage (semi-auto)
- The command below uses select_data.py to narrow down the target and improve calculation accuracy.
```
bash run_seq_select.sh
```
- The # at the beginning of the run_step line in the code selects which calculation method to use. If there is no # at the beginning of the line, calculation will be performed, and if there is a # at the beginning of the line, it will be commented out and no calculation will be performed.

---

## Comments on code improvements
- Adjust unit cell parameters based on space group symmetry: I created this by hand using VESTA as a reference, so there may be errors in my input. If you find any errors, please let me know and I will correct them.
- Analyze point group and derive candidate space groups: Setting a wider point group using "related_point_groups_physical" reduces the chance of missing structure.
- It may also be a good idea to calculate user_precursor_energy_per_atom at the precursor.mol stage. This has been omitted because it would complicate the Python3 code. We also attempted calculations for space group 1, but in some cases it made it difficult to understand problems with the crystal structure, so we have decided not to do so here.

---

## requirements.txt
```
pip freeze > requirements.txt
```

---

## lammps_reaxff_md_windows11
- This calculation is inefficient because it uses packmol to randomly arrange the particles and then apply high pressure.
- It is better to apply the calculation to the crystal structure obtained with mol2crystal.py.
- The calculation includes dielectric constant. In other words, this is an example that takes the solvent into account.

---

## Tested Environment
- OS: Ubuntu 22.04 LTS (WLS2, Windows 11)
- Python: 3.10.12
- ASE: 3.22.1
- scipy: 1.13.0
- psutil: 7.0.0
- pymsym: 0.3.4
- Numpy: 1.21.5
- Matplotlib: 3.5.1
- spglib: 2.6.0
- gfortran: gcc version 11.4.0 (Ubuntu 11.4.0-1ubuntu1~22.04)

---

## References
- [1] S. OBATA et al., Nihon Kessho Gakkaishi 62 (2020) 260-268. (Japanese): https://doi.org/10.5940/jcrsj.62.260
- [2] https://makoto-yoneya.github.io/
- [3] ASE (Atomic Simulation Environment): https://wiki.fysik.dtu.dk/ase/
- [4] pymsym: https://github.com/yoneya/pymsym
- [5] spglib: https://github.com/spglib
- [6] Reaxff potential list: https://github.com/by-student-2017/lammps_education_reaxff_win/tree/master/potentials (When presenting your results, be sure to cite the potential you utilized.)
- [7] dftd3: https://github.com/loriab/dftd3
- [8] Point Group Tables: https://www.cryst.ehu.es/rep/point

---

## Error of Copilot (Edit: 2025/Aug/28)
- "‘quasi-DFT method’ (準DFT法) and ‘quasi-first-principles calculation’ (準第一原理計算) are incorrect translations of ‘semi-empirical DFT method’. Copilot has mistakenly used these terms. I sincerely hope this will be corrected.
- The prefixes ‘semi-’ and ‘quasi-’ are fundamentally different in meaning, and it is difficult to confuse them—even from the perspective of conditional probability in Bayesian inference or attention mechanisms in machine learning. It would be advisable to investigate thoroughly why such a mistranslation has occurred."

---
## Fig.4 Results and computational time of crystal structure searches (Ref: https://doi.org/10.5940/jcrsj.62.260)

| Crystal Structure | Rank on MMFF94s | Crystal Energy Difference ΔEcrystal (kcal/mol) | Heavy Atom RMSD (Angstrom) | Computation Time (h) |
|----------------|--------------------|------------------------|--------------|--------------------|
| 1（Orth.）     |                  3 |                  0.262 |        0.347 |               3.97 |
| 1（Mono.）     |                269 |                  1.913 |        0.601 |               3.97 |
| 2              |                  1 |                  0     |        0.393 |               4.21 |
| 4 （Form I）   |                 55 |                  1.014 |        0.299 |               9.43 |
| 5              |                 56 |                  1.637 |        0.617 |              15.03 |
| 6（Form I）    |                 70 |                  3.084 |        0.401 |               24.4 |
| 6（Form III）  |                107 |                  3.606 |        0.674 |               24.4 |
| 7              |                 22 |                  0.164 |        0.155 |               1.94 |
| 8              |                  9 |                  0.965 |        0.398 |               4.34 |
| 9              |                 54 |                  1.875 |        0.507 |              21.34 |
| 10             |               2350 |                  6.053 |        0.442 |               19.9 |
| 12             |                  8 |                  0.254 |        0.302 |               2.66 |
| 13             |                104 |                  1.232 |        0.673 |               3.55 |
| 14             |                216 |                  1.801 |        1.245 |              15.93 |
| 16             |                454 |                  2.292 |        0.51  |               6.61 |
| 17             |                 83 |                  2.464 |        0.559 |               7.39 |
| 20             |                 39 |                  2.387 |        0.361 |             154.31 |
| 26             |                  2 |                  0.862 |        0.55  |              91.17 |

**Footnotes:**
- a: Crystal energy ranking using the MMFF94s force field.
- b: Crystal energy difference from the most stable structure.
- c: Root mean square difference of heavy atom positions when superimposing the experimental structure with 20 molecules.
- d: Computation time when performing MPI/OpenMP hybrid parallel calculations using an Intel Xeon Gold 6154 3.00 GHz CPU with 64 cores.
