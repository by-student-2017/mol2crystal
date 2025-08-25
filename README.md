# mol2crystal

## Feature
- A crystal structure is created by applying a space group to a single precursor. The output is in VASP's POSCAR format.
- The generated crystal structure can then be used to obtain energy using the various quantum chemistry calculation codes listed below to create a density-energy diagram. From the resulting diagram, crystal structures with high density and low energy are selected as candidates.

## Selection of evaluation method
- Classical molecular dynamics (GAFF and UFF in OpenBabel, GAFF and ReaxFF in Lammps) have been developed. OpenBabel's GAFF and UFF are suitable for rough screening because they cannot perform in-cell geometry optimization. On the other hand, Lammps' GAFF can perform in-cell geometry optimization, but the environment setup is complicated. ReaxFF is quite limited in the elements it can handle, but is expected to be a promising candidate search method.
- Empirical quantum chemical calculations (MOPAC, xTB, DFTB+) have also been developed. MOPAC is not currently recommended because it does not have a cell optimization function. xTB and DFTB+ are expected to be promising candidate search methods.
- First-principles calculation codes (GPAW, CP2k, Siesta, QE, Abinit, Elk, etc.) are also available, but are not recommended due to their high computational cost. They may run without problems on medium- to large-scale computers.
- Not yet developed: Lammps (or lammps + moltemplate), QE, Abinit, Elk

## Install libraries
- mol2crystal.py
```
pip install ase==3.22.1 scipy==1.13.0 psutil==7.0.0
pip install pymsym==0.3.4
```
- OpenBabel version (GAFF or UFF)
```
### Install libraries
pip install ase==3.22.1 scipy==1.13.0 psutil==7.0.0
pip install pymsym==0.3.4

### OpenBable
sudo apt update
sudo apt install openbabel
sudo apt install libopenbabel-dev
```
- Lammps version (GAFF)
```
### Install libraries
pip install ase==3.22.1 scipy==1.13.0 psutil==7.0.0
pip install pymsym==0.3.4

# lammps + moltemplate + antechamber + mol22lt.pl (Ref. [2])
sudo apt update
sudo apt -y install dos2unix python3-pip libgfortran5 liblapack3
wget https://github.com/makoto-yoneya/makoto-yoneya.github.io/raw/master/LAMMPS-organics/install_moltemplate.sh
sh install_moltemplate.sh
wget https://github.com/makoto-yoneya/makoto-yoneya.github.io/raw/master/LAMMPS-organics/install_WSLmisc.sh
sh install_WSLmisc.sh

# lammps (Installation: 2025/Aug/22)
cd $HOME
sudo apt -y install cmake gfortran gcc libopenmpi-dev
git clone -b stable https://github.com/lammps/lammps.git
cd lammps
mkdir build && cd build
cmake -D BUILD_MPI=yes -D BUILD_SHARED_LIBS=no -D PKG_KSPACE=yes -D PKG_MOLECULE=yes -D PKG_EXTRA-MOLECULE=yes -D PKG_USER-MISC=yes -D PKG_EXTRA-DUMP=yes -D PKG_REAXFF=yes -D PKG_QEQ=yes -D PKG_MC=yes -D PKG_EAM=yes -D PKG_RIGID=yes -D PKG_USER-CG-CMM=yes ../cmake
make -j$(nproc)
sudo make install
```
- MOPAC version
```
### Install libraries
pip install ase==3.22.1 scipy==1.13.0 psutil==7.0.0
pip install pymsym==0.3.4

### MOPAC ver. 23.1.2
wget https://github.com/openmopac/mopac/releases/download/v23.1.2/mopac-23.1.2-linux.tar.gz
tar xvf mopac-23.1.2-linux.tar.gz
echo 'export PATH=$PATH:$HOME/mopac-23.1.2-linux/bin' >> ~/.bashrc
```
- xTB version
```
### Install libraries
pip install ase==3.22.1 scipy==1.13.0 psutil==7.0.0
pip install pymsym==0.3.4

### xTB ver. 6.7.1
cd $HOME
wget https://github.com/grimme-lab/xtb/releases/download/v6.7.1/xtb-6.7.1-linux-x86_64.tar.xz
tar xvf xtb-6.7.1-linux-x86_64.tar.xz
echo 'export PATH=$PATH:$HOME/xtb-dist/bin' >> ~/.bashrc
```
- DFTB+ version
```
pip install ase==3.22.1 scipy==1.13.0 psutil==7.0.0
pip install pymsym==0.3.4

### DFTB+ ver. 24.1
cd $HOME
wget https://github.com/dftbplus/dftbplus/releases/download/24.1/dftbplus-24.1.x86_64-linux.tar.xz
tar -xvf dftbplus-24.1.x86_64-linux.tar.xz
echo 'export PATH=$PATH:$HOME/dftbplus-24.1.x86_64-linux/bin' >> ~/.bashrc
```
- GPAW version
```
### Install libraries + GPAW ver. 25.7.0
pip install ase==3.26.0 scipy==1.13.0 psutil==7.0.0 gpaw==25.7.0
pip install "numpy<2.0"
pip install pymsym==0.3.4
```
- CP2k version
```
### Install libraries
pip install ase==3.22.1 scipy==1.13.0 psutil==7.0.0
pip install pymsym==0.3.4

### CP2k ver.9.1
sudo apt -y install cp2k
```
- SIESTA version
```
### Install libraries
pip install ase==3.26.0 scipy==1.13.0 psutil==7.0.0 gpaw==25.7.0
pip install pymsym==0.3.4

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
- OpenBabel version (GAFF or UFF): Geometry optimization was not performed. Note that this code outputs energies relative to the precursor energy. The computational cost is very low.
```
pyton3 mol2crystal_gaff.py
```
- Lammps version (GAFF): Cells can also be optimized. The prediction accuracy is not too bad either.
```
pyton3 mol2crystal_gaff_pbc.py
```
- MOPAC version: Geometry optimization was not performed. Note that this code outputs energies relative to the precursor energy. Cell optimization is not possible.
```
pyton3 mol2crystal_mopac.py
```
- xTB version: Intermediate accuracy and computational cost between classical MD and first-principles calculations.
```
pyton3 mol2crystal_xtb.py
```
- DFTB+ version: Intermediate accuracy and computational cost between classical MD and first-principles calculations.
```
pyton3 mol2crystal_dftb.py
```
- GPAW version: High accuracy due to first-principles calculation, but high calculation cost.
```
pyton3 mol2crystal_gpaw.py
```
- CP2k version
```
pyton3 mol2crystal_cp2k.py
```
- SIESTA version
```
pyton3 mol2crystal_siesta.py
```

## plot
- matplotlib version
```
python3 plot.py
```
- gnuplot version
```
gnuplot plot.gpl
```

## lammps_reaxff_md_windows11
- This calculation is inefficient because it uses packmol to randomly arrange the particles and then apply high pressure.
- It is better to apply the calculation to the crystal structure obtained with mol2crystal.py.
- The calculation includes dielectric constant. In other words, this is an example that takes the solvent into account.

## Tested Environment
- Ubuntu: 22.04 LTS (WSL2, Windows 11)
- Python: 3.10
- ASE: 3.22.1
- LAMMPS version: stable_22Jul2025

## References
- [1] S. OBATA et al., Nihon Kessho Gakkaishi 62 (2020) 260-268. (Japanese): https://doi.org/10.5940/jcrsj.62.260
- [2] https://makoto-yoneya.github.io/
