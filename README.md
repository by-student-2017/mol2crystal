# mol2crystal

## Install libraries
- mol2crystal.py
```
pip install ase==3.22.1 scipy==1.13.0 psutil==7.0.0
```
- OpenBabel version (GAFF or UFF)
```
pip install ase==3.22.1 scipy==1.13.0 psutil==7.0.0
sudo apt update
sudo apt install openbabel
sudo apt install libopenbabel-dev
```
- MOPAC version
```
pip install ase==3.22.1 scipy==1.13.0 psutil==7.0.0
wget https://github.com/openmopac/mopac/releases/download/v23.1.2/mopac-23.1.2-linux.tar.gz
tar xvf mopac-23.1.2-linux.tar.gz
echo 'export PATH=$PATH:$HOME/mopac-23.1.2-linux/bin' >> ~/.bashrc
```
- xTB version
```
pip install ase==3.22.1 scipy==1.13.0 psutil==7.0.0
cd $HOME
wget https://github.com/grimme-lab/xtb/releases/download/v6.7.1/xtb-6.7.1-linux-x86_64.tar.xz
tar xvf xtb-6.7.1-linux-x86_64.tar.xz
echo 'export PATH=$PATH:$HOME/xtb-dist/bin' >> ~/.bashrc
```
- DFTB+ version
```
pip install ase==3.22.1 scipy==1.13.0 psutil==7.0.0
cd $HOME
wget https://github.com/dftbplus/dftbplus/releases/download/24.1/dftbplus-24.1.x86_64-linux.tar.xz
tar -xvf dftbplus-24.1.x86_64-linux.tar.xz
echo 'export PATH=$PATH:$HOME/dftbplus-24.1.x86_64-linux/bin' >> ~/.bashrc
```
- GPAW version
```
pip install ase==3.26.0 scipy==1.13.0 psutil==7.0.0 gpaw==25.7.0
pip install "numpy<2.0"
```
- CP2k version
```
pip install ase==3.22.1 scipy==1.13.0 psutil==7.0.0
sudo apt -y install cp2k
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
- MOPAC version: Geometry optimization was not performed. Note that this code outputs energies relative to the precursor energy. Cell optimization is not possible.
```
pyton3 mol2crystal_mopac.py
```
- xTB version: Intermediate accuracy and computational cost between classical MD and first-principles calculations. Cell optimization is not possible.
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

## plot
```
python3 plot.py
```

## lammps_reaxff_md_windows11
- This calculation is inefficient because it uses packmol to randomly arrange the particles and then apply high pressure.
- It is better to apply the calculation to the crystal structure obtained with mol2crystal.py.
- The calculation includes dielectric constant. In other words, this is an example that takes the solvent into account.

## Test
- Ubuntu 22.04 LTS (WSL2, Windows 11)
