# mol2crystal

## Install libraries
- mol2crystal.py
```
pip install ase==3.22.1 scipy==1.13.0 psutil==7.0.0
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

## Usage
- mol2crystal.py
```
pyton3 mol2crystal.py
```
- xTB version
```
pyton3 mol2crystal_xtb.py
```
- DFTB+ version
```
pyton3 mol2crystal_dftb.py
```

## Test
- Ubuntu 22.04 LTS (WSL2, Windows 11)
