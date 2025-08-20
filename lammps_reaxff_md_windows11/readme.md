# Lammps + reaxff version

## lammps (windows 11 (64 bit))

### Installation (Lammps)
1. LAMMPS Windows Installer Repository (http://packages.lammps.org/windows.html) -> LAMMPS Binaries Repository: ./legacy/admin/64bit (https://rpm.lammps.org/windows/legacy/admin/64bit/index.html)
2. LAMMPS-64bit-22Dec2022-MSMPI-admin.exe (https://rpm.lammps.org/windows/legacy/admin/64bit/LAMMPS-64bit-22Dec2022-MSMPI-admin.exe)
- LAMMPS Windows Installer Repository -> legacy -> admin -> 64bit -> LAMMPS-64bit-22Dec2022-MSMPI-admin.exe

### Microsoft MPI
1. Microsoft MPI v10.1.2 (https://www.microsoft.com/en-us/download/details.aspx?id=100593)

## Usage
1. input conditions on input.inp
2. click "run_packmol.bat"
3. output.pdb -> Ovito -> output.data (lammps data: [charge format])
4. open output.data and check atomic symbol
5. input atomic symbol for elem on in.lmp
6. click "run_lammps.bat"
