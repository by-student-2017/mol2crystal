#!/bin/bash

python3 mol2crystal_gaff_pbc.py

cp -r optimized_structures_vasp optimized_structures_vasp_gaff_pbc

python3 mol2crystal_reaxff.py

cp -r optimized_structures_vasp optimized_structures_vasp_reaxff

python3 mol2crystal_dftb.py

cp -r optimized_structures_vasp optimized_structures_vasp_dftb

python3 mol2crystal_xtb.py

cp -r optimized_structures_vasp optimized_structures_vasp_xtb

python3 mol2crystal_qe.py

cp -r optimized_structures_vasp optimized_structures_vasp_qe

python3 mol2crystal_siesta.py

cp -r optimized_structures_vasp optimized_structures_vasp_siesta

python3 mol2crystal_cp2k.py

cp -r optimized_structures_vasp optimized_structures_vasp_cp2k

python3 mol2crystal_nwchem.py

cp -r optimized_structures_vasp optimized_structures_vasp_nwchem

