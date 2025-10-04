#!/bin/bash

LOGFILE="mol2crystal_run.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "=== Script started at $(date) ==="

run_step() {
    echo ""
    echo "=== Starting $1 at $(date) ==="
    python3 "$2"
    cp -r optimized_structures_vasp "optimized_structures_vasp_$1"
    cp structure_vs_energy.txt "structure_vs_energy_$1.txt"
    echo "=== Finished $1 at $(date) ==="
    
    if [ -d "valid_structures_postprocess" ]; then
      cp -r valid_structures_postprocess "valid_structures_postprocess_$1"
      mv valid_structures_postprocess valid_structures
    fi
    
    python3 select_data.py
    
    if [ -d "valid_structures" ]; then
      mv valid_structures "valid_structures_$1"
    fi
    
    if [ -d "valid_structures_selected" ]; then
      cp -r valid_structures_selected valid_structures
      cp -r valid_structures_selected "valid_structures_selected_$1"
    fi
}

run_step "gaff_pbc" "mol2crystal_gaff_pbc.py"
#run_step "reaxff" "mol2crystal_reaxff.py"
#run_step "dftb" "mol2crystal_dftb.py"
#run_step "xtb" "mol2crystal_xtb.py"
#run_step "qe" "mol2crystal_qe.py"
#run_step "siesta" "mol2crystal_siesta.py"
#run_step "cp2k" "mol2crystal_cp2k.py"
#run_step "nwchem" "mol2crystal_nwchem.py"

#run_step "reaxff" "postprocess_reaxff.py"
run_step "dftb" "postprocess_dftb.py"
#run_step "xtb" "postprocess_xtb.py"
#run_step "qe" "postprocess_qe.py"
run_step "siesta" "postprocess_siesta.py"
#run_step "cp2k" "postprocess_cp2k.py"
#run_step "nwchem" "postprocess_nwchem.py"

echo ""
echo "=== Script finished at $(date) ==="
