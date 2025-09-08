#!/usr/bin/env python3

# user settings selected %
selected_percent = 10

# Minimum number to display (takes precedence over %).
min_num_shows = 6

import os
import shutil
import pandas as pd

if (os.path.exists('valid_structures_selected_old')):
    shutil.rmtree( 'valid_structures_selected_old')   

if (os.path.exists('valid_structures_selected')):
    os.rename(     'valid_structures_selected','valid_structures_selected_old')

# Read the file
with open("structure_vs_energy.txt", "r") as f:
    lines = f.readlines()

# Filter out comment lines and parse the data
data = [line.strip().split() for line in lines if not line.startswith('#')]

# Create DataFrame
columns = ['POSCAR', 'Relative_Energy', 'Total_Energy', 'Density', 'Num_Atoms', 'Volume']
df = pd.DataFrame(data, columns=columns)

# Convert numeric columns to float
for col in ['Relative_Energy', 'Total_Energy', 'Density', 'Num_Atoms', 'Volume']:
    df[col] = df[col].astype(float)

# Remove prefix from POSCAR
df['POSCAR'] = df['POSCAR'].str.replace('optimized_structures_vasp/', '', regex=False)

# Sort by Total Energy and select bottom X%
n_selected = max(min_num_shows, int(len(df) * (selected_percent / 100)))
selected_percent_df = df.sort_values(by='Total_Energy').head(n_selected)

# Sort the selected structures by Total Energy ascending
sorted_by_energy = selected_percent_df.sort_values(by='Total_Energy')

# Sort the selected structures by Density descending
sorted_by_density = selected_percent_df.sort_values(by='Density', ascending=False)

# Output both sorted results
print("Sorted by Total Energy (ascending):")
print(sorted_by_energy[['POSCAR', 'Total_Energy', 'Density']])

#print("\nSorted by Density (descending):")
#print(sorted_by_density[['POSCAR', 'Total_Energy', 'Density']])

# Create output directory
output_dir = "valid_structures_selected"
os.makedirs(output_dir, exist_ok=True)

# Copy corresponding files to valid_structures_selected directory
for poscar_file in selected_percent_df['POSCAR']:
    src_path = os.path.join("optimized_structures_vasp", poscar_file)
    dst_path = os.path.join(output_dir, poscar_file)
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
    else:
        print(f"Warning: {src_path} not found.")

print(f"\nCopied {len(selected_percent_df)} files to '{output_dir}' directory.")
