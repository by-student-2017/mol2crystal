#!/usr/bin/env python3

# user settings top %
top_percent = 30

import pandas as pd

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
n_top = int(len(df) * (top_percent / 100))
top_percent_df = df.sort_values(by='Total_Energy').head(n_top)

# Sort the selected structures by Total Energy ascending
sorted_by_energy = top_percent_df.sort_values(by='Total_Energy')

# Sort the selected structures by Density descending
sorted_by_density = top_percent_df.sort_values(by='Density', ascending=False)

# Output both sorted results
print("Sorted by Total Energy (ascending):")
print(sorted_by_energy[['POSCAR', 'Total_Energy', 'Density']])

#print("\nSorted by Density (descending):")
#print(sorted_by_density[['POSCAR', 'Total_Energy', 'Density']])
