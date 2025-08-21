#!/usr/bin/env python3
import matplotlib.pyplot as plt

# Read data from structure_vs_energy.txt
x_density = []
y_energy = []
labels = []

with open("structure_vs_energy.txt", "r") as file:
    for line in file:
        parts = line.strip().split()
        if len(parts) >= 3:
            label = parts[0].split("/")[-1]
            energy = float(parts[1])
            density = float(parts[2])
            x_density.append(density)
            y_energy.append(energy)
            labels.append(label)

# Plotting with color map
plt.figure(figsize=(10, 7))
sc = plt.scatter(x_density, y_energy, c=y_energy, cmap='viridis', alpha=0.8, edgecolors='k')
#plt.colorbar(sc, label="Energy [KJ/mol/atom]")
plt.xlabel("Density [g/cm^3]")
plt.ylabel("Energy KJ/mol/atom]")
plt.title("Energy vs Density of Valid Structures")
plt.grid(True)

## Annotate outliers (energy > 100)
#for i, (x, y) in enumerate(zip(x_density, y_energy)):
#    if y > 100:
#        plt.annotate(labels[i], (x, y), textcoords="offset points", xytext=(5,5), ha='left', fontsize=8)

plt.tight_layout()
plt.savefig("energy_vs_density_plot_from_file.png")
plt.show()
