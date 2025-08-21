#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor

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

# Plotting with interactive annotations
fig, ax = plt.subplots(figsize=(10, 7))
sc = ax.scatter(x_density, y_energy, c=y_energy, cmap='viridis', alpha=0.8, edgecolors='k')
ax.set_xlabel("Density [g/cm^3]")
ax.set_ylabel("Energy [KJ/mol/atom]")
ax.set_title("Energy vs Density of Valid Structures")
ax.grid(True)

# Add interactive cursor
cursor = Cursor(ax, useblit=True, color='red', linewidth=1)

# Add annotation box
annot = ax.annotate("", xy=(0,0), xytext=(10,10), textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

# Function to update annotation
def update_annot(ind):
    idx = ind["ind"][0]
    pos = sc.get_offsets()[idx]
    annot.xy = pos
    text = f"{labels[idx]}\nDensity: {pos[0]:.3f}\nEnergy: {pos[1]:.3f}"
    annot.set_text(text)
    annot.get_bbox_patch().set_facecolor('lightyellow')
    annot.get_bbox_patch().set_alpha(0.9)

# Hover event
def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)
plt.tight_layout()
plt.savefig("energy_vs_density_plot_interactive.png")
plt.show()
