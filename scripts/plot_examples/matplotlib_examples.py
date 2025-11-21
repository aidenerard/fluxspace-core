"""
Matplotlib Visual Examples
Run this script to see colormap, heatmap, and anomaly map visualizations.
"""

import matplotlib.pyplot as plt
import numpy as np

# Set style for better-looking plots
plt.style.use('default')

print("Creating matplotlib examples...")
print("=" * 50)

# ============================================================================
# EXAMPLE 1: Different Colormaps (like 'seismic' in your script)
# ============================================================================
print("\n1. Creating scatter plots with different colormaps...")

# Generate data with a pattern
x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)  # Create a pattern

# Flatten for scatter plot
x_flat = X.flatten()
y_flat = Y.flatten()
z_flat = Z.flatten()

colormaps = ['seismic', 'viridis', 'plasma', 'coolwarm']
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for i, cmap in enumerate(colormaps):
    sc = axes[i].scatter(x_flat, y_flat, c=z_flat, cmap=cmap, s=50, edgecolors='k', linewidth=0.3)
    axes[i].set_title(f'Colormap: {cmap}', fontsize=12, fontweight='bold')
    axes[i].set_xlabel('X')
    axes[i].set_ylabel('Y')
    axes[i].set_aspect('equal')
    plt.colorbar(sc, ax=axes[i], label='Value')

plt.suptitle('Different Colormaps Comparison', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('example_1_colormaps.png', dpi=150)
print("   Saved: example_1_colormaps.png")
plt.close()

# ============================================================================
# EXAMPLE 2: Heatmap (2D data visualization)
# ============================================================================
print("\n2. Creating heatmap...")

# Generate 2D data
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = np.exp(-(X**2 + Y**2) / 2)  # Gaussian-like pattern

plt.figure(figsize=(10, 8))
im = plt.imshow(Z, extent=[-5, 5, -5, 5], origin='lower', 
                cmap='hot', interpolation='bilinear')
plt.colorbar(im, label='Intensity')
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.title('Heatmap: 2D Gaussian Pattern', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('example_2_heatmap.png', dpi=150)
print("   Saved: example_2_heatmap.png")
plt.close()

# ============================================================================
# EXAMPLE 3: Anomaly Map (similar to your script)
# ============================================================================
print("\n3. Creating anomaly map (similar to your script)...")

# Simulate magnetic field data
np.random.seed(42)
x = np.linspace(0, 10, 50)
y = np.linspace(0, 10, 50)
X, Y = np.meshgrid(x, y)

# Create a pattern with some anomalies
Z = 50 + 10 * np.sin(X) * np.cos(Y) + np.random.normal(0, 2, X.shape)
# Add some anomalies
Z[20:25, 20:25] += 30  # Hot spot
Z[35:40, 35:40] -= 25  # Cold spot

# Flatten for scatter plot
x_flat = X.flatten()
y_flat = Y.flatten()
z_flat = Z.flatten()

# Normalize for color mapping
z_min, z_max = z_flat.min(), z_flat.max()
z_norm = (z_flat - z_min) / (z_max - z_min)

plt.figure(figsize=(10, 8))
sc = plt.scatter(x_flat, y_flat, c=z_norm, cmap='seismic', 
                 s=100, edgecolor='k', linewidth=0.5, alpha=0.8)
plt.colorbar(sc, label='Normalized Anomaly Value')
plt.xlabel('X (m)', fontsize=12)
plt.ylabel('Y (m)', fontsize=12)
plt.title('Magnetic Anomaly Map (Example)', fontsize=14, fontweight='bold')
plt.gca().set_aspect('equal', 'box')
plt.tight_layout()
plt.savefig('example_3_anomaly_map.png', dpi=150)
print("   Saved: example_3_anomaly_map.png")
plt.close()

print("\n" + "=" * 50)
print("All examples created successfully!")
print("Check the generated PNG files to see the visualizations.")
print("\nKey matplotlib functions demonstrated:")
print("  - plt.scatter()    : Scatter plots with colors")
print("  - plt.imshow()     : Heatmaps/2D images")
print("  - plt.colorbar()   : Color scales")
print("  - plt.subplots()   : Multiple plots in one figure")
print("  - plt.figure()     : Create new figure")
print("  - plt.savefig()    : Save plot to file")
