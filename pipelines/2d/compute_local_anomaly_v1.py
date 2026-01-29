import pandas as pd              # easy to use CSV files and read tabular data
import numpy as np               # fast math on arrays (distances)
import matplotlib.pyplot as plt  # plotting module (heatmap)

# ---- PARAMETERS YOU CAN TUNE ----
CSV_PATH = "mag_data.csv"      # input file
RADIUS = 0.3                   # neighborhood radius, in same units as x,y (cm or m)

# --------------------------------

# 1) Load CSV
df = pd.read_csv(CSV_PATH)     # loads into pandas DataFrame, each column (x, y, B_total) is easily accessible

# Expect columns: x, y, B_total
coords = df[["x", "y"]].values   # tales x,y columns from df and converts into a NumPY array, shape (N, 2). each row = [x_i,y_i]
B = df["B_total"].values         # NumPy array of length N, magnetic field value at each point, shape (N,)

N = len(df)
anomalies = np.zeros(N)

# 2) For each point, find neighbors within RADIUS and compute local anomaly
for i in range(N):
    xi, yi = coords[i]
    # Compute distance from this point to all others
    dx = coords[:, 0] - xi
    dy = coords[:, 1] - yi
    dist = np.sqrt(dx*dx + dy*dy)

    # Boolean mask for neighbors; include everything within RADIUS, exclude self
    neighbor_mask = (dist <= RADIUS) & (dist > 0)

    # If no neighbors (shouldn’t happen in a decent grid), fall back to global mean
    if not np.any(neighbor_mask):
        local_mean = B.mean()
    else:
        local_mean = B[neighbor_mask].mean()

    anomalies[i] = B[i] - local_mean

# 3) Normalize anomaly values for color mapping
# You can use absolute anomaly if you only care about magnitude:
# anomalies_for_color = np.abs(anomalies)
anomalies_for_color = anomalies  # keep sign for now

amin = anomalies_for_color.min()
amax = anomalies_for_color.max()
# Avoid division by zero
if amax - amin < 1e-9:
    norm = np.zeros_like(anomalies_for_color)
else:
    norm = (anomalies_for_color - amin) / (amax - amin)

# 4) Attach anomaly back into the DataFrame (optional, for exporting to QGIS later)
df["local_anomaly"] = anomalies
df["local_anomaly_norm"] = norm

# Save to CSV if you want to use in QGIS or other tools
df.to_csv("mag_data_with_anomaly.csv", index=False)

# 5) Quick visualization (scatter "heatmap")
plt.figure(figsize=(6, 5))
sc = plt.scatter(
    df["x"], df["y"],
    c=df["local_anomaly_norm"],
    cmap="seismic",   # blue-white-red; you can try 'plasma', 'viridis', etc.
    s=120,
    edgecolor="k"
)

plt.colorbar(sc, label="Normalized local anomaly")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Magnetic Local Anomaly Map")
plt.gca().set_aspect("equal", "box")
plt.tight_layout()
plt.show()
