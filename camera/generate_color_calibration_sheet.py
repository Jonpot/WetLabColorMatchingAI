import numpy as np
import matplotlib.pyplot as plt

# Define the Macbeth color patches
MACBETH_24_BGR = [
    ( 68,  82,115),(130,150,194),(157,122, 98),( 67,108, 87),
    (177,128,133),(170,189,103),( 44,126,214),(166, 91, 80),
    ( 99,  90,193),(108, 60, 94),( 64,188,157),( 46,163,224),
    (150, 61, 56),( 73,148, 70),( 60, 54,175),( 31,199,231),
    (149, 86,187),(161,133,  8),(242,243,242),(200,200,200),
    (160,160,160),(121,122,122),( 85, 85, 85),( 52, 52, 52)
]
MACBETH_24_RGB = np.array([[b, g, r] for r, g, b in MACBETH_24_BGR], np.float32)

# Configuration
n_cols = 6
n_rows = 4
patch_size = 0.5  # inches
dpi = 300
fig_width = n_cols * patch_size
fig_height = n_rows * patch_size

# Create figure
fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, n_cols)
ax.set_ylim(0, n_rows)

# Draw patches
for i, color in enumerate(MACBETH_24_RGB):
    col = i % n_cols
    row = i // n_cols
    rect = plt.Rectangle((col, n_rows - 1 - row), 1, 1, facecolor=color/255, edgecolor='none')
    ax.add_patch(rect)

ax.axis('off')

# Save and show
output_path = 'camera/macbeth_chart.png'
fig.savefig(output_path, dpi=dpi)
plt.show()
