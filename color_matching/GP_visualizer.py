import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # necessary for 3D plotting

def plot_gp_predictions(gp_models, X_train, Y_train):
    # --- 1. Generate points on the valid input simplex (d1+d2+d3 = 200) ---
    # This technique uses barycentric coordinates to create a grid on the triangle.

    print("Generating a grid of points on the valid input simplex...")
    resolution = 50  # Increase for a smoother plot, decrease for speed
    max_vol = 200.0

    # Create a 2D grid of weights
    w1, w2 = np.meshgrid(
        np.linspace(0, 1, resolution),
        np.linspace(0, 1, resolution)
    )
    w1 = w1.flatten()
    w2 = w2.flatten()

    # The third weight is determined by the other two
    w3 = 1 - w1 - w2

    # Filter out points outside the triangle (where w3 would be negative)
    valid_indices = w3 >= 0
    w1, w2, w3 = w1[valid_indices], w2[valid_indices], w3[valid_indices]

    # Define the vertices of our input triangle
    v1 = np.array([max_vol, 0, 0])
    v2 = np.array([0, max_vol, 0])
    v3 = np.array([0, 0, max_vol])

    # Calculate the 3D points in dye-volume space
    simplex_points = w1[:, np.newaxis] * v1 + w2[:, np.newaxis] * v2 + w3[:, np.newaxis] * v3
    print(f"Generated {len(simplex_points)} valid points to visualize.")


    # --- 2. Predict Colors and Uncertainty for Each Point on the Simplex ---

    print("Predicting color and uncertainty for each point...")
    predicted_means = np.zeros_like(simplex_points)
    predicted_stds = np.zeros_like(simplex_points)

    for c in range(3):
        mean, std = gp_models[c].predict(simplex_points, return_std=True)
        predicted_means[:, c] = mean
        predicted_stds[:, c] = std

    # Calculate a single uncertainty value per point (Euclidean norm of stds)
    joint_uncertainty = np.linalg.norm(predicted_stds, axis=1)


    # --- 3. Create the Visualizations ---

    print("Generating plots...")
    fig = plt.figure(figsize=(18, 8))
    fig.suptitle('GP Model Behavior on the Constrained Input Space (Total Volume = 200uL)', fontsize=16)

    # ----- PLOT 1: PREDICTED COLOR MANIFOLD (in RGB Space) -----
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')

    # The predicted colors, with each point colored by its own value
    # Normalize RGB from [0, 255] to [0, 1] for matplotlib
    plot_colors = np.clip(predicted_means / 255.0, 0, 1)
    ax1.scatter(
        predicted_means[:, 0], predicted_means[:, 1], predicted_means[:, 2],
        c=plot_colors,
        marker='.'
    )

    # Overlay the actual training data points for context
    ax1.scatter(
        Y_train[:, 0], Y_train[:, 1], Y_train[:, 2],
        c=Y_train / 255.0,
        edgecolor='black',
        s=100,
        depthshade=False,
        label='Training Data'
    )

    ax1.set_title('Predicted Color Manifold')
    ax1.set_xlabel('Red Channel')
    ax1.set_ylabel('Green Channel')
    ax1.set_zlabel('Blue Channel')
    ax1.legend()
    ax1.grid(True)


    # ----- PLOT 2: UNCERTAINTY PLOT (on the Input Simplex) -----
    ax2 = fig.add_subplot(1, 2, 2)

    # We use the barycentric weights (w1, w2) to plot the 2D triangle
    # The color of each point is its associated uncertainty
    scatter_uncertainty = ax2.scatter(
        w1, w2, c=joint_uncertainty, cmap='viridis', s=20
    )

    # Convert the original X_train points to barycentric coordinates to overlay them
    # w1 = d1/max_vol, w2 = d2/max_vol
    x_train_barycentric = X_train / max_vol
    ax2.scatter(
        x_train_barycentric[:, 0], x_train_barycentric[:, 1],
        facecolor='none',
        edgecolor='red',
        s=100,
        label='Training Points'
    )

    # Add a color bar to show the uncertainty scale
    plt.colorbar(scatter_uncertainty, ax=ax2, label='Joint Uncertainty (Std. Dev.)')

    ax2.set_title('Model Uncertainty on Input Simplex')
    ax2.set_xlabel('Weight on Dye 1')
    ax2.set_ylabel('Weight on Dye 2')
    ax2.legend()
    ax2.grid(True)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    # Add text to label the corners of the triangle
    ax2.text(1.0, -0.05, '100% Dye 1', ha='center')
    ax2.text(-0.05, 1.0, '100% Dye 2', va='center', ha='right')
    ax2.text(0, -0.05, '100% Dye 3', ha='center')

    return fig


if __name__ == "__main__":
    # Updated training data with more points
    X_train_old = np.array([
        [32, 0, 168],
        [200, 0, 0],
        [0, 0, 200],
        [105, 95, 0],
        [146, 34, 20],
        [123, 29, 48],
        [0, 32, 168],
        [100, 0, 100],
        [0, 135, 65],
        [121, 79, 0],
        [80, 120, 0]
    ], dtype=np.uint8)

    Y_train_old = np.array([
        [46.82222, 101.99156, 113.283905],
        [130.056, 49.082573, 34.056904],
        [45.589336, 133.04184, 142.25505],
        [123.73465, 81.86804, 56.66032],
        [119.37567, 62.056694, 52.305008],
        [94.7716, 52.81526, 38.19274],
        [56.275337, 123.00963, 88.06221],
        [76.887505, 75.82702, 81.79092],
        [113.50485, 148.1822, 53.886776],
        [157.34193, 81.05773, 15.489141],
        [128.47874, 77.12489, 17.126038]
    ], dtype=np.float32)

    X_train = np.array([
        [0, 80, 120],
        [20, 180, 0],
        [200, 0, 0],
        [20, 180, 0],
        [20, 100, 80],
        [0, 80, 120],
        [60, 140, 0]
    ], dtype=np.uint8)

    Y_train = np.array([
        [74.13977, 121.73086, 56.07698],
        [109.75567, 80.83364, 0.0],
        [122.94749, 48.381863, 33.310375],
        [169.26698, 149.90884, 51.39355],
        [94.0433, 116.50242, 42.533962],
        [77.744705, 135.48436, 59.44623],
        [115.36029, 70.30394, 1.9244199]
    ], dtype=np.float32)


    # Fit three GPs (one for each channel) using full 3D input
    #kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(50.0, (1e-3, 1e3)) + WhiteKernel()
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(50.0, (1e-3, 1e3)) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-3, 9e-2))
    gp_models = [
        GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42 + i)
        for i in range(3)
    ]
    for c in range(3):
        gp_models[c].fit(X_train, Y_train[:, c])


    fig = plot_gp_predictions(gp_models, X_train, Y_train)

    plt.tight_layout()
    plt.show()