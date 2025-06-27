###############################################################################
#  GP‑driven colour‑mixing visualisation
#  ------------------------------------
#  • Fits three independent Gaussian‑Process regressors (one per RGB channel)
#  • Computes predictions + joint uncertainty on a 2‑D simplex of 3 dyes
#  • Shows two scenes:
#       1) A 3‑D “colour manifold” in RGB space with training points
#       2) A 2‑D heat‑map of model uncertainty on the simplex
#
#  Save this file as `gp_colour_demo.py` and render either scene with
#      manim -pql gp_colour_demo.py GPColourManifold
#      manim -pql gp_colour_demo.py UncertaintySimplex
#  (add ‑pqh / ‑p to bump quality).
###############################################################################

from manim import *                          # Manim CE ≥ 0.18
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

# --------------------------------------------------------------------------- #
# 1.  Training data  (volumes add to 0‒200 µL)                                #
# --------------------------------------------------------------------------- #
X_train = np.array(
    [
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
        [80, 120, 0],
    ],
    dtype=float,
)

Y_train = np.array(
    [
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
        [128.47874, 77.12489, 17.126038],
    ],
    dtype=float,
)

# All GPs are fitted to *scaled* data (helps kernel length‑scales)
MAX_VOL = 200.0          #  μL
X_train_norm = X_train / MAX_VOL          # → within [0,1]
Y_train_norm = Y_train / 255.0            # RGB to [0,1]

# --------------------------------------------------------------------------- #
# 2.  Train three independent GP regressors                                   #
# --------------------------------------------------------------------------- #
kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(0.5, (1e-3, 1e3)) \
         + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-4, 1e-1))

gp_models = [
    GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42 + i)
    for i in range(3)
]
for c in range(3):
    gp_models[c].fit(X_train_norm, Y_train_norm[:, c])

# --------------------------------------------------------------------------- #
# 3.  Create a grid of valid simplex points                                   #
# --------------------------------------------------------------------------- #
def generate_simplex_grid(resolution=50):
    """Return barycentric weights w1,w2,w3 (each (N,)) and cartesian volumes X."""
    w1, w2 = np.meshgrid(
        np.linspace(0, 1, resolution),
        np.linspace(0, 1, resolution)
    )
    w1, w2 = w1.flatten(), w2.flatten()
    w3 = 1.0 - w1 - w2
    mask = w3 >= 0
    w1, w2, w3 = w1[mask], w2[mask], w3[mask]

    # Volumes (for GP input)
    simplex_points = np.stack([w1, w2, w3], axis=1)       # shape (N,3) ‑‑ already normalised
    return w1, w2, w3, simplex_points

w1, w2, w3, GRID_X = generate_simplex_grid(resolution=55)

# --------------------------------------------------------------------------- #
# 4.  Predict mean RGB and joint uncertainty at every grid point              #
# --------------------------------------------------------------------------- #
pred_means = np.zeros((GRID_X.shape[0], 3))
pred_stds  = np.zeros_like(pred_means)

for c in range(3):
    mu, std = gp_models[c].predict(GRID_X, return_std=True)
    pred_means[:, c] = mu
    pred_stds[:, c]  = std

joint_uncertainty = np.linalg.norm(pred_stds, axis=1)      # scalar per grid point

# Keep values handy for scenes
DATA = dict(
    w1=w1,
    w2=w2,
    GRID_X=GRID_X,
    mu_rgb=pred_means,
    joint_unc=joint_uncertainty,
    y_train=Y_train_norm,
)

# --------------------------------------------------------------------------- #
# 5.  Helper utilities                                                        #
# --------------------------------------------------------------------------- #

def rgb_to_hex(rgb):
    r, g, b = (int(255*x) for x in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"

# ─── Scene 1  (colour manifold in RGB space) ────────────────────────────────
class GPColourManifold(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-110 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.06)

        axes = ThreeDAxes(
            x_range=[0, 1, 0.2], y_range=[0, 1, 0.2], z_range=[0, 1, 0.2],
            x_length=6, y_length=6, z_length=6,
            axis_config={"color": GREY_B}
        )

        # axis labels using plain Text
        labels = VGroup(
            Text("Red",   font_size=36).next_to(axes.x_axis.get_end(), DOWN),
            Text("Green", font_size=36).next_to(axes.y_axis.get_end(), RIGHT),
            Text("Blue",  font_size=36).next_to(axes.z_axis.get_end(), UP),
        )

        # predicted manifold points
        dots = VGroup(*[
            Dot3D(axes.c2p(*rgb), radius=0.03,
                color=rgb_to_hex(rgb), stroke_width=0)
            for rgb in DATA["mu_rgb"]
        ])

        train_dots = VGroup(*[
            Dot3D(axes.c2p(*rgb), radius=0.07,
                color=rgb_to_hex(rgb),
                stroke_color=WHITE, stroke_width=1.5)
            for rgb in DATA["y_train"]
        ])

        title = Text("Predicted Colour Manifold", font_size=42).to_edge(UP)

        self.add(axes, labels, dots, train_dots, title)
        self.wait(8)


# ─── Scene 2  (unchanged except colour handling) ───────────────────────────────
class UncertaintySimplex(Scene):
    def construct(self):
        ax = Axes(
            x_range=[0, 1, 0.2], y_range=[0, 1, 0.2],
            x_length=6, y_length=6,
            axis_config={"include_ticks": False, "color": GREY_B},
        ).add_coordinates()

        x_lab = Text("Weight on Dye 1", font_size=30).next_to(ax.x_axis, DOWN)
        y_lab = Text("Weight on Dye 2", font_size=30).next_to(ax.y_axis, LEFT)

        umin, umax = DATA["joint_unc"].min(), DATA["joint_unc"].max()
        dots = VGroup(*[
            Dot(ax.c2p(w1, w2), radius=0.035,
                color=interpolate_color(PURPLE, YELLOW, (u - umin)/(umax - umin)),
                stroke_width=0)
            for w1, w2, u in zip(DATA["w1"], DATA["w2"], DATA["joint_unc"])
        ])

        train_overlay = VGroup(*[
            Dot(ax.c2p(*bary[:2]), radius=0.08,
                fill_opacity=0, stroke_color=RED, stroke_width=2)
            for bary in X_train / MAX_VOL
        ])

        title = Text("Model Uncertainty on Input Simplex", font_size=42).to_edge(UP)

        self.add(ax, dots, train_overlay, x_lab, y_lab, title)
        self.wait(6)