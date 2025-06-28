###############################################################################
#  Gaussian‑Process color‑mixing • Bayesian Optimisation  (no point‑cloud)   #
###############################################################################
from manim import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

# --------------------------------------------------------------------------- #
# Tunables                                                                    #
# --------------------------------------------------------------------------- #
GRID_RES = 35          # 35 → ≈300 dots; lower for faster renders
DOT_RAD  = 0.03
# --------------------------------------------------------------------------- #

# 1 ▸ DATA (removed last recipe) ------------------------------------------- #
X_FULL = np.array([
    [32, 0, 168],[200, 0, 0],[0, 0, 200],[105, 95, 0],[146, 34, 20],
    [123, 29, 48],[0, 32, 168],[100, 0, 100],[0, 135, 65],[121, 79, 0]],
    dtype=float)

Y_FULL = np.array([
    [46.82222,101.99156,113.28391],[130.056,49.08257,34.05690],
    [45.58934,133.04184,142.25505],[123.73465,81.86804,56.66032],
    [119.37567,62.05669,52.30501],[94.7716,52.81526,38.19274],
    [56.27534,123.00963,88.06221],[76.88751,75.82702,81.79092],
    [113.50485,148.1822,53.88678],[157.34193,81.05773,15.48914]],
    dtype=float)

MAX_VOL   = 200.0
X_FULL_N  = X_FULL / MAX_VOL
Y_FULL_N  = Y_FULL / 255.0

# 2 ▸ helpers --------------------------------------------------------------- #
def simplex_grid(res=35):
    w1, w2 = np.meshgrid(np.linspace(0,1,res), np.linspace(0,1,res))
    w1, w2 = w1.flatten(), w2.flatten()
    w3 = 1 - w1 - w2
    mask = w3 >= -1e-6
    return np.stack([w1[mask], w2[mask], w3[mask]], axis=1)

GRID = simplex_grid(GRID_RES)

def fit_gp(x, y):
    ker = ConstantKernel(1.0,(1e-3,1e3))*RBF(0.5,(1e-3,1e3)) \
        + WhiteKernel(0.01,(1e-4,1e-1))
    models = [GaussianProcessRegressor(kernel=ker, normalize_y=True,
                                       random_state=11+i).fit(x, y[:,i])
              for i in range(3)]
    mu  = np.column_stack([m.predict(GRID) for m in models])
    std = np.column_stack([m.predict(GRID, return_std=True)[1] for m in models])
    return mu, np.linalg.norm(std, axis=1)

STEPS=[]
for k in range(len(X_FULL)+1):
    curX, curY = X_FULL_N[:k], Y_FULL_N[:k]
    mu, unc = fit_gp(curX, curY) if k else \
              (np.zeros_like(GRID), np.ones(len(GRID))*0.45)
    STEPS.append(dict(mu=mu, unc=unc,
                      next=X_FULL_N[k] if k < len(X_FULL) else None))
N_STEPS = len(STEPS)

def hex_from_rgb(rgb):
    return f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"

VIRIDIS_STOPS=[PURPLE, BLUE, GREEN, YELLOW]
def viridis(val, vmin, vmax):
    alpha=0.0 if vmax<=vmin else (val-vmin)/(vmax-vmin)
    alpha=max(0,min(1,alpha))
    return color_gradient(VIRIDIS_STOPS,256)[int(alpha*255)]

# --------------------------------------------------------------------------- #
# Scene                                                                       #
# --------------------------------------------------------------------------- #
class ColorBayesOptDemo(ThreeDScene):
    def construct(self):
        # Camera & axes
        self.set_camera_orientation(phi=65*DEGREES, theta=-110*DEGREES)
        self.begin_ambient_camera_rotation(rate=0.05)
        ax3d = ThreeDAxes([0,1,0.2],[0,1,0.2],[0,1,0.2],
                          x_length=5, y_length=5, z_length=5,
                          axis_config={"color": GREY_B}).shift(RIGHT*3)
        ax2d = Axes([0,1,0.2],[0,1,0.2], x_length=5, y_length=5,
                    axis_config={"include_ticks":False,"color":GREY_B}).shift(LEFT*3)

        self.add(ax3d, ax2d,
                 Text("R").next_to(ax3d.x_axis.get_end(),DOWN),
                 Text("G").next_to(ax3d.y_axis.get_end(),RIGHT),
                 Text("B").next_to(ax3d.z_axis.get_end(),UP),
                 Text("w₁").next_to(ax2d.x_axis,DOWN),
                 Text("w₂").next_to(ax2d.y_axis,LEFT),
                 Text("Gaussian‑Process Bayesian Optimisation\nColor‑Mixing Demo",
                      font_size=38).to_edge(UP))

        # Trackers
        idx   = ValueTracker(0)   # optimisation step
        alpha = ValueTracker(0)   # morph 0→1
        fold  = ValueTracker(0)   # fold demo

        simplex_world = np.array([ax2d.c2p(*p[:2]) for p in GRID])

        # LEFT cloud as individual dots
        dots_left = VGroup(*[Dot(radius=DOT_RAD) for _ in GRID])
        def left_upd(group):
            k=int(idx.get_value())
            unc=STEPS[k]['unc']
            if k<N_STEPS-1:
                unc=(1-alpha.get_value())*unc + alpha.get_value()*STEPS[k+1]['unc']
            umin,umax=unc.min(),unc.max()
            mu=STEPS[k]['mu']
            if k<N_STEPS-1:
                mu=(1-alpha.get_value())*mu + alpha.get_value()*STEPS[k+1]['mu']

            for i,d in enumerate(group):
                # position
                if fold.get_value()>0:
                    rgb_world=ax3d.c2p(*mu[i])
                    pos = (1-fold.get_value())*simplex_world[i] + fold.get_value()*rgb_world
                else:
                    pos = simplex_world[i]
                d.move_to(pos)
                # color
                d.set_color(viridis(unc[i], umin, umax))
        dots_left.add_updater(left_upd)
        self.add(dots_left)

        # RIGHT cloud as individual dots
        dots_right = VGroup(*[Dot3D(radius=DOT_RAD) for _ in GRID])
        def right_upd(group):
            k=int(idx.get_value())
            mu=STEPS[k]['mu']
            if k<N_STEPS-1:
                mu=(1-alpha.get_value())*mu + alpha.get_value()*STEPS[k+1]['mu']
            for i,d in enumerate(group):
                d.move_to(ax3d.c2p(*mu[i]))
                d.set_color(hex_from_rgb(mu[i]))
                d.set_opacity(1-fold.get_value())
        dots_right.add_updater(right_upd)
        self.add(dots_right)

        # Training dots
        self._add_training_dots(ax2d, ax3d, idx)

        # Pulse
        self._add_pulse(ax2d, idx)

        # optimisation loop
        for _ in range(len(X_FULL)):
            self.wait(0.8)
            self.play(alpha.animate.set_value(1), run_time=0.9)
            self.play(idx.animate.increment_value(1), run_time=0.1)
            alpha.set_value(0)

        self.wait(1)

        for _ in range(3):
            self.play(fold.animate.set_value(1), run_time=2)
            self.play(fold.animate.set_value(0), run_time=2)

        self.wait(1)

    # helpers
    def _add_training_dots(self, ax2d, ax3d, idx):
        left,right=VGroup(),VGroup()
        for j,(w,rgb) in enumerate(zip(X_FULL_N,Y_FULL_N)):
            d2=Dot(ax2d.c2p(w[0],w[1]),radius=0.09, fill_opacity=0,
                   stroke_color=RED,stroke_width=2)
            d3=Dot3D(ax3d.c2p(*rgb),radius=0.08,
                     color=hex_from_rgb(rgb),
                     stroke_color=WHITE,stroke_width=1.2)
            for m in (d2,d3):
                m.add_updater(lambda m_,jj=j:
                    m_.set_opacity(1 if int(idx.get_value())>jj else 0))
            left.add(d2); right.add(d3)
        self.add(left,right)

    def _add_pulse(self, ax2d, idx):
        pulse=Dot(radius=0.15,color=YELLOW,stroke_width=0)
        def upd(m):
            k=int(idx.get_value())
            if k>=len(X_FULL): m.set_opacity(0); return
            w=STEPS[k]['next']; m.move_to(ax2d.c2p(w[0],w[1]))
            m.set_opacity(1); m.scale(1+0.25*np.sin(TAU*self.time))
        pulse.add_updater(upd); self.add(pulse)
