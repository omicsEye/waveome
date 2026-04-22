"""
Generates (relative to waveome repo root):
  figures/inducing_prior_ex.png   -- prior GP with evenly-spaced inducing points
  figures/inducing_posterior_ex.png -- posterior after VI with movement arrows

Run from the waveome repo root:
  conda run -n waveome python examples/simulations/make_inducing_figures.py
"""

import os

import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# ── Reproducibility ───────────────────────────────────────────────────────────
np.random.seed(42)
tf.random.set_seed(42)

# ── Output directory (figures/ at repo root) ──────────────────────────────────
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
OUT_DIR = os.path.join(REPO_ROOT, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Simulated data: slow trend everywhere, localised oscillation near X = 3 ──
# The SE kernel will focus inducing points near x=3 where curvature is highest.
N = 1000
X_all = np.linspace(0, 6, N).reshape(-1, 1).astype(np.float64)
x = X_all.ravel()
# Sinusoidal oscillation near x=3, decaying to flat outside [2,4].
# Lower frequency (sin(4x)) keeps the period wide enough for ls=0.5 to capture.
f_true = 3.0 * np.exp(-2.0 * (x - 3) ** 2) * np.sin(4 * x)
Y_all = (f_true + np.random.normal(0, 0.3, N)).reshape(-1, 1)

# ── Shared settings ───────────────────────────────────────────────────────────
X_test = np.linspace(-0.2, 6.2, 400).reshape(-1, 1).astype(np.float64)
N_SAMPLES = 10
M = 7
Z_init = np.linspace(0.2, 5.8, M).reshape(-1, 1).astype(np.float64)

INDUCING_FINAL_COLOR = "#1f77b4"  # blue  – final positions
INDUCING_INIT_COLOR = "#d62728"  # red   – initial positions
ARROW_COLOR = "#d62728"  # red arrows
DATA_COLOR = "#aaaaaa"  # gray dots
MEAN_COLOR = "#2ca02c"  # dark green mean
BAND_COLOR = "#b8e4b8"  # light green band
SAMPLE_COLOR = "#dddddd"  # light gray samples

FIXED_VARIANCE = 3.0
FIXED_LENGTHSCALE = (
    0.5  # resolves sin(4x) (period≈1.6) without needing full domain coverage
)
NOISE_VARIANCE = 0.1


def make_kernel():
    return gpflow.kernels.SquaredExponential(
        variance=FIXED_VARIANCE, lengthscales=FIXED_LENGTHSCALE
    )


# ── (a) Prior: samples from kernel MVN; evenly-spaced inducing points at f=0 ─
# Use a longer lengthscale for the prior figure so samples look like smooth GP
# draws rather than high-frequency noise (illustrative only).
kernel_prior = gpflow.kernels.SquaredExponential(
    variance=FIXED_VARIANCE, lengthscales=FIXED_LENGTHSCALE
)
K_prior = kernel_prior.K(X_test).numpy()
K_prior += 1e-6 * np.eye(len(X_test))
std_prior = np.sqrt(np.diag(K_prior))
samples_prior = np.random.multivariate_normal(np.zeros(len(X_test)), K_prior, N_SAMPLES)

fig, ax = plt.subplots(figsize=(7, 3.5))
for s in samples_prior:
    ax.plot(X_test, s, color=SAMPLE_COLOR, lw=0.8, zorder=1)
ax.fill_between(
    X_test.ravel(),
    -1.96 * std_prior,
    1.96 * std_prior,
    color=BAND_COLOR,
    alpha=0.7,
    zorder=2,
)
ax.plot(
    X_test, np.zeros(len(X_test)), color=MEAN_COLOR, lw=2, zorder=3, label="GP mean"
)
ax.scatter(
    X_all, Y_all, color=DATA_COLOR, s=4, alpha=0.3, zorder=4, label="observed (N=1000)"
)
ax.scatter(
    Z_init,
    np.zeros(M),
    marker="*",
    s=200,
    color=INDUCING_INIT_COLOR,
    zorder=5,
    label="inducing points",
)
ax.set_title("(a) Prior GP with evenly spaced inducing points", fontsize=11)
ax.set_xlabel("X", fontsize=10)
ax.set_ylabel("f(X)", fontsize=10)
ax.set_xlim(-0.2, 6.2)
ax.legend(fontsize=8, loc="upper left")
plt.tight_layout()
out_prior = os.path.join(OUT_DIR, "inducing_prior_ex.png")
plt.savefig(out_prior, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out_prior}")

# ── (b) Posterior: optimise via VI; draw movement arrows ─────────────────────
model_post = gpflow.models.SVGP(
    kernel=make_kernel(),
    likelihood=gpflow.likelihoods.Gaussian(variance=NOISE_VARIANCE),
    inducing_variable=Z_init.copy(),
    num_data=N,
)


# Natural gradient updates q_mu/q_sqrt (variational params);
# Adam updates kernel hyperparameters + inducing locations.
# Freeze variational params from Adam so each optimizer owns its variables.
gpflow.set_trainable(model_post.q_mu, False)
gpflow.set_trainable(model_post.q_sqrt, False)

natgrad_opt = gpflow.optimizers.NaturalGradient(gamma=0.1)
adam_opt = tf.optimizers.Adam(learning_rate=0.01)
variational_vars = [(model_post.q_mu, model_post.q_sqrt)]


@tf.function
def step():
    natgrad_opt.minimize(lambda: -model_post.elbo((X_all, Y_all)), variational_vars)
    adam_opt.minimize(
        lambda: -model_post.elbo((X_all, Y_all)), model_post.trainable_variables
    )


N_STEPS = 10000
for i in range(N_STEPS):
    step()
    if (i + 1) % 1000 == 0:
        elbo = model_post.elbo((X_all, Y_all)).numpy()
        print(f"  step {i+1}/{N_STEPS}  ELBO={elbo:.1f}")

# Posterior predictions
mean_post, var_post = model_post.predict_y(X_test)
mean_post = mean_post.numpy().ravel()
std_post = np.sqrt(var_post.numpy().ravel())
samples_post = model_post.predict_f_samples(X_test, N_SAMPLES).numpy()

Z_final = model_post.inducing_variable.Z.numpy().ravel()
# GP posterior mean at initial and final inducing locations
Z_init_mean, _ = model_post.predict_f(Z_init)
Z_init_mean = Z_init_mean.numpy().ravel()
Z_final_mean, _ = model_post.predict_f(Z_final.reshape(-1, 1))
Z_final_mean = Z_final_mean.numpy().ravel()

fig, ax = plt.subplots(figsize=(7, 3.5))

# GP samples
for s in samples_post[:, :, 0]:
    ax.plot(X_test, s, color=SAMPLE_COLOR, lw=0.8, zorder=1)

# Mean and band
ax.fill_between(
    X_test.ravel(),
    mean_post - 1.96 * std_post,
    mean_post + 1.96 * std_post,
    color=BAND_COLOR,
    alpha=0.7,
    zorder=2,
)
ax.plot(X_test, mean_post, color=MEAN_COLOR, lw=2, zorder=3, label="GP mean")

# Data
ax.scatter(
    X_all, Y_all, color=DATA_COLOR, s=4, alpha=0.3, zorder=4, label="observed (N=1000)"
)

# Initial inducing positions (open red stars)
ax.scatter(
    Z_init.ravel(),
    Z_init_mean,
    marker="*",
    s=200,
    facecolors="none",
    edgecolors=INDUCING_INIT_COLOR,
    linewidths=1.5,
    zorder=6,
    label="initial inducing points",
)

# Final inducing positions (filled blue stars)
ax.scatter(
    Z_final,
    Z_final_mean,
    marker="*",
    s=200,
    color=INDUCING_FINAL_COLOR,
    zorder=7,
    label="final inducing points",
)

# Arrows from initial to final positions
for xi, yi, xf, yf in zip(Z_init.ravel(), Z_init_mean, Z_final, Z_final_mean):
    ax.annotate(
        "",
        xy=(xf, yf),
        xytext=(xi, yi),
        arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR, lw=1.4, mutation_scale=12),
        zorder=8,
    )

ax.set_title("(b) Posterior GP after inducing point optimisation (VI)", fontsize=11)
ax.set_xlabel("X", fontsize=10)
ax.set_ylabel("f(X)", fontsize=10)
ax.set_xlim(-0.2, 6.2)
ax.legend(fontsize=8, loc="upper left")
plt.tight_layout()
out_post = os.path.join(OUT_DIR, "inducing_posterior_ex.png")
plt.savefig(out_post, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out_post}")

print(f"Initial inducing locations: {np.round(Z_init.ravel(), 2)}")
print(f"Final   inducing locations: {np.round(Z_final, 2)}")
