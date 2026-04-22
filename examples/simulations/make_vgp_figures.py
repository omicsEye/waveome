"""
Generates a single combined variational GP illustration figure:
  figures/vgp_ex.png

Four-panel, column-major layout:

  (a) [top-left]     Single observation: GP prior × NB likelihood → non-Gaussian posterior
  (b) [bottom-left]  Single observation: true posterior vs variational Gaussian q(f)
  (c) [top-right]    Full function: split-violin glyphs at each x_i —
                       left half = prior p(f) [gray], right half = q(f_i) [blue/red]
  (d) [bottom-right] Full function: NB count observations on the count scale,
                       linking the smooth latent GP to discrete overdispersed data

Pure numpy / scipy — no GPflow required.

Run from the waveome repo root:
  conda run -n waveome python examples/simulations/make_vgp_figures.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, optimize, stats
from scipy.ndimage import gaussian_filter1d
from scipy.stats import nbinom as sp_nbinom

# ── Output directory ───────────────────────────────────────────────────────────
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
OUT_DIR = os.path.join(REPO_ROOT, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Colours (match make_inducing_figures.py) ───────────────────────────────────
PRIOR_COLOR = "#aaaaaa"  # gray   – GP prior
TRUE_COLOR = "#d62728"  # red    – true posterior / highlighted glyph
VARI_COLOR = "#1f77b4"  # blue   – variational posterior
LIKE_COLOR = "#ff7f0e"  # orange – likelihood
MEAN_COLOR = "#2ca02c"  # green  – GP posterior mean
BAND_COLOR = "#b8e4b8"  # light green – uncertainty band
DATA_COLOR = "#555555"  # dark gray – observed data points

# ── Shared model parameters ────────────────────────────────────────────────────
TAU = 2.5  # GP prior std on latent log-rate f
R = 2.0  # NB dispersion (r → ∞ → Poisson; smaller = more overdispersed)
Y_OBS = 5  # single observed count for panels (a) and (b)

np.random.seed(42)


def log_nb_lik(f, y, r):
    """Log NB likelihood (log link, mean=exp(f)), terms depending on f only."""
    return y * f - (y + r) * np.log(r + np.exp(f))


N_GH = 40
gh_pts, gh_wts = np.polynomial.hermite.hermgauss(N_GH)


def fit_variational(y_i, tau, r, x0=None):
    """Return (μ, σ) of the optimal q(f) = N(μ,σ²) for one observation y_i."""
    x0 = x0 if x0 is not None else [np.log(max(y_i, 0.5)), np.log(0.8)]

    def neg_elbo(params):
        mu, log_sigma = params
        sigma = np.exp(log_sigma)
        f_samp = mu + np.sqrt(2) * sigma * gh_pts
        E_loglik = np.dot(gh_wts, log_nb_lik(f_samp, y_i, r)) / np.sqrt(np.pi)
        E_logprior = (
            -0.5 * np.log(2 * np.pi * tau**2) - 0.5 * (mu**2 + sigma**2) / tau**2
        )
        H_q = stats.norm(mu, sigma).entropy()
        return -(E_loglik + E_logprior + H_q)

    res = optimize.minimize(neg_elbo, x0=x0, method="L-BFGS-B")
    return res.x[0], np.exp(res.x[1])


# ══════════════════════════════════════════════════════════════════════════════
# Data for panels (a) & (b): single observation
# ══════════════════════════════════════════════════════════════════════════════
f_grid = np.linspace(-3, 8, 3000)
log_prior = stats.norm.logpdf(f_grid, 0, TAU)
log_lik_grid = log_nb_lik(f_grid, Y_OBS, R)
log_unnorm = log_lik_grid + log_prior
log_unnorm -= log_unnorm.max()
unnorm_post = np.exp(log_unnorm)
true_post = unnorm_post / np.trapz(unnorm_post, f_grid)

mu_q, sigma_q = fit_variational(Y_OBS, TAU, R)
q_pdf = stats.norm.pdf(f_grid, mu_q, sigma_q)
print(f"Single-point q: μ={mu_q:.3f}, σ={sigma_q:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# Data for panels (c) & (d): full-function view
# ══════════════════════════════════════════════════════════════════════════════
# Two-bump latent function.  Both bumps sit well inside [0.2, 5.8] and a small
# negative baseline ensures the edges are clearly low (mean count < 1), so the
# spline has nowhere to go but down at the boundaries.
N_OBS = 7  # 10
X_OBS = np.linspace(0.2, 5.8, N_OBS)

f_true_obs = (
    2.2 * np.exp(-(((X_OBS - 2.0) / 0.85) ** 2))
    + 1.8 * np.exp(-(((X_OBS - 4.1) / 0.75) ** 2))
    - 0.4  # baseline: edges decay to exp(-0.4) ≈ 0.67 mean count
)
y_multi = np.random.negative_binomial(R, R / (R + np.exp(f_true_obs))).astype(float)

# Per-observation variational optimisation
mu_q_obs = np.empty(N_OBS)
sigma_q_obs = np.empty(N_OBS)
for i, yi in enumerate(y_multi):
    mu_q_obs[i], sigma_q_obs[i] = fit_variational(yi, TAU, R)

# Smooth GP posterior via cubic spline.  Clamped BCs (zero slope at both ends)
# prevent the spline from overshooting upward at the boundaries.
X_FINE = np.linspace(0, 6, 1000)
cs_mean = interpolate.CubicSpline(X_OBS, mu_q_obs, bc_type="clamped")
cs_sigma = interpolate.CubicSpline(X_OBS, sigma_q_obs, bc_type="clamped")
mean_fine = cs_mean(X_FINE)
sigma_fine = np.clip(cs_sigma(X_FINE), 0.05, None)

# NB predictive quantiles on the count scale
lambda_fine = np.exp(mean_fine)
p_fine = R / (R + lambda_fine)
y_lo = sp_nbinom.ppf(0.05, R, p_fine)
y_hi = sp_nbinom.ppf(0.95, R, p_fine)

# Highlighted observation: pick the one closest to Y_OBS to link back to (a)/(b)
hi = int(np.argmin(np.abs(y_multi - Y_OBS)))

# ══════════════════════════════════════════════════════════════════════════════
# Build the combined figure
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(13, 7))
gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.38)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1], sharex=ax_a)
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1], sharex=ax_c)

# ── Panel (a): Prior × NB Likelihood → non-Gaussian posterior ─────────────────
prior_pdf = np.exp(log_prior)
lik_scaled = np.exp(log_lik_grid - log_lik_grid.max())

ax_a.plot(
    f_grid,
    prior_pdf / prior_pdf.max(),
    color=PRIOR_COLOR,
    lw=2,
    label=r"Prior  $p(f)=\mathcal{N}(0,\tau^2)$",
)
ax_a.fill_between(f_grid, 0, prior_pdf / prior_pdf.max(), color=PRIOR_COLOR, alpha=0.20)
ax_a.plot(
    f_grid,
    lik_scaled,
    color=LIKE_COLOR,
    lw=2,
    linestyle="--",
    label=rf"Likelihood  $p(y\!=\!{Y_OBS}\mid f)$  [NB, $r\!=\!{R:.0f}$]",
)
ax_a.plot(
    f_grid,
    true_post / true_post.max(),
    color=TRUE_COLOR,
    lw=2.5,
    label=r"True posterior  $p(f\mid y)$  [non-Gaussian]",
)
ax_a.fill_between(f_grid, 0, true_post / true_post.max(), color=TRUE_COLOR, alpha=0.15)
ax_a.set_xlim(-3, 8)
ax_a.set_ylim(0, 1.25)
ax_a.set_xlabel(r"Latent log-rate  $f(x_i = 2.1)$", fontsize=9)
ax_a.set_ylabel("Scaled density", fontsize=9)
ax_a.set_title("(a) NB likelihood creates a non-Gaussian posterior", fontsize=10)
ax_a.legend(fontsize=7.5, loc="upper right")

# ── Panel (b): True posterior vs variational Gaussian q*(f) ───────────────────
ax_b.plot(
    f_grid,
    true_post,
    color=TRUE_COLOR,
    lw=2.5,
    label=r"True posterior  $p(f\mid y)$  [intractable]",
)
ax_b.fill_between(f_grid, 0, true_post, color=TRUE_COLOR, alpha=0.15)
ax_b.plot(
    f_grid,
    q_pdf,
    color=VARI_COLOR,
    lw=2.5,
    linestyle="--",
    label=rf"$q(f)=\mathcal{{N}}(\mu\!=\!{mu_q:.2f},\,\sigma\!=\!{sigma_q:.2f})$",
)
ax_b.fill_between(f_grid, 0, q_pdf, color=VARI_COLOR, alpha=0.15)
ax_b.fill_between(
    f_grid,
    true_post,
    q_pdf,
    where=(true_post > q_pdf),
    color=TRUE_COLOR,
    alpha=0.35,
    label="KL divergence gap",
)
ax_b.fill_between(
    f_grid, true_post, q_pdf, where=(q_pdf > true_post), color=VARI_COLOR, alpha=0.35
)
ax_b.set_xlim(-3, 8)
ax_b.set_xlabel(r"Latent log-rate  $f(x_i = 2.1)$", fontsize=9)
ax_b.set_ylabel("Density", fontsize=9)
ax_b.set_title(r"(b) Best Gaussian $q(f)\approx p(f\mid y)$ via ELBO", fontsize=10)
ax_b.legend(fontsize=7.5, loc="upper right")

# ── Panel (c): Split-violin glyphs — left=prior, right=variational posterior ──
ax_c.fill_between(
    X_FINE,
    mean_fine - 2 * sigma_fine,
    mean_fine + 2 * sigma_fine,
    color=BAND_COLOR,
    alpha=0.6,
    zorder=1,
)
ax_c.plot(X_FINE, mean_fine, color=MEAN_COLOR, lw=2, zorder=2)

# Common density range for all glyphs — chosen to show prior bell and posterior peak
GLYPH_F_LO, GLYPH_F_HI = -6.0, 6.0
f_glyph = np.linspace(GLYPH_F_LO, GLYPH_F_HI, 200)
prior_dens_base = stats.norm.pdf(f_glyph, 0, TAU)

GHW = 0.22  # glyph half-width in x-axis units

for i in range(N_OBS):
    x_i = X_OBS[i]
    mu_i = mu_q_obs[i]
    sig_i = sigma_q_obs[i]
    is_hi = i == hi
    post_color = VARI_COLOR  # TRUE_COLOR if is_hi else VARI_COLOR
    post_alpha = 0.85 if is_hi else 0.65

    # Left half: prior (always N(0, TAU²), same shape at every x_i)
    pd = prior_dens_base / prior_dens_base.max() * GHW
    ax_c.fill_betweenx(f_glyph, x_i - pd, x_i, color=PRIOR_COLOR, alpha=0.70, zorder=3)
    ax_c.plot(x_i - pd, f_glyph, color=PRIOR_COLOR, lw=0.7, zorder=4)

    # Right half: variational posterior q(f_i) = N(μ_i, σ_i²)
    qd = stats.norm.pdf(f_glyph, mu_i, sig_i)
    qd = qd / qd.max() * GHW
    ax_c.fill_betweenx(
        f_glyph, x_i, x_i + qd, color=post_color, alpha=post_alpha, zorder=3
    )
    ax_c.plot(x_i + qd, f_glyph, color=post_color, lw=0.7, zorder=4)

    # Spine at the split
    ax_c.plot(
        [x_i, x_i], [GLYPH_F_LO, GLYPH_F_HI], color="white", lw=0.8, zorder=3, alpha=0.6
    )

# Variational means μᵢ — these are the correct posterior estimates and lie on the spline
ax_c.scatter(
    X_OBS,
    mu_q_obs,
    color=DATA_COLOR,
    s=28,
    zorder=5,
    marker="o",
    label=r"Variational means $\mu_i$",
)

# Annotation linking highlighted glyph to panels (a)/(b).
# Place text in the lower-left quadrant so it stays inside the axes.
ax_c.annotate(
    "Right half = $q(f_i)$\nas in panel (b)",
    xy=(X_OBS[hi] + GHW, mu_q_obs[hi]),
    xytext=(X_OBS[hi] + 0.6, GLYPH_F_LO + 1.5),
    fontsize=7.5,
    ha="left",
    color=post_color,
    arrowprops=dict(arrowstyle="-|>", color=post_color, lw=1.2),
    zorder=6,
)

# Custom legend patches
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor=PRIOR_COLOR, alpha=0.7, label="Left: prior $p(f)$"),
    Patch(facecolor=VARI_COLOR, alpha=0.7, label="Right: variational $q(f_i)$"),
    plt.Line2D([0], [0], color=MEAN_COLOR, lw=2, label=r"Variational mean $\mu(x)$"),
]
ax_c.legend(handles=legend_elements, fontsize=7.5, loc="upper right")
ax_c.set_xlim(0, 6)
ax_c.set_ylim(GLYPH_F_LO, GLYPH_F_HI)
ax_c.set_xlabel("X", fontsize=9)
ax_c.set_ylabel(r"Latent log-rate  $f(x)$", fontsize=9)
ax_c.set_title(
    r"(c) Split violins: prior $p(f)$ [gray] vs $q(f_i)$ [blue] at each $x_i$",
    fontsize=10,
)

# ── Panel (d): NB count observations on the count scale ───────────────────────
# Smooth posterior rate samples: draw f_s ~ q(f) by adding a length-scale-matched
# smooth perturbation to the posterior mean, then exponentiate to the count scale.
N_RATE_SAMP = 10
for _ in range(N_RATE_SAMP):
    eps = gaussian_filter1d(np.random.randn(len(X_FINE)), sigma=20)
    eps = eps / eps.std()
    f_s = mean_fine + sigma_fine * eps
    ax_d.plot(X_FINE, np.exp(f_s), color="#dddddd", lw=0.9, zorder=0)

ax_d.fill_between(
    X_FINE,
    y_lo,
    y_hi,
    color=BAND_COLOR,
    alpha=0.7,
    zorder=1,
    label="NB 90% predictive interval",
)
ax_d.plot(
    X_FINE,
    lambda_fine,
    color=MEAN_COLOR,
    lw=2,
    zorder=2,
    label=r"Posterior mean rate  $e^{\mu(x)}$",
)
ax_d.scatter(
    X_OBS, y_multi, color=DATA_COLOR, s=28, zorder=3, label="Observed counts  $y_i$"
)
ax_d.scatter(X_OBS[hi], y_multi[hi], color=TRUE_COLOR, s=70, zorder=4, marker="o")

# NB PMF glyphs at a few representative x locations
pmf_xs = [X_OBS[1], X_OBS[hi], X_OBS[-2]]
PMF_HW = 0.22
for xi in pmf_xs:
    lam_xi = float(np.exp(cs_mean(xi)))
    p_xi = R / (R + lam_xi)
    max_k = int(sp_nbinom.ppf(0.97, R, p_xi)) + 1
    k_vals = np.arange(max_k + 1)
    pmf = sp_nbinom.pmf(k_vals, R, p_xi)
    pmf_sc = pmf / pmf.max() * PMF_HW
    color = TRUE_COLOR if np.isclose(xi, X_OBS[hi]) else VARI_COLOR
    for k, pw in zip(k_vals, pmf_sc):
        ax_d.barh(
            k, 2 * pw, left=xi - pw, height=0.6, color=color, alpha=0.55, linewidth=0
        )

ax_d.set_xlim(0, 6)
ax_d.set_xlabel("X", fontsize=9)
ax_d.set_ylabel("Count  $y$", fontsize=9)
ax_d.set_title(
    r"(d) NB counts: discrete, overdispersed — linked to $f(x)$ via log link",
    fontsize=10,
)
ax_d.legend(fontsize=7.5, loc="upper right")

out = os.path.join(OUT_DIR, "vgp_ex.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
print(f"Saved {out}")
