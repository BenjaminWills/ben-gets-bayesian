"""
Day 5 -- Summarizing the Posterior: Three Decisions
=====================================================
Three case studies, reusing Day 3/4's update functions to get posteriors,
then asking what you actually DO with one. See README.md.

Run: python exercise.py
(Produces three PNGs in this folder -- open them, that's the actual point.)
"""

import matplotlib.pyplot as plt
import os

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
import matplotlib
# remove this line if you want interactive windows instead
matplotlib.use("Agg")


def beta_map(a: float, b: float) -> float:
    """Mode of a Beta(a, b) distribution. Only defined for a, b > 1."""
    return (a-1)/(a+b-2)


def gamma_poisson_map(a: float, b: float) -> float:
    """Mode of a Gamma(a, b) [rate parameterization] distribution. Only defined for a > 1."""
    return (a-1)/b


def equal_tailed_interval(dist, mass: float = 0.95) -> tuple:
    """
    `dist` is a frozen scipy.stats distribution (e.g. stats.beta(4, 156)).
    Returns (lower, upper) cutting equal probability off both tails.

    We can find this the old fashioned way. The remaining mass we need to fill is 1-mass, then
    we want to divide this remaining mass between the two end points. So we're looking for the first
    time the CDF is greater than remaining_mass/2 and when it is greater than 1-remaining_mass/2. Those
    are our start and end points
    """
    remaining_mass = 1 - mass
    left_tail_mass = remaining_mass / 2
    right_tail_mass = 1 - left_tail_mass
    return dist.ppf(left_tail_mass), dist.ppf(right_tail_mass)


def hpd_interval(dist, mass: float = 0.95) -> tuple:
    """
    `dist` is a frozen scipy.stats distribution. Returns (lower, upper):
    the NARROWEST interval containing `mass` probability. See README for
    what to minimize and with what.
    """
    def width(alpha):
        return dist.ppf(alpha + mass) - dist.ppf(alpha)

    result = minimize_scalar(width, bounds=(0, 1 - mass), method="bounded")
    alpha_opt = result.x
    return dist.ppf(alpha_opt), dist.ppf(alpha_opt + mass)


def prob_exceeds(dist, threshold: float) -> float:
    """P(true parameter > threshold), under this posterior."""
    return 1 - dist.cdf(threshold)


# ---------------------------------------------------------------------------
# Provided: plotting. Once your functions above are correct, this "just
# works" -- the point of today is looking at what it produces, not writing
# matplotlib calls.
# ---------------------------------------------------------------------------

def plot_posterior_summary(dist, title: str, save_path: str, is_beta=False, is_gamma=False, mass: float = 0.95):
    if is_beta:
        map_val = beta_map(dist.args[0], dist.args[1])
        x = np.linspace(dist.ppf(0.0001), dist.ppf(0.9999), 2000)
    elif is_gamma:
        map_val = gamma_poisson_map(dist.args[0], 1 / dist.kwds["scale"])
        x = np.linspace(dist.ppf(0.0001), dist.ppf(0.9999), 2000)
    else:
        map_val = dist.mean()
        x = np.linspace(dist.ppf(0.0001), dist.ppf(0.9999), 2000)

    y = dist.pdf(x)
    eq_lo, eq_hi = equal_tailed_interval(dist, mass)
    hpd_lo, hpd_hi = hpd_interval(dist, mass)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, y, color="black", lw=1.5)
    ax.fill_between(x, y, where=(x >= eq_lo) & (x <= eq_hi), alpha=0.25,
                    color="tab:blue", label=f"equal-tailed {int(mass*100)}%")
    ax.fill_between(x, y, where=(x >= hpd_lo) & (x <= hpd_hi), alpha=0.35,
                    color="tab:orange", label=f"HPD {int(mass*100)}%", hatch="//")

    ax.axvline(dist.mean(), color="tab:red", ls="-",
               lw=2, label=f"mean = {dist.mean():.4g}")
    ax.axvline(dist.median(), color="tab:green", ls="--",
               lw=2, label=f"median = {dist.median():.4g}")
    ax.axvline(map_val, color="tab:purple", ls=":",
               lw=2, label=f"MAP = {map_val:.4g}")

    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"saved: {save_path}")


def beta_update(a, b, k, n):
    return a + k, b + (n - k)


def gamma_poisson_update(a, b, counts):
    return a + sum(counts), b + len(counts)


if __name__ == "__main__":
    OUT_DIR = os.path.dirname(__file__) or "."

    # =========================================================
    # Case 1: defect rate -- Beta-Binomial
    # =========================================================
    # Historical baseline: ~2% defect rate, worth ~100 pseudo-units.
    prior_a, prior_b = 2, 98
    post_a, post_b = beta_update(prior_a, prior_b, k=2, n=60)
    dist1 = stats.beta(post_a, post_b)

    map1 = beta_map(post_a, post_b)
    assert abs(
        map1 - 0.0189873417721519) < 1e-6, f"beta_map: expected ~0.01899, got {map1}"

    eq1_lo, eq1_hi = equal_tailed_interval(dist1)
    hpd1_lo, hpd1_hi = hpd_interval(dist1)
    assert (hpd1_hi - hpd1_lo) <= (eq1_hi - eq1_lo) + \
        1e-6, "HPD must never be wider than equal-tailed"
    assert (eq1_hi - eq1_lo) - (hpd1_hi - hpd1_lo) > 0.001, (
        "HPD should be MEANINGFULLY narrower than equal-tailed for this skewed "
        "posterior -- if they're nearly identical, hpd_interval is probably "
        "just returning equal-tailed bounds."
    )

    p1 = prob_exceeds(dist1, 0.02)
    assert abs(
        p1 - 0.6065746417071881) < 1e-6, f"prob_exceeds: expected ~0.6066, got {p1}"
    decision1 = "HALT the line" if p1 > 0.90 else "do NOT halt the line"
    print(f"Case 1: P(defect rate > 2%) = {p1:.4f} -> {decision1}")

    plot_posterior_summary(
        dist1, "Case 1: defect rate posterior, Beta({}, {})".format(
            post_a, post_b),
        os.path.join(OUT_DIR, "case1_defect_rate.png"), is_beta=True,
    )

    # =========================================================
    # Case 2: error rate -- Gamma-Poisson
    # =========================================================
    # Historical baseline: ~3 errors/hour, worth 10 hours of prior observation.
    prior_a2, prior_b2 = 30, 10
    post_a2, post_b2 = gamma_poisson_update(prior_a2, prior_b2, [5, 6, 4, 7])
    dist2 = stats.gamma(post_a2, scale=1 / post_b2)

    map2 = gamma_poisson_map(post_a2, post_b2)
    assert abs(
        map2 - 3.642857142857143) < 1e-6, f"gamma_poisson_map: expected ~3.6429, got {map2}"

    p2 = prob_exceeds(dist2, 3.0)
    assert abs(
        p2 - 0.925087084591484) < 1e-6, f"prob_exceeds: expected ~0.9251, got {p2}"
    print(f"Case 2: P(true error rate > historical 3/hr) = {p2:.4f}")

    plot_posterior_summary(
        dist2, "Case 2: error rate posterior, Gamma({}, {})".format(
            post_a2, post_b2),
        os.path.join(OUT_DIR, "case2_error_rate.png"), is_gamma=True,
    )

    # =========================================================
    # Case 3: heart rate -- Normal-Normal (symmetric contrast case)
    # =========================================================
    post_mean3, post_var3 = 69.404761904761905, 2.678571428571429
    dist3 = stats.norm(post_mean3, np.sqrt(post_var3))

    eq3_lo, eq3_hi = equal_tailed_interval(dist3)
    hpd3_lo, hpd3_hi = hpd_interval(dist3)
    assert abs((eq3_hi - eq3_lo) - (hpd3_hi - hpd3_lo)) < 1e-3, (
        "For a SYMMETRIC posterior, HPD and equal-tailed should coincide -- "
        "if they don't here, that's a bug in hpd_interval, not an interesting result."
    )
    print(
        f"Case 3: mean={dist3.mean():.4f}, median={dist3.median():.4f} "
        f"(MAP == mean for a Normal) -- equal-tailed and HPD widths match: "
        f"{eq3_hi-eq3_lo:.4f} vs {hpd3_hi-hpd3_lo:.4f}"
    )

    plot_posterior_summary(
        dist3, "Case 3: heart rate posterior, Normal({:.4f}, {:.4f})".format(
            post_mean3, post_var3),
        os.path.join(OUT_DIR, "case3_heart_rate.png"),
    )

    print("\nDay 5 core checks passed. Now go open the three PNGs.")
