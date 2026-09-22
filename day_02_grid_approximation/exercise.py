"""
Day 2 -- Grid Approximation
============================
Grid approximation for a Binomial likelihood with a continuous prior over
theta in [0, 1]. Much less scaffolding than Day 1 -- you're building every
piece of the pipeline yourself, and no formulas are given.

Run: python exercise.py
"""

import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt


def make_grid(size: int) -> np.ndarray:
    """Return `size` evenly spaced points covering [0, 1], inclusive of both endpoints."""
    return np.linspace(0, 1, size)


def uniform_prior(grid: np.ndarray) -> np.ndarray:
    """
    Uniform prior density evaluated at every point in `grid`.
    Must sum to 1 across the grid points.
    """
    size = grid.size
    uniform_probability = 1/size
    return np.array([uniform_probability]*size)


def piecewise_line(peak: float, point: float) -> float:
    """Makes a piecewise line (a triangle) going from 0,1
    with a peak at a specified point 0 < point < 1."""
    arb_max = 1
    increasing_gradient = arb_max/peak
    decreasing_gradient = arb_max / (peak - 1)
    if point <= peak:
        return point * increasing_gradient
    else:
        return decreasing_gradient * (point - 1)


def normalise(vector: np.array) -> np.array:
    """Normalises a vector"""
    return vector / sum(vector)


def triangular_prior(grid: np.ndarray, peak: float = 0.5) -> np.ndarray:
    """
    A prior that rises linearly from 0 at theta=0 to a maximum at
    theta=`peak`, then falls linearly back to 0 at theta=1.
    Must sum to 1 across the grid points.
    """
    points = np.array([piecewise_line(peak, point) for point in grid])
    return normalise(points)


def binomial_likelihood(grid: np.ndarray, k: int, n: int) -> np.ndarray:
    """
    P(k successes in n trials | theta) for every theta in `grid`.
    Does NOT need to sum to 1 -- it's a likelihood over data, not a
    distribution over theta.
    """
    return np.array([
        binom.pmf(k, n, theta) for theta in grid
    ])


def grid_posterior(prior_vals: np.ndarray, likelihood_vals: np.ndarray) -> np.ndarray:
    """
    Combine a prior and a likelihood (both evaluated on the same grid,
    elementwise) into a posterior that sums to 1.
    """
    posterior = np.array([
        p_val * l_val for p_val, l_val in zip(prior_vals, likelihood_vals)
    ])
    return normalise(posterior)


def posterior_mean(grid: np.ndarray, posterior_vals: np.ndarray) -> float:
    """Expected value of theta under the posterior."""
    return sum([theta*posterior_theta for theta, posterior_theta in zip(grid, posterior_vals)])


def posterior_credible_interval(grid: np.ndarray, posterior_vals: np.ndarray, mass: float = 0.95) -> tuple:
    """
    Equal-tailed credible interval: (lower, upper) such that `mass`
    probability lies between them under the posterior. Use the cumulative
    sum of posterior_vals over the grid.
    """
    cumsum = np.array([sum(posterior_vals[:i]) for i in range(grid.size)])
    # we're given a mass, we have the cumulative sum of posteriors
    remaining_mass = 1 - mass
    # We want to discount the lower remaining_mass / 2 probs, and the upper remaining_mass / 2 which is anything greater than 1 - remaining_mass / 2
    lower_bound = remaining_mass / 2
    upper_bound = 1 - lower_bound

    # the first index where there's a value greater than the lower bound (searching from bottom)
    for i in range(len(cumsum)):
        if cumsum[i] >= lower_bound:
            lb = i
            break
    # the first index where there's a value less than the upper bound (searching from top)
    for i in range(len(cumsum[::-1])):
        if cumsum[i] <= upper_bound:
            ub = i
    return (grid[lb], grid[ub])


# n = 10
# dimension = 10_000
# grid = make_grid(dimension)
# likelihood = binomial_likelihood(grid, 3, n)
# prior = triangular_prior(grid, peak=0.6)
# posterior = grid_posterior(prior, likelihood)
# # grid = make_grid(dimension)
# plt.plot(grid, posterior, color='red')
# # plt.plot(grid, prior, color='green')
# plt.axvline(x=posterior_mean(grid, posterior))
# lb, ub = posterior_credible_interval(grid, posterior)
# plt.axvline(x=lb)
# plt.axvline(x=ub)
# plt.show()
# raise Exception


GRID_SIZE = 2000  # increase if your self-checks don't converge


def run(k: int, n: int, prior_fn=uniform_prior):
    grid = make_grid(GRID_SIZE)
    prior_vals = prior_fn(grid)
    lik_vals = binomial_likelihood(grid, k, n)
    post_vals = grid_posterior(prior_vals, lik_vals)
    return grid, post_vals


if __name__ == "__main__":
    from scipy import stats

    datasets = [(3, 10), (30, 100), (300, 1000)]
    means, stds = [], []

    for k, n in datasets:
        grid, post = run(k, n, uniform_prior)
        m = posterior_mean(grid, post)
        lo, hi = posterior_credible_interval(grid, post)
        std = np.sqrt(np.sum(post * (grid - m) ** 2))
        means.append(m)
        stds.append(std)
        print(
            f"k={k:>3}, n={n:>4}: mean={m:.4f}, 95% CI=({lo:.4f}, {hi:.4f}), std={std:.5f}")

    # --- self-check 1: smallest dataset vs. the known closed-form solution ---
    # (a uniform prior + binomial likelihood has an exact closed form --
    #  you'll properly derive and understand this on Day 3; for now it's
    #  just your ground truth)
    ref = stats.beta(1 + 3, 1 + 10 - 3)
    assert abs(means[0] - ref.mean()) < 0.01, (
        "Posterior mean doesn't match the analytic reference -- "
        "check grid resolution or normalization."
    )
    ref_lo, ref_hi = ref.interval(0.95)
    grid0, post0 = run(3, 10, uniform_prior)
    lo0, hi0 = posterior_credible_interval(grid0, post0)
    assert abs(lo0 - ref_lo) < 0.02 and abs(hi0 - ref_hi) < 0.02, (
        "Credible interval doesn't match the analytic reference."
    )

    # --- self-check 2: posterior should concentrate as n grows ---
    # std should shrink by roughly sqrt(10) each time n scales by 10x.
    # This can't be gamed by hard-coding a single return value -- it has
    # to hold across three independent runs of your own functions.
    expected_ratio = np.sqrt(10)
    ratio_1 = stds[0] / stds[1]
    ratio_2 = stds[1] / stds[2]
    assert abs(ratio_1 - expected_ratio) / expected_ratio < 0.15, (
        f"std didn't shrink as expected from n=10 to n=100 "
        f"(ratio={ratio_1:.2f}, expected ~{expected_ratio:.2f})"
    )
    assert abs(ratio_2 - expected_ratio) / expected_ratio < 0.15, (
        f"std didn't shrink as expected from n=100 to n=1000 "
        f"(ratio={ratio_2:.2f}, expected ~{expected_ratio:.2f})"
    )

    print("\nDay 2 core checks passed.")

    # --- plotting (uncomment once the checks above pass) ---
    import matplotlib.pyplot as plt

    plt.figure()
    for (k, n), label in zip(datasets, ["n=10", "n=100", "n=1000"]):
        grid, post = run(k, n, uniform_prior)
        plt.plot(grid, post, label=label)
    plt.legend()
    plt.xlabel("theta")
    plt.title("Posterior sharpening with more data")
    plt.show()

    grid_u, post_u = run(3, 10, uniform_prior)
    grid_t, post_t = run(3, 10, triangular_prior)
    plt.figure()
    plt.plot(grid_u, post_u, label="uniform prior")
    plt.plot(grid_t, post_t, label="triangular prior (peak=0.5)")
    plt.legend()
    plt.xlabel("theta")
    plt.title("Effect of prior choice, n=10")
    plt.show()
