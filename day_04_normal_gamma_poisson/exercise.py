"""
Day 4 -- Conjugate Priors: Normal-Normal & Gamma-Poisson
==========================================================
Two scenarios: estimating your true resting heart rate from noisy
smartwatch readings (Normal-Normal), and estimating a student's true
bugs-per-review rate from a handful of code reviews (Gamma-Poisson).

No formulas are given anywhere in this file. See README.md -- both
derivations should be done on paper before you write a line of code here.

Run: python exercise.py
"""


def normal_update(prior_mean: float, prior_var: float, data: list, known_var: float) -> tuple:
    """
    `data` is a list of one or more Normal(mu, known_var) readings, where
    mu is the unknown true value you're estimating (e.g. your true resting
    heart rate) and known_var is the *measurement* noise variance (assumed
    known -- e.g. from the watch's spec sheet).

    Returns (posterior_mean, posterior_var) for mu.
    """
    n = len(data)
    mean = sum(data) / n
    precision = (n / (known_var)) + (1 / (prior_var))
    mean = ((n*mean/known_var) + (prior_mean / (prior_var))) / precision
    variance = 1 / precision
    return mean, variance


def gamma_poisson_update(prior_a: float, prior_b: float, counts: list) -> tuple:
    """
    `counts` is a list of one or more observed Poisson(lambda) counts,
    where lambda is the unknown true rate you're estimating (e.g. a
    student's true bugs-per-review rate).

    Returns (new_a, new_b).
    """
    T = sum(counts)
    n = len(counts)
    return prior_a + T, n + prior_b


def sequential_vs_batch_normal(prior_mean: float, prior_var: float, readings: list, known_var: float) -> tuple:
    """
    Update once per reading, sequentially (each posterior becomes the next
    prior). Separately, update once on the full batch of readings at once.
    Return (sequential_result, batch_result), each a (mean, var) tuple.

    Unlike Day 3, do not assume these come out bit-for-bit identical --
    investigate first (see README), then decide what comparison in the
    self-check below is actually justified.
    """
    batch_params = normal_update(prior_mean, prior_var, readings, known_var)

    # Sequential version would go like
    # Initial prior definition
    pm = prior_mean
    pv = prior_var
    for reading in readings:
        pm, pv = normal_update(
            pm, pv, [reading], known_var)

    return ((pm, pv), batch_params)


def effective_prior_sample_size(prior_var: float, known_var: float) -> float:
    """
    The single number that answers: "this prior is worth as much as how
    many actual data points?" -- the Normal-Normal analogue of Day 3's
    alpha+beta pseudo-count.
    """
    # In the spirit of that problem, alpha + beta was the number of trials that had been run, and alpha / alpha + beta was just the
    # estimated probability based on frequentist tactics, so we were content with that definition.
    # In this case we're looking at a prior variance and a known variance, and we want to return
    return known_var / prior_var


if __name__ == "__main__":
    # --- self-check 1: normal_update against a worked reference case ---
    # Prior belief: resting heart rate ~ Normal(70, 25) (i.e. mean 70 bpm,
    # std 5 bpm -- you're fairly but not extremely sure).
    # Watch's known measurement noise variance: 9 (std 3 bpm).
    # Three mornings' readings: 68, 71, 69.
    post_mean, post_var = normal_update(70, 25, [68, 71, 69], 9)
    assert abs(post_mean - 69.404761904761905) < 1e-6, (
        f"Expected posterior mean ~69.4048, got {post_mean}"
    )
    assert abs(post_var - 2.678571428571429) < 1e-6, (
        f"Expected posterior variance ~2.6786, got {post_var}"
    )
    print(
        f"normal_update reference case: mean={post_mean:.4f}, var={post_var:.4f}")

    # --- self-check 2: behavioral checks, matching your step-2 predictions ---
    # An extremely vague prior + one reading: the reading should dominate.
    wide_mean, _ = normal_update(70, 1e12, [55.0], 9)
    assert abs(wide_mean - 55.0) < 1e-4, (
        f"With a near-flat prior, one reading of 55 should give a posterior "
        f"mean near 55, got {wide_mean}"
    )

    # An extremely confident prior + a thousand conflicting readings: the
    # prior should barely move.
    narrow_mean, _ = normal_update(70, 1e-12, [10.0] * 1000, 9)
    assert abs(narrow_mean - 70.0) < 1e-4, (
        f"With an extremely confident prior of 70, even 1000 readings of "
        f"10 shouldn't move the posterior mean far from 70, got {narrow_mean}"
    )
    print("normal_update behavioral checks passed")

    # --- self-check 3: sequential vs batch, Normal-Normal ---
    readings = [68.0, 71.0, 69.0, 70.5, 67.2, 69.8, 71.3]
    seq_result, batch_result = sequential_vs_batch_normal(70, 25, readings, 9)
    assert abs(seq_result[0] - batch_result[0]) < 1e-9, (
        f"Sequential mean {seq_result[0]} and batch mean {batch_result[0]} "
        f"disagree by more than expected floating-point noise."
    )
    assert abs(seq_result[1] - batch_result[1]) < 1e-9, (
        f"Sequential var {seq_result[1]} and batch var {batch_result[1]} "
        f"disagree by more than expected floating-point noise."
    )
    print(f"Sequential vs batch (Normal): {seq_result} vs {batch_result}")

    # --- self-check 4: effective_prior_sample_size ---
    n_eff = effective_prior_sample_size(prior_var=9, known_var=36)
    assert abs(
        n_eff - 4.0) < 1e-9, f"Expected effective sample size 4.0, got {n_eff}"
    print(f"effective_prior_sample_size(9, 36) = {n_eff}")

    # --- self-check 5: gamma_poisson_update against a worked reference case ---
    # Prior belief about a new student's bugs-per-review rate: Gamma(4, 2)
    # (mean 2 bugs/review, based on experience with many past students).
    # Their first three reviews turn up 3, 1, and 4 bugs.
    new_a, new_b = gamma_poisson_update(4, 2, [3, 1, 4])
    assert (new_a, new_b) == (
        12, 5), f"Expected (12, 5), got ({new_a}, {new_b})"
    print(f"gamma_poisson_update reference case: (a, b) = ({new_a}, {new_b})")

    # --- self-check 6: sequential vs batch, Gamma-Poisson -- exact this time ---
    a, b = 4, 2
    for c in [3, 1, 4]:
        a, b = gamma_poisson_update(a, b, [c])
    batch_a, batch_b = gamma_poisson_update(4, 2, [3, 1, 4])
    assert (a, b) == (batch_a, batch_b), (
        f"Sequential {(a, b)} != batch {(batch_a, batch_b)} -- unlike the "
        f"Normal case, this should be EXACTLY equal. If it isn't, something "
        f"is oddly implemented -- Gamma-Poisson updating is pure integer "
        f"addition, same as Beta-Binomial on Day 3."
    )
    print(
        f"Sequential vs batch (Gamma-Poisson, exact): {(a, b)} == {(batch_a, batch_b)}")

    print("\nDay 4 core checks passed.")
