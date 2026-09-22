"""
Day 3 -- Conjugate Priors: Beta-Binomial
==========================================
Scenario: you're courtside tracking a player's first-serve-in percentage.
A pre-match scout gives you a prior belief in plain English; then you watch
serves land one at a time and update as you go. See README.md for the full
story -- this file assumes you've already derived the update rule on paper.

Run: python exercise.py
"""


def beta_update(alpha: float, beta: float, k: int, n: int) -> tuple:
    """
    Apply the Beta-Binomial conjugate update rule you derived on paper.
    `k` = serves landed in, `n` = serves attempted, for whatever stretch
    of the match you're updating on. Returns (new_alpha, new_beta).
    """
    return (k+alpha, n + beta - k)


def beta_mean(alpha: float, beta: float) -> float:
    """Mean of a Beta(alpha, beta) distribution."""
    return alpha / (alpha + beta)


def beta_variance(alpha: float, beta: float) -> float:
    """Variance of a Beta(alpha, beta) distribution."""
    mean = beta_mean(alpha, beta)
    return mean * (1-mean) / (alpha + beta + 1)


def sequential_vs_batch(prior_alpha: float, prior_beta: float, serve_outcomes: list) -> tuple:
    """
    `serve_outcomes` is a list of 0/1 outcomes (0 = fault/long/wide,
    1 = in), one per first serve, in the order they were played.

    Update once per serve, sequentially -- live-scoring style, each
    posterior becoming the next serve's prior. Separately, update once on
    the full tally (total in, total attempted, computed from
    `serve_outcomes`). Return (sequential_result, batch_result), each an
    (alpha, beta) tuple.
    """
    sequential_alpha = prior_alpha
    sequential_beta = prior_beta
    for outcome in serve_outcomes:
        if outcome == 0:
            # loss
            sequential_beta += 1
        else:
            sequential_alpha += 1

    return ((sequential_alpha, sequential_beta), beta_update(prior_alpha, prior_beta, sum(serve_outcomes), len(serve_outcomes)))


def pseudo_count_prior(believed_p: float, confidence: float) -> tuple:
    """
    Turn a scout's read -- "I believe the true rate is about `believed_p`,
    and I'd trade that belief for `confidence` equivalent prior serves
    already observed" -- into a Beta(alpha, beta) prior.

    Must satisfy: alpha + beta == confidence (the number of serves), and
    alpha / (alpha + beta) == believed_p (the expected probability).
    """
    alpha = believed_p * confidence  # predicted number of successes
    beta = confidence - alpha  # predicted number of failures
    return alpha, beta


if __name__ == "__main__":
    import importlib.util
    import os

    # --- self-check 1: live scoring must EXACTLY match the changeover tally ---
    # Your courtside log of the first 10 first serves: 3 in, 7 out.
    serve_outcomes = [1, 0, 0, 1, 0, 0, 0, 0, 1, 0]
    seq_result, batch_result = sequential_vs_batch(1, 1, serve_outcomes)
    assert seq_result == batch_result, (
        f"Sequential {seq_result} != batch {batch_result} -- these must be "
        f"EXACTLY equal, not approximately close. If they differ, something "
        f"in your update implicitly depends on the order the serves arrived "
        f"in, which a Beta-Binomial update should never do."
    )
    print(f"Live scoring == changeover tally: {seq_result}")

    # --- self-check 2: pseudo_count_prior, using the scout's actual read ---
    # "About 65%, and I'd trust that about as much as 20 serves I'd already watched."
    for p, conf in [(0.65, 20), (0.5, 10), (0.1, 4)]:
        a, b = pseudo_count_prior(p, conf)
        assert abs((a + b) - conf) < 1e-9, (
            f"alpha+beta should equal confidence={conf}, got {a + b}"
        )
        assert abs(a / (a + b) - p) < 1e-6, (
            f"alpha/(alpha+beta) should equal believed_p={p}, got {a / (a + b)}"
        )
    scout_alpha, scout_beta = pseudo_count_prior(0.65, 20)
    print(f"Scout's read as a prior: Beta({scout_alpha}, {scout_beta})")

    # --- self-check 3: grid approximation (Day 2) should converge to this ---
    # Same first-set numbers as above: 3 of the first 10 first serves in,
    # starting from a flat (uniform) prior -- no scouting report yet.
    #
    # Loaded by explicit file path (not sys.path + `import exercise`) because
    # Day 2's file is ALSO named exercise.py -- appending to sys.path doesn't
    # help, since Python puts this script's own directory at the FRONT of
    # sys.path automatically, so a same-named `import exercise` would just
    # re-import this file. Loading by path sidesteps the collision entirely.
    day2_path = os.path.join(
        os.path.dirname(
            __file__), "..", "day_02_grid_approximation", "exercise.py"
    )
    spec = importlib.util.spec_from_file_location("day2_exercise", day2_path)
    day2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(day2)

    k, n = 3, 10
    alpha_post, beta_post = beta_update(
        1, 1, k, n)  # uniform prior == Beta(1,1)
    analytic_mean = beta_mean(alpha_post, beta_post)

    errors = []
    for grid_size in (50, 500, 5000):
        grid = day2.make_grid(grid_size)
        prior_vals = day2.uniform_prior(grid)
        lik_vals = day2.binomial_likelihood(grid, k, n)
        post_vals = day2.grid_posterior(prior_vals, lik_vals)
        grid_mean = day2.posterior_mean(grid, post_vals)
        errors.append(abs(grid_mean - analytic_mean))
        print(
            f"grid_size={grid_size:>5}: grid_mean={grid_mean:.6f}, "
            f"analytic_mean={analytic_mean:.6f}, error={errors[-1]:.2e}"
        )

    assert errors[0] > errors[1] > errors[2], (
        "Grid approximation error should shrink monotonically as grid_size "
        "increases -- if it doesn't, this is a signal to revisit Day 2, not "
        "today's code."
    )

    print("\nDay 3 core checks passed.")
