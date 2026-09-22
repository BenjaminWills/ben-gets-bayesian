# Day 2 — Grid Approximation

## Concept

Same three-step algorithm as Day 1 (multiply, sum, divide) — but the
hypothesis space is now continuous. Instead of a `dict` over a handful of
named hypotheses, you'll use a fine grid of points covering θ ∈ [0, 1], and
treat the prior and the likelihood as arrays evaluated at every grid point.
Everything else is identical: elementwise multiply, then normalize.

No formulas are given below. You did a stats degree — pull the Binomial
PMF from memory or a reference. The point of today is assembling the
pipeline correctly, not memorizing formulas.

## Task

Implement, in `exercise.py`, from scratch:

1. `make_grid(size)` — evenly spaced points covering [0, 1].
2. `uniform_prior(grid)` — flat prior, must sum to 1 across the grid points.
3. `triangular_prior(grid, peak=0.5)` — a prior that rises semi-linearly to a
   maximum at `peak` and falls semi-linearly back down at the edges of [0, 1].
   Must sum to 1. Hint: it'll be symmetricly linear for odd sized grids and *almost* symmetricly linear for even sized grids.
4. `binomial_likelihood(grid, k, n)` — P(k successes in n trials | θ) for
   every θ in the grid. This does **not** need to sum to 1 — it's a
   likelihood over data, not a distribution over θ.
5. `grid_posterior(prior_vals, likelihood_vals)` — generalize your Day 1
   `bayes_update` to arrays. (Elementwise multiply, then normalize — you
   already wrote this out as vector algebra a few messages ago.)
6. `posterior_mean(grid, posterior_vals)` and
   `posterior_credible_interval(grid, posterior_vals, mass=0.95)` — the
   equal-tailed interval, using the cumulative sum of the posterior over
   the grid.

Then run all of the above on three datasets, all at the same underlying
rate (0.3): `(k=3, n=10)`, `(k=30, n=100)`, `(k=300, n=1000)`, using the
uniform prior, and report the posterior mean, 95% credible interval, and
standard deviation for each.

Finally, plot:
- All three posteriors on one axis (uncomment the plotting block once your
  checks pass) — you should see it visibly sharpen.
- The uniform-prior vs. triangular-prior posteriors, on the smallest
  dataset only (`k=3, n=10`) — this shows the prior's influence when data
  is scarce, which is exactly what should vanish by the `n=1000` case.

## Self-check

Two kinds of check in `exercise.py`, and they're stricter than Day 1's:

1. Your `(k=3, n=10)` uniform-prior posterior mean and interval are checked
   against the known closed-form solution for this exact setup (you'll
   properly derive and understand *why* this closed form exists on Day 3
   — for now, it's just your ground truth). If this fails, suspect your
   grid resolution (`GRID_SIZE`) before suspecting your logic.
2. The posterior's spread (std) is checked for shrinking by roughly the
   expected factor as you go from `n=10 → n=100 → n=1000`. This can't be
   satisfied by hard-coding a single return value — it has to hold across
   three independent runs of your own functions, which makes it a much
   better test of whether your normalization is actually correct.

## Definition of done

- [ ] All 6 functions implemented — no hard-coded probabilities anywhere
- [ ] Both self-checks pass (the scaling one usually catches normalization
      bugs the first check misses)
- [ ] Both plots produced and visually inspected
- [ ] You can explain, without notes, why increasing `GRID_SIZE` improves
      accuracy but costs runtime — and why that tradeoff is precisely the
      reason grid approximation gets abandoned once you hit more than 1–2
      parameters (this is the setup for Day 8)
