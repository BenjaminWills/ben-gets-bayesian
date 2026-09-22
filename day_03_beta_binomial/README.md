# Day 3 — Conjugate Priors: Beta-Binomial

## Concept

A prior is **conjugate** to a likelihood if the posterior lands in the same
family as the prior. For a Binomial likelihood, the Beta distribution is
exactly that family — and unlike Day 2, this isn't approximated on a grid,
it's exact.

**Before touching any code, derive this on paper.** It's short, and it's
the actual point of today — everything below is just typing up a result
you should already own.

Given:
- Prior: θ ~ Beta(α, β), so `p(θ) ∝ θ^(α−1) (1−θ)^(β−1)`
- Data: k successes in n trials, so `p(data|θ) ∝ θ^k (1−θ)^(n−k)` (the
  binomial coefficient is dropped — it doesn't involve θ, so it's just a
  constant that Bayes' theorem would divide back out anyway)

Multiply the two, exactly as you did by hand for the medical test problem
on Day 1 (prior × likelihood), and show the result is proportional to
`θ^(α+k−1) (1−θ)^(β+n−k−1)` — which you should recognize as an
*unnormalized* Beta(α+k, β+n−k) density. That's the whole proof: three
lines of algebra, no calculus required, since you never have to normalize
by hand — you just have to recognize the shape.

## Derivation

The probability density function of our prior is:

$$
p(θ) \propto θ^{(α−1)} (1−θ)^{(β−1)}
$$

This is saying that $\theta \sim \text{Beta}(\alpha,\beta)$. A beta distribution is a very good distribution for estimating probabilities, as it is bounded strictly between 0 and 1 and has two params that control shape.

The binomial likelihood is just a bernoulli random variable with k successes out of n trials, the constant infront is just the number of ways that this can happen which makes it binomial. So

$$
p(data|\theta) = p(\text{k successes out of n trials} | \theta) \propto \theta^{k} (1-\theta)^{n-k} \sim \text{Beta}(k+1,n-k+1)
$$

So then our conditional probability 

$$
p(\theta | data) \propto p(data | \theta)p(\theta) = \theta^{k} (1-\theta)^{n-k} θ^{(α−1)} (1−θ)^{(β−1)} \\ = \theta^{k+\alpha-1}(1-\theta)^{n+\beta-k-1} = \text{beta}(k+\alpha,n+\beta-k)
$$

Where $\alpha, \beta$ are the parameters of $\theta$'s distribution, and $k$ is the number of successful trials out of $n$ trials. Therefore we can derive a closed form of a posterior given that the prior and likelihood are both beta distributions.

## The expected value given the params

$$
\int_0^1 \theta p(\theta)d\theta = \int_0^1θ^{α} (1−θ)^{(β−1)}d\theta = \frac{\alpha}{\alpha + \beta}
$$

I.e. the ratio of successes to trials.

## The scenario

You're courtside, scouting a player's **first-serve-in percentage** —
exactly the kind of rate you'll be modeling properly in your final capstone,
just without the score-context and hierarchy yet. Today it's one player,
one rate, tracked as the match unfolds.

Before the match, a scout hands you a read: *"I think this player lands
about 65% of first serves — and I'd trust that read about as much as if
I'd already personally watched them serve 20 times."* That's a prior,
handed to you in plain English — your job is to turn it into
`Beta(alpha, beta)` numbers.

Then the match starts. You watch their first 10 first-serve attempts:
3 land in, 7 don't. You update your belief — and you can do that either by
reacting to each serve as it happens (serve-by-serve, live) or by waiting
until the changeover and updating once from the tally (3 of 10). Today's
whole point is confirming those give you the identical belief either way.

## Task

No formulas provided beyond what you derive above.

1. **`beta_update(alpha, beta, k, n)`** — apply the rule you just derived.
   `k` = serves landed in, `n` = serves attempted, for whatever stretch of
   the match you're updating on.
2. **`beta_mean(alpha, beta)`** and **`beta_variance(alpha, beta)`** —
   moments of a Beta distribution. Not given here — pull them from a
   reference, same as you did for the Binomial PMF on Day 2.
3. **`sequential_vs_batch(prior_alpha, prior_beta, serve_outcomes)`** —
   `serve_outcomes` is a list of 0/1 (0 = fault/long/wide, 1 = in), one
   entry per first serve, in the order they were played. Update once per
   serve, sequentially — live-scoring style, each posterior becoming the
   next serve's prior. Separately, update once on the full tally (total
   in, total attempted). Return both final `(alpha, beta)` pairs. These
   should match **exactly** — not approximately, there's no numerical
   method anywhere in this pipeline. (If your courtside read and your
   changeover read of the same player disagreed, something would be very
   wrong with Bayesian updating.)
4. **`pseudo_count_prior(believed_p, confidence)`** — turn the scout's
   sentence above into numbers: construct a `Beta(alpha, beta)` prior
   encoding "I believe the rate is about `believed_p`, and that belief is
   worth `confidence` equivalent prior serves already observed." Must
   satisfy `alpha + beta == confidence` and
   `alpha / (alpha + beta) == believed_p`. Test it on the scout's actual
   numbers: `pseudo_count_prior(0.65, 20)`.
5. **Compare against Day 2.** Import your Day 2 `make_grid`,
   `uniform_prior`, `binomial_likelihood`, `grid_posterior`, and
   `posterior_mean` functions. Using the first-set numbers above
   (`k=3, n=10`, flat/uniform prior), confirm the grid approximation's
   posterior mean converges toward your analytic `beta_mean` as
   `GRID_SIZE` grows — report the error at grid sizes 50, 500, and 5000.

## Self-check

- Sequential and batch updates must match **exactly** (floating-point
  equality, no tolerance). If they don't, something in your update is
  implicitly order-dependent, which the true update never is — reread
  what "addition" means for the α and β parameters before assuming it's a
  precision issue.
- `pseudo_count_prior` is checked by re-deriving `believed_p` and
  `confidence` back out of your returned `(alpha, beta)`.
- The grid-vs-analytic error must shrink **monotonically** across grid
  sizes 50 → 500 → 5000 — not merely "be small" at 5000. (Verified this is
  achievable with a correct Day 2 implementation before writing the
  assert — the errors drop by roughly 4 orders of magnitude at each step.)

## Definition of done

- [ ] Paper derivation actually done (nothing asserts this — but skipping
      it will make Day 6 much harder to reason about, not just code)
- [ ] All 5 functions implemented, no hard-coded numbers
- [ ] All self-checks pass
- [ ] You can explain, in terms of what addition means for α and β, why
      sequential and batch updating are forced to agree exactly
- [ ] You can explain, in one sentence, why an *exact* method existing
      today doesn't make Day 2's grid approximation pointless — this is
      the setup for Day 8, where you'll hit models where no exact method
      exists at all
