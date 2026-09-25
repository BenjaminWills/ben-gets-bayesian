# Day 4 — Conjugate Priors: Normal-Normal & Gamma-Poisson

Two new conjugate pairs today. Same recipe as Day 3 (prior × likelihood,
recognize the shape) — but this time, work through the questions below on
paper before you look at anything else. Nothing in this file states the
update formulas. If you get stuck, that's the point where you go back to
the questions, not where you search for the formula.

## The scenarios

**A — your resting heart rate.** Your smartwatch measures resting heart
rate every morning, with some known measurement noise (the manufacturer
publishes it — say, ±3 bpm standard deviation). Before you owned the
watch, you'd have guessed your resting heart rate was somewhere around the
population average, with real uncertainty about your own number
specifically. Each morning's reading should update that belief.

**B — bugs per code review.** Across many students you've coached, you've
built a sense of roughly how many bugs turn up per pull request, on
average. For a *specific* new student, you start from that general
experience, then update as you actually review their first few PRs.

Notice what's different about these two: A is a continuous measurement
(heart rate could be 68.3, 68.31, anything), B is a count (0, 1, 2 bugs —
never 1.5). That's exactly why they need two different likelihood
families, and why today isn't just "Day 3 again."

## Work it out — Normal-Normal (scenario A)

1. Write down the Normal prior density for the true rate μ:
   `p(μ) ∝ exp(−(μ−μ₀)² / 2τ₀²)`. Write down the likelihood for `n`
   independent readings `x₁...xₙ`, each `Normal(μ, σ²)` with **known** σ².
   Multiply them together — same move as Day 3, just with exponentials
   instead of powers of θ.

2. **Before doing any algebra**, answer these from intuition, and write
   your answers down somewhere you'll check them later:
   - If your prior belief about your own heart rate were extremely
     vague (τ₀² huge) and you'd taken exactly one reading, what should the
     posterior mean roughly equal?
   - If your prior belief were extremely confident (τ₀² tiny) and you'd
     taken a thousand readings that all disagreed with it, what should the
     posterior mean roughly equal?

3. Now actually multiply the prior and likelihood, expand the exponent,
   and group everything as a quadratic in μ (this is "completing the
   square" — you did this in school, it just didn't come with a
   probability distribution attached). You should land on another Normal
   distribution. Its variance and mean are both expressible cleanly — but
   not in terms of τ₀² and σ² directly. Try rewriting everything in terms
   of **1/τ₀²** and **1/σ²** instead. Does a natural "weighted average"
   fall out? What's it weighted by?

4. Check your derived formula against your two intuitive predictions from
   step 2. If it doesn't match either one, you've made an algebra error —
   go find it before writing any code.

### Derivation

What do we know: We know that we're estimating the mean heart rate given some data. Our prior belief is that this mean heart rate is distributed normally with some mean $\mu_0$  and variance $\tau_0^2$, and our likelihood here is defined with a normal distribution with mean $\mu$ (what we want to estimate) and known standard deviation $\sigma^2$. The more estimates we get, the more our likelihood function will dominate the posterior distribution. See below to see the conjugate priors come about.

The PDF for a normal variable with mean $\mu$ and variance $\sigma^2$ looks like:

$$
p(x) ∝ \exp(\frac{−(x-\mu_0)^2}{2\sigma²})
$$

The likelihood of n independently normally distributed events is just a product, due to their independence:

$$
p(data| \mu,\sigma^2) \propto \Pi_{i =1}^{n}\exp(\frac{−(x_i-\mu)^2}{2\sigma²}) = \exp(\sum_{i=1}^{n}\frac{−(x_i-\mu)^2}{2\sigma²})
$$

So our posterior looks like this:

$$
p(\mu| data, \sigma) \propto p(data | \mu, \sigma^2)p(\mu|\mu_0) = \exp(\sum_{i=1}^{n}\frac{−(x_i-\mu)^2}{2\sigma²})\exp(\frac{−(\mu-\mu_0)^2}{2\tau_0^2}) \\ = \exp(\sum_{i=1}^{n}\frac{−(x_i-\mu)^2}{2\sigma²}+\frac{−(\mu-\mu_0)^2}{2\tau_0^2}) \implies \\ \text{precision: } \; P = \frac{1}{\text{variance}}= \frac{n}{\sigma^2}+\frac{1}{\tau_0^2}\\
\mu \mid \text{data} \;\sim\; \mathcal{N}\!\left(\frac{1}{P}\left(\frac{n\bar{x}}{\sigma^2}+\frac{\mu_0}{\tau_0^2}\right),\; \frac{1}{P}\right)
$$

From this we see that as n increases the $\frac{n}{\sigma^2}$ term starts to dominate the precision, and grows large, which means the variance decreases and thus our certainty increases. This makes sense based on our model getting smarter the more data it ingests.

One thought I had which was rather annoying, is that the mean when all is said an expanded becomes very dominated by the sample mean, which implies that we just approach the sample mean $\bar{x}$ (it looks like $\frac{n\bar{x}\,\tau_0^2 + \mu_0\sigma^2}{n\tau_0^2 + \sigma^2}$ which approaches $\bar{x}$ as n gets very large.). You may. be thinking: why not just take the frequentist approach, wouldn't that be way easier, however; the Baysean approach lets you consider your uncertainty for small sample sizes where we have a lot more uncertainty.

## Work it out — Gamma-Poisson (scenario B)

Same three-line trick as Beta-Binomial, so this should go quickly *if*
you actually do it rather than assume it transfers:

- Prior: λ ~ Gamma(a, b), so `p(λ) ∝ λ^(a−1) e^(−bλ)`
- Data: `n` independent Poisson(λ) counts, total observed count `T`, so
  `p(data|λ) ∝ λ^T e^(−nλ)` (as before, the part that doesn't involve λ
  gets dropped — it's absorbed into the normalizing constant)

Multiply, and read off the new Gamma parameters. If you find yourself
just pattern-matching "it's probably `a+something, b+something`" without
writing out the multiplication, do it properly anyway — Day 6 is going to
ask you to reason about *why* these updates behave the way they do, and
pattern-matching won't survive that.

### Derivation

We know that $P(\lambda | \text{data}) = p(\text{data} | \lambda)p(\lambda) \propto \lambda^{a-1}e^{-b\lambda}\lambda^{T}e^{-n\lambda} = \lambda^{a+T-1}e^{-\lambda(n+b)} \sim \text{Gamma}(a+T,n+b)$ where $T=\sum{x_i}$

But what does this mean? The mean of the gamma distribution is $\frac{a+T}{n+b} = \frac{\frac{a}{n} +  \frac{T}{n}}{1 + \frac{b}{n}} = \frac{T}{n} \rightarrow \lambda^*$ (the actual parameter) as $n \rightarrow \infin$ by the law of large numbers. So we've reached the same point, baysean methods lead to the same conclusions as frequentist methods given enough data, but for low amounts of data (and some more complicated things to come) they win.


## Task

1. **`normal_update(prior_mean, prior_var, data, known_var)`** — `data` is
   a list of one or more readings. Returns `(posterior_mean,
   posterior_var)`.
2. **`gamma_poisson_update(prior_a, prior_b, counts)`** — `counts` is a
   list of one or more observed counts. Returns `(new_a, new_b)`.
3. **`sequential_vs_batch_normal(prior_mean, prior_var, readings,
   known_var)`** — same idea as Day 3: update one reading at a time vs.
   all at once. **Don't assume the equality is exact this time** — Day 3's
   update was pure integer addition; this one involves dividing by
   variances. Investigate whether repeated division and floating-point
   arithmetic preserve exact equality the way integer addition did, and
   pick a comparison (exact, or some tolerance) that's actually justified
   by what you find — not just copy-pasted from Day 3.
4. **`effective_prior_sample_size(prior_var, known_var)`** — Day 3's
   `pseudo_count_prior` treated `alpha + beta` as "how many observations
   this prior is worth." Is there an equivalent single number here? Look
   at what you derived in step 3 above: the prior contributes `1/prior_var`
   to the posterior's precision, and *each* data point contributes
   `1/known_var`. How many data points' worth of precision is the prior
   supplying, by itself?

## Self-check

- `normal_update` and `gamma_poisson_update` are checked against specific
  worked cases (see `exercise.py`) — get those right and the general
  formula is very likely right too.
- Two **behavioral** checks for `normal_update`, matching your intuition
  questions from step 2 above: an extremely wide prior should let one data
  point completely determine the posterior; an extremely narrow prior
  should barely budge no matter how much (conflicting) data arrives.
- `sequential_vs_batch_normal` is checked with a small numerical tolerance,
  not exact equality — if your own reasoning above concluded it *should*
  be bit-exact, that's worth resolving before you decide the check is
  "too loose."
- `gamma_poisson_update`'s sequential-vs-batch equality **is** checked
  exactly, no tolerance — think about why that's allowed here when it
  isn't for the Normal case, given both are "just addition" at heart.

## Definition of done

- [ ] Both derivations actually done on paper first
- [ ] All 4 functions implemented, no hard-coded numbers
- [ ] All self-checks pass
- [ ] You can state, in your own words, what quantity a Normal-Normal
      prior's "weight" really is (not variance — something derived from it)
- [ ] You can explain why `gamma_poisson_update`'s sequential/batch
      equality is exact while `normal_update`'s isn't, even though both
      updates are "the prior's contribution plus each data point's
      contribution"
- [ ] (Stretch) Day 2's grid approximation never actually required a
      Binomial likelihood — it just needed *some* likelihood function
      evaluated on a grid. Try swapping in a Poisson likelihood over a
      grid of λ values instead of θ, and confirm it converges to your
      analytic Gamma-Poisson posterior mean, the same way Day 3 checked
      convergence for Beta-Binomial.
