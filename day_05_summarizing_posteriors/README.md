# Day 5 — Summarizing the Posterior: Three Decisions

Less derivation today, more "you have a posterior — now what do you *do*
with it." Three case studies, each a real decision someone would actually
need to make. You'll reuse `beta_update` and `gamma_poisson_update` from
Days 3–4 to get the posteriors; today's new work is what you compute *from*
them, and — this time — what you can *see* in them.

## The case studies

**1. Should you halt the production line?** Historically, this line runs
about a 2% defect rate — call it `Beta(2, 98)` (2% mean, worth about 100
pseudo-inspected units, same construction as Day 3's `pseudo_count_prior`).
Today you inspect 60 units and find 2 defective. Company policy: halt the
line if you're **more than 90% confident** the true defect rate exceeds
2%. Do you halt it?

**2. Did the new deployment make things worse?** Historically, this
service errors about 3 times/hour — `Gamma(30, 10)` (mean 3, worth 10
hours of prior observation, Day 4's construction). After a deployment, you
watch it for 4 hours and log `[5, 6, 4, 7]` errors per hour. How confident
are you that the true error rate is now *worse* than the historical 3/hour?

**3. Does any of this even matter here?** Reuse your exact Day 4 heart-rate
posterior — `Normal(69.404762, 2.678571)`. Compute everything from cases 1
and 2 for this posterior too, and see what happens when the distribution
is symmetric instead of skewed.

## Two small derivations, then straight to applying them

**MAP (posterior mode).** Unlike mean and median, scipy doesn't hand you
a distribution's mode directly for an arbitrary shape — but for Beta and
Gamma, it's one derivative away. Take the log of the Beta density
(`∝ θ^(a−1)(1−θ)^(b−1)`), differentiate with respect to θ, set to zero,
solve. Same move for Gamma (`∝ λ^(a−1)e^(−bλ)`). Both only have a genuine
interior mode when their shape parameter is `> 1` — worth noticing what
the density looks like when it isn't (you won't need to handle that case
in code, just know why the formula would misbehave there).

### MAP derivation

Why would I wand to set a PDFs derivative to 0? That would give me the *most likely* value for a given parameter, which therefore should be the mode in a dataset... makes sense. We'll start with Beta:

$$
\frac{d}{d\theta}\Big[\theta^{a-1}(1-\theta)^{b-1}\Big]
= (a-1)\theta^{a-2}(1-\theta)^{b-1} \;-\; (b-1)\theta^{a-1}(1-\theta)^{b-2}
$$
$$= \theta^{a-2}(1-\theta)^{b-2}\Big[(a-1)(1-\theta) - (b-1)\theta\Big]$$
$$= \theta^{a-2}(1-\theta)^{b-2}\Big[(a-1) - \theta(a-1) - \theta(b-1)\Big]$$
$$= \theta^{a-2}(1-\theta)^{b-2}\Big[(a-1) - \theta(a+b-2)\Big]$$
$$(a-1) - \theta(a+b-2) = 0 \quad\Longrightarrow\quad \boxed{\theta_{\text{MAP}} = \dfrac{a-1}{a+b-2}}$$

This is **really** similar to the expected rate of success in a+b trials, though there are some odd quirks, especialy that -1 and -2... what do they mean? Well for a Beta distrbution remember $\alpha = a - 1, \beta = b - 1$ therefore the mean is literally $\frac{\alpha}{\alpha + \beta} = \frac{a-1}{a+b-2}$ which is the MAP! Interesting.

For the Gamma distribution we can find the posterior mode too, a little easier:

$$\frac{d}{d\theta}(\lambda^{a-1}e^{-b\lambda}) = (a-1)\lambda^{a-2}e^{-b\lambda} - b\lambda^{a-1}e^{-b\lambda}$$
$$= \lambda^{a-1}e^{-b\lambda}(\frac{(a-1)}{\lambda}-b) \implies \lambda_{\text{MAP}} = \frac{a-1}{b}$$

The intuition here is a little harder, but we can simplify this to literally $\frac{a}{b} - \frac{1}{b} = \text{mean} - \frac{1}{b}$, this is the wobble that we fill in with our gamma updates as we get more data, so eventually the MAP once again, will approach the mean value.

**HPD interval.** You already built an equal-tailed interval on Day 2, by
cutting the same probability off both tails. The **highest posterior
density (HPD)** interval instead finds the *narrowest* interval containing
your target probability mass — which, for a skewed distribution, is *not*
the same interval. Concretely: for a target mass (say 0.95), search over
where the lower cut `α` goes (with the upper cut then fixed at `α + 0.95`,
so the interval always contains exactly 95%), and find the `α` that
minimizes `ppf(α + 0.95) − ppf(α)`. `scipy.optimize.minimize_scalar` with
`method="bounded"` and `bounds=(0, 1 − mass)` will do this search for you
— your job is recognizing *what* to minimize, not writing an optimizer

### HPD Derivation

On day 2 we looked at the cumulative distribution, by looking at our confidence interval we divided one minus that by 2, then approached it from both ends of the distribution creating a symmetric two tailed credibility interval. However here we're looking to find the highest possible density in the smallest possible region. If we suppose we're looking for a mass of probability $m$ in an interval $I$ then in the CDF of our distribution (from which I comes) should vertically show something like CDF(x + a) - CDF(x) = m, we want to find the value of $a$ for which this is minimised, to do this we'd need to differentiate by a, which leaves us with this equation PDF(x+a) = 0, if we can solve this for a we get the width of the minimum credible interval... but not the staring point - do we need to optimise for both x and a?!

After a think, the solution is to make a = a(x), since it won't just be a constant a, it has to depend on x to make the interval work. So then the equation becomes:

$$
CDF(x-a(x)) - CDF(x) = m
$$

Differentiating both sides w.r.t x gives us:

$$
(1-a'(x))PDF(x-a) - PDF(x) = 0 
$$

We're looking for the value of x when $a(x)$ is minimised, so when $a'(x) = 0$  then we have that:

$$
PDF(x-a) = PDF(x)
$$

i.e draw a line horizontally from PDF(x) and wherever that intersects your density curve will give you a potential solution. Notice too how this doesn't depend on m...



## Task

1. **`beta_map(a, b)`** and **`gamma_poisson_map(a, b)`** — the two mode
   formulas you just derived.
2. **`equal_tailed_interval(dist, mass=0.95)`** — same idea as Day 2,
   using `dist.ppf` (a "frozen" scipy distribution, e.g. `stats.beta(4,
   156)`, gives you `.ppf`, `.cdf`, `.mean()`, `.median()` for free).
3. **`hpd_interval(dist, mass=0.95)`** — the narrowest-interval search
   described above.
4. **`prob_exceeds(dist, threshold)`** — the probability the true
   parameter is above some value. (One line, once you know which scipy
   method already computes "1 minus the CDF.")
5. **Answer the three case studies**, using the functions above:
   - Case 1: compute `prob_exceeds` for the defect-rate posterior against
     0.02. Do you halt the line? Say so in a print statement, in plain
     English, not just a number.
   - Case 2: compute `prob_exceeds` for the error-rate posterior against
     3.0. State your confidence the deployment made things worse.
   - Case 3: compute mean, median, MAP, equal-tailed, and HPD for the
     heart-rate posterior. What do you notice?
6. **Plot all three.** `plot_posterior_summary()` is provided in
   `exercise.py`, mostly complete — it calls your functions above, so once
   they're correct, running the file produces three saved PNGs (one per
   case study) showing the density curve, mean/median/MAP marked, and both
   interval types shaded. Open them. This is the actual point of today —
   look at where MAP sits relative to the mean on the skewed cases, and
   look at how the HPD shading compares to the equal-tailed shading.

## Self-check

- `beta_map` / `gamma_poisson_map` checked against worked values for both
  case studies' actual posteriors.
- `hpd_interval`'s width must never exceed `equal_tailed_interval`'s width
  for the same distribution and mass — that's not a coincidence to verify,
  it's what "HPD is the *narrowest* interval" mathematically guarantees.
  For the skewed case studies, it should be **meaningfully** narrower, not
  just tied — if your two intervals come out identical for Case 1 or 2,
  something's returning equal-tailed bounds under a different name.
- For Case 3 (Normal, symmetric), HPD and equal-tailed should coincide
  almost exactly — if they don't for a symmetric posterior, that's a bug,
  not an interesting result.
- `prob_exceeds` checked against exact reference values for both decision
  case studies.

## Definition of done

- [ ] Both mode formulas derived on paper first
- [ ] All 4 functions implemented, all self-checks pass
- [ ] All three PNGs generated and actually looked at
- [ ] You can state, from the plots (not the numbers), why MAP sits to the
      *left* of the mean on both skewed posteriors, and why that's not a
      coincidence
- [ ] You've made an actual halt/no-halt call for Case 1 and can defend it
      in one sentence to someone who only trusts the mean, not the
      probability
