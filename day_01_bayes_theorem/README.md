# Day 1 — The Bayesian Mindset & Bayes' Theorem

## Concept

Bayesian inference treats an unknown as a random variable with a
distribution, not a fixed value estimated by some procedure. Everything
follows from the definition of conditional probability:

```
P(θ | data) = P(data | θ) · P(θ) / P(data)
   posterior  ∝   likelihood   ×  prior
```

`P(data)` is just a normalizing constant — the sum (or integral) over all
hypotheses of `likelihood × prior` — which is why, for a discrete hypothesis
space, you can compute the posterior by:

1. Multiplying prior × likelihood for every hypothesis.
2. Dividing each by the total, so they sum to 1.

That's the whole algorithm for today. The hard part is never the arithmetic
— it's setting up the prior and likelihood correctly.

### Intuition

More generally Bayes' theorem is that the probability of event A happening given that we know B has happened is equal to the probability that A and B happen, divided by the probability that B happened at all. This intuitively makes sense as we're looking for the probability that A and B happen given that B already happens.

This thread gives a really good intuition: https://stats.stackexchange.com/questions/239014/bayes-theorem-intuition. Generally P(B|A) in a venn diagram is the proportion of the B bubble that is taken up by P(B and A).

Another useful fact is that P(A) = P(A|B)P(B) + P(A| not B)P(not B).

P(A|B) = # times A and B happen / # times that B happen = (# times A and B happen / N) / (times that B happen / N) = P(A,B)/P(B).

## Task

Open `exercise.py`. Implement `bayes_update()`, then use it to solve:

1. **The medical test paradox** — a disease with 1% prevalence, a test
   that's 95% sensitive and 90% specific. Given a positive result, what's
   the probability of actually having the disease? (It's lower than most
   people's intuition — that gap is the whole point of the exercise.)
2. **The Cookie Problem** — two bowls with different vanilla/chocolate
   ratios; you draw one cookie at random from a randomly chosen bowl and
   observe its type. Infer which bowl it came from.

Run the file directly:

```bash
python exercise.py
```

The asserts at the bottom check your posteriors against the known correct
values — if they pass, you've got it right. If they fail, don't guess at
the numbers; re-derive the likelihood terms by hand first (write out
"P(observed data | this hypothesis)" in words for each hypothesis before
touching code).

## Stretch (optional)

Bayes' theorem has an equivalent **odds form**:

```
posterior odds = (likelihood ratio) × (prior odds)
```

Implement `medical_test_via_odds()` and confirm it agrees with your answer
from `medical_test_posterior()`. This form is worth having, because it's
often how Bayesian updating gets described in applied fields (e.g. medicine,
forensics) — "how much should this new evidence shift my odds?"

## Definition of done

- [ ] `bayes_update()` implemented (no hard-coded numbers inside it — it
      should work for any prior/likelihood dict you pass in)
- [ ] Both asserts in `exercise.py` pass
- [ ] You can explain out loud, without notes, why the medical test posterior
      is much lower than 95% (the sensitivity) — this is the single most
      important intuition from today
- [ ] (Stretch) odds-form implementation agrees with the direct calculation
