# Bayesian Statistics: Theory & Implementation

My progress log for a self-directed, 2-week Bayesian statistics course —
exact inference in Week 1, computational methods (MCMC, PyMC) in Week 2,
capped off with a capstone predicting tennis serve direction from prior
information.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Each day lives in its own folder: a `README.md` with the concept recap and
task, and either an `exercise.py` (run it directly — self-checking asserts
will tell you if you're right) or a notebook, for days that need plots.

## Progress

### Week 1 — Foundations: Exact Bayesian Inference
- [ ] Day 1 — The Bayesian Mindset & Bayes' Theorem
- [ ] Day 2 — Grid Approximation
- [ ] Day 3 — Conjugate Priors: Beta-Binomial
- [ ] Day 4 — Conjugate Priors: Normal-Normal & Gamma-Poisson
- [ ] Day 5 — Summarizing the Posterior (credible intervals, HPD)
- [ ] Day 6 — Choosing Priors & Sensitivity Analysis
- [ ] Day 7 — **Checkpoint project:** Bayesian A/B Testing Tool

### Week 2 — Computation: MCMC and Modern Practice
- [ ] Day 8 — Why We Need Sampling
- [ ] Day 9 — Metropolis-Hastings From Scratch
- [ ] Day 10 — Probabilistic Programming with PyMC + Diagnostics
- [ ] Day 11 — Hierarchical (Multilevel) Models
- [ ] Day 12 — Bayesian Linear Regression
- [ ] Day 13 — Model Checking & Comparison (PPCs, WAIC/LOO)
- [ ] Day 14 — **Final project:** see `final_project_tennis_serves/`

## Final project

Predicting a server's next serve direction (wide / body / T) using a
Dirichlet-Multinomial model, extended hierarchically across score contexts
(deuce/ad court, serve number, break point or not). Details and data source
in `final_project_tennis_serves/README.md`.
