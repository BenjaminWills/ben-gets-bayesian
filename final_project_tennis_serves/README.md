# Final Project — Predicting Serve Direction

**Question:** given a server's history and the current score context, what's
the posterior distribution over their next serve direction (wide / body / T)?

**Data source:** [Jeff Sackmann's Match Charting Project](https://github.com/JeffSackmann/tennis_MatchChartingProject)
— shot-by-shot data (type, direction, depth, outcome) for 5,000+ pro matches.
No API needed, just clone the repo and load the CSVs.

## Staged plan

1. **Plain Dirichlet-Multinomial** — one player, no context. "How does this
   player serve overall?" Direct extension of Day 3's Beta-Binomial to 3+
   categories: Dirichlet prior over (P(wide), P(body), P(T)), conjugate
   update by adding observed counts per direction.
2. **Add context, go hierarchical** — split by deuce/ad court, first/second
   serve, break point or not. Data gets sparse per context fast, which is
   exactly the partial-pooling problem from Day 11: a player's break-point
   tendency should shrink toward their overall tendency until there's enough
   break-point data to override it.
3. **Evaluate** — hold out some points from the same player, check whether
   the posterior predictive distribution over direction assigns reasonable
   probability to what they actually did.
4. **(Stretch)** reframe as a Bayesian multinomial logistic regression —
   direction as a function of score state, previous shot direction, serve
   number — and compare against the hierarchical bucket model with LOO
   (Day 13).

## Scope note

Deliberately scoped to **serve direction**, not general rally shot
placement — serves are a clean, well-defined categorical outcome at a
specific decision point. Full rally shot prediction (long dependency
chains, adversarial dynamics) is a much harder problem and is out of scope
unless the core model comes together with time to spare.
