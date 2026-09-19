"""
Day 1 -- The Bayesian Mindset & Bayes' Theorem
================================================
Two classic problems to build intuition for discrete Bayesian updating.

Fill in the TODOs. Run this file directly (`python exercise.py`) -- the
asserts at the bottom will tell you whether your posteriors are correct.
"""


def bayes_update(prior: dict, likelihood: dict) -> dict:
    """
    Compute the posterior distribution over a discrete hypothesis space.

    Parameters
    ----------
    prior : dict
        Mapping {hypothesis: P(hypothesis)}. Values should sum to 1.
    likelihood : dict
        Mapping {hypothesis: P(observed data | hypothesis)}.
        (Same keys as `prior`.)

    Returns
    -------
    dict
        Mapping {hypothesis: P(hypothesis | observed data)}, normalized
        so the values sum to 1.
    """
    #   1. For each hypothesis h, compute unnormalized = prior[h] * likelihood[h]
    unormalised_posteriors = {}
    for hypothesis in prior:
        unormalised_posterior = prior[hypothesis] * likelihood[hypothesis]
        unormalised_posteriors[hypothesis] = unormalised_posterior
    #   2. Sum the unnormalized values across all hypotheses, P(data) = SUM(P(hypothesis)P(data|hypothesis))
    normalisation_factor = sum(unormalised_posteriors.values())
    #   3. Divide each unnormalized value by that sum
    normalised_posterior = {hypothesis: probability /
                            normalisation_factor for hypothesis, probability in unormalised_posteriors.items()}
    return normalised_posterior


# ---------------------------------------------------------------------------
# Problem 1: The medical test paradox
# ---------------------------------------------------------------------------
# A disease has 1% prevalence in the population.
#   Sensitivity: P(positive | disease)    = 0.95
#   Specificity: P(negative | no disease) = 0.90
#
# A random person tests positive. What's P(disease | positive)?

def medical_test_posterior():
    prior = {"disease": 0.01, "no_disease": 0.99}
    likelihood = {
        "disease": 0.95,        # P(positive | disease)
        "no_disease": 1 - 0.90,  # P(positive | no disease) = 1 - specificity
    }
    return bayes_update(prior, likelihood)


# ---------------------------------------------------------------------------
# Problem 2: The Cookie Problem
# ---------------------------------------------------------------------------
# Bowl 1: 30 vanilla, 10 chocolate  -> P(vanilla | Bowl 1) = 30/40
# Bowl 2: 20 vanilla, 20 chocolate  -> P(vanilla | Bowl 2) = 20/40
#
# You pick a bowl at random, draw one cookie without looking -- it's vanilla.
# What's P(Bowl 1 | vanilla)?

def cookie_problem_posterior():
    prior = {"bowl1": 0.5, "bowl2": 0.5}
    likelihood = {
        "bowl1": 30 / 40,  # P(vanilla | bowl1)
        "bowl2": 20 / 40,  # P(vanilla | bowl2)
    }
    return bayes_update(prior, likelihood)


# ---------------------------------------------------------------------------
# Stretch: posterior odds = Bayes factor x prior odds
# ---------------------------------------------------------------------------

def medical_test_via_odds():
    """
    Recompute the medical test posterior using the odds form of Bayes'
    theorem, and confirm it agrees with medical_test_posterior().

        prior odds        = P(disease) / P(no_disease)
        likelihood ratio   = P(+|disease) / P(+|no_disease)
        posterior odds     = likelihood ratio * prior odds
        P(disease | +)     = posterior_odds / (1 + posterior_odds)
    """
    # TODO (stretch): implement this and compare against medical_test_posterior()
    raise NotImplementedError("Stretch: implement the odds-form calculation")


if __name__ == "__main__":
    med = medical_test_posterior()
    print(f"P(disease | positive test) = {med['disease']:.4f}")
    assert abs(med["disease"] -
               0.0876) < 0.001, "Check your medical test posterior."

    cookie = cookie_problem_posterior()
    print(f"P(Bowl 1 | vanilla)        = {cookie['bowl1']:.4f}")
    assert abs(cookie["bowl1"] -
               0.6) < 0.001, "Check your cookie problem posterior."

    print("\nDay 1 exercises passed.")

    # Uncomment once you've implemented the stretch goal:
    # odds_result = medical_test_via_odds()
    # assert abs(odds_result - med["disease"]) < 0.0001, "Odds form should agree."
    # print("Stretch goal passed -- odds form agrees with direct calculation.")
