# adaptive_fitness.py
# Adaptive fitness functions for Set Cover GA.
#
# We keep the real objective as the solution cost,
# but use an adjusted selection score during parent selection.

import random
from distance import get_distance_function


def objective_quality_scores(costs):
    """
    Convert minimization cost into normalized quality score.

    best cost  -> score close to 1
    worst cost -> score close to 0
    """
    best_cost = min(costs)
    worst_cost = max(costs)

    if worst_cost == best_cost:
        return [1.0 for _ in costs]

    return [
        (worst_cost - cost) / (worst_cost - best_cost)
        for cost in costs
    ]


def novelty_scores(population, distance_name="jaccard", sample_size=30, sample_seed=0):
    """
    Novelty score = average distance from other sampled individuals.
    Higher value means the chromosome is more different / novel.
    """
    if len(population) < 2:
        return [0.0 for _ in population]

    rng = random.Random(sample_seed)
    distance_fn = get_distance_function(distance_name)

    scores = []

    for i in range(len(population)):
        candidates = [j for j in range(len(population)) if j != i]

        if sample_size is None or sample_size >= len(candidates):
            sampled = candidates
        else:
            sampled = rng.sample(candidates, sample_size)

        if not sampled:
            scores.append(0.0)
            continue

        avg_distance = sum(
            distance_fn(population[i], population[j])
            for j in sampled
        ) / len(sampled)

        scores.append(avg_distance)

    return scores


def age_scores(ages, age_threshold=10):
    """
    Age score gives higher score to older surviving individuals.
    This tests whether preserving old lineages helps the GA.
    """
    if not ages:
        return []

    if age_threshold <= 0:
        return [1.0 for _ in ages]

    return [
        max(0.0, min(1.0, age / age_threshold))
        for age in ages
    ]


def compute_adaptive_fitness_scores(
    mode,
    population,
    costs,
    ages,
    alpha=0.8,
    distance_name="jaccard",
    novelty_sample_size=30,
    sample_seed=0,
    age_threshold=10,
):
    """
    Compute adaptive selection score:

        F(x,t) = alpha * objective_quality(x) + (1 - alpha) * g(x,t)

    Supported modes:
    - novelty
    - age
    """
    quality = objective_quality_scores(costs)

    if mode == "novelty":
        g_scores = novelty_scores(
            population=population,
            distance_name=distance_name,
            sample_size=novelty_sample_size,
            sample_seed=sample_seed,
        )

    elif mode == "age":
        g_scores = age_scores(
            ages=ages,
            age_threshold=age_threshold,
        )

    else:
        raise ValueError("Unknown adaptive fitness mode: " + str(mode))

    alpha = max(0.0, min(1.0, alpha))

    adaptive_scores = [
        alpha * q + (1.0 - alpha) * g
        for q, g in zip(quality, g_scores)
    ]

    info = {
        "avg_g_score": sum(g_scores) / len(g_scores) if g_scores else 0.0,
        "max_g_score": max(g_scores) if g_scores else 0.0,
        "min_g_score": min(g_scores) if g_scores else 0.0,
        "avg_adaptive_score": sum(adaptive_scores) / len(adaptive_scores) if adaptive_scores else 0.0,
        "max_adaptive_score": max(adaptive_scores) if adaptive_scores else 0.0,
        "min_adaptive_score": min(adaptive_scores) if adaptive_scores else 0.0,
    }

    return adaptive_scores, info