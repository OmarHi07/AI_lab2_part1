# niching.py
# Fitness Sharing / Niching for Set Cover GA.
#
# The goal is to reduce the selection advantage of individuals
# that are located in crowded regions of the population.

from distance import get_distance_function


def sharing_function(distance, sigma_share, alpha=1.0):
    """
    Sharing function from the lecture:

    sh(d_ij) = 1 - (d_ij / sigma_share)^alpha   if d_ij < sigma_share
             = 0                                otherwise
    """
    if sigma_share <= 0:
        return 0.0

    if distance >= sigma_share:
        return 0.0

    return 1.0 - ((distance / sigma_share) ** alpha)


def fitness_sharing_selection_scores(
    population,
    costs,
    distance_name="jaccard",
    sigma_share=0.35,
    alpha=1.0,
    sample_size=30,
    sample_seed=0,
):
    """
    Compute shared selection scores.

    Exact fitness sharing compares every individual with every other individual.
    That is O(population_size^2), which is slow.

    This version estimates the niche count using a fixed sample of neighbors.
    Complexity becomes approximately O(population_size * sample_size).
    """

    import random

    if not population:
        return [], {
            "avg_niche_count": 0.0,
            "max_niche_count": 0.0,
            "min_niche_count": 0.0,
        }

    distance_fn = get_distance_function(distance_name)
    rng = random.Random(sample_seed)

    worst_cost = max(costs)
    raw_scores = [(worst_cost - cost + 1.0) for cost in costs]

    shared_scores = []
    niche_counts = []

    population_size = len(population)

    for i in range(population_size):
        if sample_size is None or sample_size >= population_size - 1:
            sampled_indices = [j for j in range(population_size) if j != i]
            scaling_factor = 1.0
        else:
            candidates = [j for j in range(population_size) if j != i]
            sampled_indices = rng.sample(candidates, sample_size)
            scaling_factor = (population_size - 1) / sample_size

        # Self sharing is always 1.
        sampled_niche_count = 1.0

        for j in sampled_indices:
            distance = distance_fn(population[i], population[j])
            sampled_niche_count += sharing_function(distance, sigma_share, alpha)

        # Scale only the neighbor contribution, not the self contribution.
        niche_count = 1.0 + (sampled_niche_count - 1.0) * scaling_factor

        if niche_count <= 0:
            niche_count = 1.0

        shared_score = raw_scores[i] / niche_count

        shared_scores.append(shared_score)
        niche_counts.append(niche_count)

    info = {
        "avg_niche_count": sum(niche_counts) / len(niche_counts),
        "max_niche_count": max(niche_counts),
        "min_niche_count": min(niche_counts),
    }

    return shared_scores, info