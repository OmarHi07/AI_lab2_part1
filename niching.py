# niching.py
# Fitness Sharing / Niching for Set Cover GA.
#
# The goal is to reduce the selection advantage of individuals
# that are located in crowded regions of the population.

from distance import get_distance_function
import random


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

def threshold_speciation(
    population,
    distance_name="jaccard",
    threshold=0.35,
):
    """
    Divide the population into species using threshold speciation.

    Each species has a representative.
    A chromosome joins the first species whose representative is close enough.
    If no representative is close enough, a new species is created.

    This gives a variable number of species.
    """

    if not population:
        return [], [], {
            "species_count": 0,
            "avg_species_size": 0.0,
            "max_species_size": 0,
            "min_species_size": 0,
        }

    distance_fn = get_distance_function(distance_name)

    representatives = []
    species_members = []
    species_of_individual = [-1] * len(population)

    for i, individual in enumerate(population):
        assigned = False

        for species_id, representative_index in enumerate(representatives):
            distance = distance_fn(individual, population[representative_index])

            if distance <= threshold:
                species_members[species_id].append(i)
                species_of_individual[i] = species_id
                assigned = True
                break

        if not assigned:
            representatives.append(i)
            species_members.append([i])
            species_of_individual[i] = len(species_members) - 1

    species_sizes = [len(members) for members in species_members]

    info = {
        "species_count": len(species_members),
        "avg_species_size": sum(species_sizes) / len(species_sizes),
        "max_species_size": max(species_sizes),
        "min_species_size": min(species_sizes),
    }

    return species_of_individual, species_members, info


def choose_species_for_reproduction(species_members):
    """
    Choose a species with probability proportional to its size.
    Larger species get more reproduction chances, but small species still survive.
    """
    total_size = sum(len(members) for members in species_members)

    if total_size == 0:
        return []

    pick = random.randint(1, total_size)
    running_sum = 0

    for members in species_members:
        running_sum += len(members)
        if pick <= running_sum:
            return members

    return species_members[-1]


def tournament_selection_from_indices(population, fitnesses, member_indices, tournament_size=3):
    """
    Tournament selection restricted to a specific species.
    This keeps mating mostly inside the same species.
    """
    if not member_indices:
        return population[random.randrange(len(population))][:]

    actual_size = min(tournament_size, len(member_indices))
    candidates = random.sample(member_indices, actual_size)

    best_idx = max(candidates, key=lambda i: fitnesses[i])
    return population[best_idx][:]