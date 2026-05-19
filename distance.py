# distance.py
# Distance functions for Set Cover chromosomes.
#
# A chromosome is a binary list:
# 1 = the set is selected
# 0 = the set is not selected


import random


def hamming_distance(a, b):
    """
    Count how many positions are different between two chromosomes.
    Example:
    a = [1, 0, 1]
    b = [1, 1, 0]
    distance = 2
    """
    return sum(x != y for x, y in zip(a, b))


def normalized_hamming_distance(a, b):
    """
    Hamming distance divided by chromosome length.

    This gives a value between 0 and 1:
    0 = identical chromosomes
    1 = completely different chromosomes
    """
    if len(a) == 0:
        return 0.0

    return hamming_distance(a, b) / len(a)


def selected_indices(chromosome):
    """
    Return the indices of the selected sets.
    """
    return {i for i, gene in enumerate(chromosome) if gene == 1}


def jaccard_distance(a, b):
    """
    Jaccard distance between two Set Cover chromosomes.

    This compares only the selected sets.

    distance = 1 - |A intersection B| / |A union B|

    This is useful for Set Cover because chromosomes are often sparse,
    meaning most genes are 0.
    """
    selected_a = selected_indices(a)
    selected_b = selected_indices(b)

    union = selected_a | selected_b

    if not union:
        return 0.0

    intersection = selected_a & selected_b

    return 1.0 - (len(intersection) / len(union))


def get_distance_function(distance_name):
    """
    Return a distance function according to its name.
    """
    if distance_name == "hamming":
        return normalized_hamming_distance

    if distance_name == "jaccard":
        return jaccard_distance

    raise ValueError("Unknown distance function: " + str(distance_name))


def sampled_average_distance(population, distance_name="jaccard", sample_pairs=50, sample_seed=0):
    """
    Estimate the average distance in the population using random pairs.

    We use sampling instead of all pairs because all-pairs comparison is expensive:
    population_size^2 comparisons.
    """
    if len(population) < 2:
        return 0.0

    rng = random.Random(sample_seed)
    distance_fn = get_distance_function(distance_name)

    total = 0.0
    count = 0

    for _ in range(sample_pairs):
        i, j = rng.sample(range(len(population)), 2)
        total += distance_fn(population[i], population[j])
        count += 1

    return total / count if count > 0 else 0.0