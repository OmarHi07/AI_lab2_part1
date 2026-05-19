# mutation.py
# Optimized bit-flip mutation.
# Instead of checking every gene with random.random(), we sample how many bits to flip
# from Binomial(n, mutation_rate), then flip that many random positions.

import random
import numpy as np


def bit_flip_mutation(chromosome, mutation_rate):
    """
    Apply bit-flip mutation to a chromosome.
    mutation_rate is still the probability that each gene flips.
    """
    mutated = chromosome[:]
    n = len(mutated)

    if mutation_rate <= 0:
        return mutated

    if mutation_rate >= 1:
        return [1 - gene for gene in mutated]

    # This gives the same distribution as independent bit mutation:
    # K ~ Binomial(n, p), then choose K positions uniformly.
    k = np.random.binomial(n, mutation_rate)

    if k > 0:
        for i in random.sample(range(n), k):
            mutated[i] = 1 - mutated[i]

    return mutated
