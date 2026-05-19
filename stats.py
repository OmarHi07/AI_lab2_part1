#stats.py

#This module computes statistics for each generation.

#Tracked values may include:
#- best fitness
#- average fitness
#- worst fitness
#- standard deviation
#- diversity measures


import math

import random

def sampled_hamming_distance(population, sample_pairs=50, sample_seed=0):
    """
    Approximate average Hamming distance using random pairs.
    Uses a local random generator so it does not affect the GA randomness.
    """
    if len(population) < 2:
        return 0.0

    rng = random.Random(sample_seed)

    total = 0
    count = 0

    for _ in range(sample_pairs):
        i, j = rng.sample(range(len(population)), 2)
        dist = sum(1 for a, b in zip(population[i], population[j]) if a != b)
        total += dist
        count += 1

    return total / count if count > 0 else 0.0


def unique_chromosome_ratio(population):
    """
    Measures how many different chromosomes exist in the population.
    Value close to 1 means high diversity.
    Value close to 0 means many duplicates.
    """
    if not population:
        return 0.0

    unique = len({tuple(chromosome) for chromosome in population})
    return unique / len(population)


def average(values):
    #Compute the arithmetic mean.
    #Args:
    #    values (list[float]): Input values.
    #Returns:
    #    float: Mean value.
    if not values:
        return 0.0
    return sum(values) / len(values)


def std_dev(values):
    #Compute the standard deviation.
    #Args:
    #    values (list[float]): Input values.
    #Returns:
    #    float: Standard deviation.
    if not values:
        return 0.0

    avg = average(values)
    variance = sum((x - avg) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def average_hamming_distance(population):
    #Compute average pairwise Hamming distance in the population.
    #Args:
    #    population (list[list[int]]): Population of chromosomes.
    #Returns:
    #    float: Average Hamming distance.
    if len(population) < 2:
        return 0.0

    total = 0
    count = 0

    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            dist = sum(1 for a, b in zip(population[i], population[j]) if a != b)
            total += dist
            count += 1

    return total / count if count > 0 else 0.0


#[1,0,1,0,1] → 3
#[1,1,1,1,1] → 5
#[0,0,0,0,1] → 1
def active_genes_std(population):
    #Compute the standard deviation of the number of active genes.
    #Args:
    #    population (list[list[int]]): Population of chromosomes.
    #Returns:
    #    float: Standard deviation of the number of selected sets.

    active_counts = [sum(chromosome) for chromosome in population]
    return std_dev(active_counts)