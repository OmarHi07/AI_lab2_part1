
#This module contains helper functions for creating and copying chromosomes.

#A chromosome is represented as a list of 0/1 values:
# 1 means the set is selected
# 0 means the set is not selected

import random


def random_chromosome(n, activation_prob=0.5):
    """
    Create a random binary chromosome.
    Args:
        n (int): Number of genes.
        activation_prob (float): Probability that a gene is 1.

    Returns:
        list[int]: Random chromosome.
    """
    return [1 if random.random() < activation_prob else 0 for _ in range(n)]


def copy_chromosome(chromosome):
    """
    Create a copy of a chromosome.

    Args:
        chromosome (list[int]): Input chromosome.

    Returns:
        list[int]: Copied chromosome.
    """
    return chromosome[:]