
#This module implements crossover operators for the genetic algorithm.

#point = 2
#parent1 = [1, 1, 0, 0, 1]
#parent2 = [0, 0, 1, 1, 0]
#child1 = [1, 1, 1, 1, 0]
#child2 = [0, 0, 0, 0, 1]

import random


def one_point_crossover(parent1, parent2):
    #Perform one-point crossover.
    #Args:
    #    parent1 (list[int]): First parent.
    #    parent2 (list[int]): Second parent.

    #Returns:
    #    tuple[list[int], list[int]]: Two offspring.

    n = len(parent1)
    if n < 2:
        return parent1[:], parent2[:]

    point = random.randint(1, n - 1)

    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]

    return child1, child2

def two_point_crossover(parent1, parent2):
    #Perform two-point crossover.
    #Two crossover points are selected.
    #The middle segment between the two points is swapped between the parents.

    #Example:
    #    parent1 = [1, 1, 0, 0, 1]
    #    parent2 = [0, 0, 1, 1, 0]

    #    If point1 = 1 and point2 = 4:
    #    child1 = [1, 0, 1, 1, 1]
    #    child2 = [0, 1, 0, 0, 0]

    #Args:
    #    parent1 (list[int]): First parent.
    #    parent2 (list[int]): Second parent.

    #Returns:
    #    tuple[list[int], list[int]]: Two offspring.

    n = len(parent1)

    if n < 3:
        return one_point_crossover(parent1, parent2)

    point1 = random.randint(1, n - 2)
    point2 = random.randint(point1 + 1, n - 1)

    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]

    return child1, child2

def uniform_crossover(parent1, parent2, swap_probability=0.5):

    #Perform uniform crossover.
    #For each gene, the child randomly receives the gene from parent1 or parent2.

    #Args:
    #    parent1 (list[int]): First parent.
    #    parent2 (list[int]): Second parent.
    #    swap_probability (float): Probability of taking the gene from the other parent.

    #Returns:
    #    tuple[list[int], list[int]]: Two offspring.

    child1 = []
    child2 = []

    for gene1, gene2 in zip(parent1, parent2):
        if random.random() < swap_probability:
            child1.append(gene2)
            child2.append(gene1)
        else:
            child1.append(gene1)
            child2.append(gene2)

    return child1, child2