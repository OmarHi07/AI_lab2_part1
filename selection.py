#This module implements parent selection methods for the genetic algorithm.
import random


def tournament_selection(population, fitnesses, tournament_size=3):
    #Select one parent using tournament selection.
    #Args:
    #    population (list[list[int]]): Current population.
    #    fitnesses (list[float]): Fitness of each individual.
    #    tournament_size (int): Number of individuals in each tournament.
    # Returns:
    #    list[int]: Selected parent chromosome.

    candidates = random.sample(range(len(population)), tournament_size)
    best_idx = max(candidates, key=lambda i: fitnesses[i])
    return population[best_idx][:]
