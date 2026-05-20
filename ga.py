# Main implementation of the Genetic Algorithm for Set Cover.

import random
import time
import matplotlib.pyplot as plt
import numpy as np

from chromosome import random_chromosome
from crossover import one_point_crossover, two_point_crossover, uniform_crossover
from mutation import bit_flip_mutation
from repair import repair_solution
from selection import tournament_selection
from fitness import compute_cost
from stats import average, std_dev, active_genes_std, sampled_hamming_distance, unique_chromosome_ratio
from distance import sampled_average_distance
from baseline import greedy_set_cover
from niching import fitness_sharing_selection_scores

def greedy_variant(problem, greedy_solution, remove_rate=0.20, add_rate=0.01):
    """
    Create a variation of the greedy solution:
    - remove some selected sets
    - add a few random sets
    - repair the result
    """
    candidate = greedy_solution[:]

    selected_sets = [s for s in range(problem.n) if candidate[s] == 1]

    for s in selected_sets:
        if random.random() < remove_rate:
            candidate[s] = 0

    zero_sets = [s for s in range(problem.n) if candidate[s] == 0]
    add_count = max(1, int(problem.n * add_rate))
    add_count = min(add_count, len(zero_sets))

    for s in random.sample(zero_sets, add_count):
        candidate[s] = 1

    candidate = repair_solution(candidate, problem)
    return candidate

def initialize_population(problem, population_size, activation_prob=0.05, use_greedy_seed=True):
    population = []

    greedy_solution = None

    if use_greedy_seed:
        greedy_solution, _ = greedy_set_cover(problem)
        greedy_solution = repair_solution(greedy_solution, problem)
        population.append(greedy_solution)

        # Add heuristic variations, similar to heuristic initialization
        heuristic_count = int(population_size * 0.20)

        while len(population) < heuristic_count:
            variant = greedy_variant(
                problem,
                greedy_solution,
                remove_rate=0.20,
                add_rate=0.01
            )
            population.append(variant)

    # Fill the rest randomly
    while len(population) < population_size:
        chromosome = random_chromosome(problem.n, activation_prob=activation_prob)
        chromosome = repair_solution(chromosome, problem)
        population.append(chromosome)

    return population

def evaluate_population(population, problem):
    """
    Faster evaluation.
    Because every chromosome is repaired before entering the population,
    all individuals are feasible, so fitness = -cost.
    """
    costs = [compute_cost(individual, problem) for individual in population]
    fitnesses = [-cost for cost in costs]
    return fitnesses, costs

def inject_diversity(population, fitnesses, problem, replace_ratio=0.2):
    replace_count = int(len(population) * replace_ratio)

    # Sort from best to worst
    sorted_indices = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)

    # Keep the best individuals
    keep_count = len(population) - replace_count
    new_population = [population[i][:] for i in sorted_indices[:keep_count]]

    # Replace the worst individuals with new random repaired chromosomes
    while len(new_population) < len(population):
        chromosome = random_chromosome(problem.n, activation_prob=0.05)
        chromosome = repair_solution(chromosome, problem)
        new_population.append(chromosome)

    return new_population

def improve_solution(solution, problem, max_passes=3):
    """
    Try to improve the final solution by removing selected sets one by one,
    repairing the solution, and keeping the change only if the cost improves.
    """
    best = repair_solution(solution, problem)
    best_cost = compute_cost(best, problem)

    for _ in range(max_passes):
        improved = False

        selected_sets = [s for s in range(problem.n) if best[s] == 1]

        # Try removing expensive sets first
        selected_sets.sort(key=lambda s: problem.costs[s], reverse=True)

        for s in selected_sets:
            candidate = best[:]
            candidate[s] = 0

            candidate = repair_solution(candidate, problem)
            candidate_cost = compute_cost(candidate, problem)

            if candidate_cost < best_cost:
                best = candidate
                best_cost = candidate_cost
                improved = True
                break

        if not improved:
            break

    return best, best_cost

def get_best_individual(population, fitnesses):
    """Return the best individual and its fitness."""
    best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
    return population[best_idx][:], fitnesses[best_idx]


def plot_ga_graphs(history):
    generations = [h["generation"] for h in history]
    best = [h["best_fitness"] for h in history]
    avg = [h["avg_fitness"] for h in history]
    worst = [h["worst_fitness"] for h in history]
    diversity_active = [h["diversity_active_genes"] for h in history]

    plt.figure()
    plt.plot(generations, best, label="Best Fitness")
    plt.plot(generations, avg, label="Average Fitness")
    plt.plot(generations, worst, label="Worst Fitness")
    plt.title("Best / Average / Worst Fitness Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(generations, diversity_active, label="Active Genes Diversity")
    plt.title("Population Diversity Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Std of Active Genes")
    plt.legend()
    plt.grid()
    plt.show()


def run_ga(problem, population_size=100, generations=100,
           crossover_rate=0.8, mutation_rate=0.05,
           elite_size=1, crossover_type="one_point", seed=None,
           show_plots=False, verbose=False, patience=80,
           min_generations=100,
           niching_method="none",
           niche_distance="jaccard",
           sigma_share=0.35,
           sharing_alpha=1.0,
           sharing_sample_size=30):
    """Run the Genetic Algorithm on a Set Cover instance."""

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    start_time = time.time()
    population = initialize_population(
        problem,
        population_size,
        activation_prob=0.05,
        use_greedy_seed=True
    )
    history = []

    best_cost_so_far = float("inf")
    best_solution_so_far = None
    best_fitness_so_far = None
    last_improvement_generation = 0

    for generation in range(generations):
        fitnesses, costs = evaluate_population(population, problem)

        # Real fitness is still used for reporting true solution quality.
        # Selection fitness may be changed by niching / fitness sharing.
        selection_fitnesses = fitnesses[:]

        sharing_avg_niche_count = 0.0
        sharing_max_niche_count = 0.0
        sharing_min_niche_count = 0.0

        if niching_method == "fitness_sharing":
            selection_fitnesses, sharing_info = fitness_sharing_selection_scores(
                population=population,
                costs=costs,
                distance_name=niche_distance,
                sigma_share=sigma_share,
                alpha=sharing_alpha,
                sample_size=sharing_sample_size,
                sample_seed=generation,
            )

            sharing_avg_niche_count = sharing_info["avg_niche_count"]
            sharing_max_niche_count = sharing_info["max_niche_count"]
            sharing_min_niche_count = sharing_info["min_niche_count"]

        elif niching_method != "none":
            raise ValueError("Unknown niching method: " + str(niching_method))

        best_individual, best_fit = get_best_individual(population, fitnesses)

        best_cost = -best_fit
        avg_cost = average(costs)

        if best_cost < best_cost_so_far:
            best_cost_so_far = best_cost
            best_solution_so_far = best_individual[:]
            best_fitness_so_far = best_fit
            last_improvement_generation = generation

        # Cost-based selection pressure: avg cost / best cost.
        # Higher value means the best individual is much better than the average.
        selection_pressure = avg_cost / best_cost if best_cost > 0 else 0.0

        gen_stats = {
            "generation": generation,
            "best_fitness": max(fitnesses),
            "avg_fitness": average(fitnesses),
            "worst_fitness": min(fitnesses),
            "std_fitness": std_dev(fitnesses),
            "best_cost": best_cost,
            "avg_cost": avg_cost,
            "worst_cost": max(costs),
            "selection_pressure": selection_pressure,
            "diversity_hamming": sampled_hamming_distance(population, sample_pairs=30, sample_seed=generation),
            "diversity_hamming_normalized": sampled_average_distance(
                population,
                distance_name="hamming",
                sample_pairs=30,
                sample_seed=generation
            ),
            "diversity_jaccard": sampled_average_distance(
                population,
                distance_name="jaccard",
                sample_pairs=30,
                sample_seed=generation
            ),
            "diversity_active_genes": active_genes_std(population),
            "diversity_unique_ratio": unique_chromosome_ratio(population),

            "niching_method": niching_method,
            "niche_distance": niche_distance,
            "sigma_share": sigma_share if niching_method == "fitness_sharing" else 0.0,
            "sharing_alpha": sharing_alpha if niching_method == "fitness_sharing" else 0.0,
            "sharing_avg_niche_count": sharing_avg_niche_count,
            "sharing_max_niche_count": sharing_max_niche_count,
            "sharing_min_niche_count": sharing_min_niche_count,

            "elapsed_time": time.time() - start_time,
        }
        history.append(gen_stats)

        if verbose and (generation % 25 == 0 or generation == generations - 1):
            print(
                f"  generation {generation + 1}/{generations} | "
                f"best_cost={best_cost} | time={time.time() - start_time:.2f}s",
                flush=True
            )

        # Early stopping
        if generation >= min_generations:
            if generation - last_improvement_generation >= patience:
                print(f"Early stopping at generation {generation + 1}, best_cost={best_cost_so_far}")
                break

        # Elitism should preserve the true best objective solutions.
        # Fitness sharing is used for parent selection, not for deleting the real best.
        sorted_indices = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
        new_population = [population[i][:] for i in sorted_indices[:elite_size]]

        while len(new_population) < population_size:
            parent1 = tournament_selection(population, selection_fitnesses)
            parent2 = tournament_selection(population, selection_fitnesses)

            if random.random() < crossover_rate:
                if crossover_type == "one_point":
                    child1, child2 = one_point_crossover(parent1, parent2)
                elif crossover_type == "two_point":
                    child1, child2 = two_point_crossover(parent1, parent2)
                elif crossover_type == "uniform":
                    child1, child2 = uniform_crossover(parent1, parent2)
                else:
                    raise ValueError("Unknown crossover type: " + crossover_type)
            else:
                child1, child2 = parent1[:], parent2[:]

            child1 = bit_flip_mutation(child1, mutation_rate)
            child2 = bit_flip_mutation(child2, mutation_rate)

            child1 = repair_solution(child1, problem)
            child2 = repair_solution(child2, problem)

            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        # If stuck, inject diversity into the new population
        stagnation = generation - last_improvement_generation

        if generation >= min_generations and stagnation > 0 and stagnation % 50 == 0:
            new_fitnesses, _ = evaluate_population(new_population, problem)
            population = inject_diversity(new_population, new_fitnesses, problem, replace_ratio=0.2)
        else:
            population = new_population

    if show_plots:
        plot_ga_graphs(history)

    best_solution_so_far, best_cost_so_far = improve_solution(
        best_solution_so_far,
        problem,
        max_passes=3
    )

    best_fitness_so_far = -best_cost_so_far

    return {
        "best_solution": best_solution_so_far,
        "best_fitness": best_fitness_so_far,
        "best_cost": best_cost_so_far,
        "history": history,
        "total_time": time.time() - start_time,
    }
