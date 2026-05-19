
#This module contains functions for:
#- computing total cost
#- checking coverage
#- checking feasibility
#- computing fitness value


def compute_cost(chromosome, problem):
    #Compute the total cost of the selected sets.
    #Args:
    #    chromosome (list[int]): Binary chromosome.
    #    problem (SetCoverProblem): Problem instance.
    #Returns:
    #    float: Total cost.
    return sum(problem.costs[i] for i in range(problem.n) if chromosome[i] == 1)


def get_covered_elements(chromosome, problem):

    #Compute the set of covered elements.
    #Args:
    #    chromosome (list[int]): Binary chromosome.
    #    problem (SetCoverProblem): Problem instance.
    #Returns:
    #    set[int]: Covered elements.

    covered = set()
    for s in range(problem.n):
        if chromosome[s] == 1:
            covered.update(problem.set_to_elements[s])
    return covered


def count_uncovered(chromosome, problem):
    #Count how many elements are still uncovered.
    #Args:
    #    chromosome (list[int]): Binary chromosome.
    #    problem (SetCoverProblem): Problem instance.

    #Returns:
    #    int: Number of uncovered elements.

    covered = get_covered_elements(chromosome, problem)
    return problem.m - len(covered)


def is_feasible(chromosome, problem):
    #Check whether the chromosome covers all elements.

    #Args:
    #    chromosome (list[int]): Binary chromosome.
    #    problem (SetCoverProblem): Problem instance.

    #Returns:
    #    bool: True if feasible, False otherwise.

    return count_uncovered(chromosome, problem) == 0


def fitness(chromosome, problem):
    """
    Compute a fitness score.

    Higher fitness is better.

    Strategy:
    - Feasible solutions are always preferred over infeasible ones.
    - Among feasible solutions, lower cost is better.
    - Infeasible solutions receive a very large penalty.
    """
    total_cost = compute_cost(chromosome, problem)

    if is_feasible(chromosome, problem):
        return -total_cost

    uncovered = count_uncovered(chromosome, problem)
    return -(1000000 + 1000 * uncovered + total_cost)