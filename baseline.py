
#This module implements a simple greedy baseline for Set Cover.
#At each step, it selects the set with the best ratio:
#(number of newly covered elements) / cost
# Baseline = פתרון פשוט , שמשמש כ־קו השוואה
from fitness import compute_cost, is_feasible


def greedy_set_cover(problem):

    #Solve the problem using a greedy heuristic.
    #Args:
    #    problem (SetCoverProblem): Problem instance.
    #Returns:
    #    tuple[list[int], float]: Selected chromosome and its total cost.

    chromosome = [0] * problem.n
    uncovered = set(range(problem.m))

    while uncovered:
        best_set = None
        best_score = -1

        for s in range(problem.n):
            new_covers = len(uncovered.intersection(problem.set_to_elements[s]))
            if new_covers > 0:
                score = new_covers / problem.costs[s]
                if score > best_score:
                    best_score = score
                    best_set = s

        if best_set is None:
            break

        #להוסיף את הקבוצה
        chromosome[best_set] = 1
        uncovered -= set(problem.set_to_elements[best_set])

    total_cost = compute_cost(chromosome, problem)
    return chromosome, total_cost