# repair.py
# Optimized repair for Set Cover.
# Strategy:
# 1. Add useful sets until all elements are covered.
# 2. Remove redundant selected sets without recomputing feasibility from scratch.


def repair_solution(chromosome, problem):
    """
    Repair a chromosome so it becomes feasible.

    Faster version:
    - Maintains cover_count[e] = how many selected sets cover element e.
    - Maintains uncovered set.
    - During removal, a selected set can be removed only if every element it covers
      is still covered by at least one other selected set.
    """
    chromosome = chromosome[:]

    cover_count = [0] * problem.m
    uncovered = set(range(problem.m))

    # Build coverage counts for currently selected sets.
    for s, gene in enumerate(chromosome):
        if gene == 1:
            for e in problem.set_to_elements[s]:
                if cover_count[e] == 0:
                    uncovered.discard(e)
                cover_count[e] += 1

    # Step 1: add sets until all elements are covered.
    while uncovered:
        best_set = None
        best_score = -1.0

        for s in range(problem.n):
            if chromosome[s] == 0:
                new_covers = 0
                for e in problem.set_to_elements[s]:
                    if e in uncovered:
                        new_covers += 1

                if new_covers > 0:
                    score = new_covers / problem.costs[s]
                    if score > best_score:
                        best_score = score
                        best_set = s

        if best_set is None:
            # Should not happen for valid SCP instances, but keeps the function safe.
            break

        chromosome[best_set] = 1
        for e in problem.set_to_elements[best_set]:
            if cover_count[e] == 0:
                uncovered.discard(e)
            cover_count[e] += 1

    # Step 2: remove redundant selected sets.
    selected_sets = [s for s, gene in enumerate(chromosome) if gene == 1]

    # Try removing expensive / low-coverage sets first.
    selected_sets.sort(
        key=lambda s: problem.costs[s] / max(1, len(problem.set_to_elements[s])),
        reverse=True
    )

    for s in selected_sets:
        if chromosome[s] == 0:
            continue

        # Safe to remove only if every covered element still has another covering set.
        can_remove = True
        for e in problem.set_to_elements[s]:
            if cover_count[e] <= 1:
                can_remove = False
                break

        if can_remove:
            chromosome[s] = 0
            for e in problem.set_to_elements[s]:
                cover_count[e] -= 1

    return chromosome
