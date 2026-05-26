import csv
import os
import statistics
import time

from parser import parse_scp_file
from ga import run_ga
from fitness import is_feasible


RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


INSTANCES = [
    "data/scp41.txt",
    "data/scp42.txt",
    "data/scp43.txt",
    "data/scp44.txt",
    "data/scp51.txt",
    "data/scp52.txt",
    "data/scp53.txt",
    "data/scpa1.txt",
    "data/scpa2.txt",
    "data/scpa3.txt",
]

LECTURER_TARGETS = {
    "scp41": (429, 433),
    "scp42": (512, 517),
    "scp43": (516, 521),
    "scp44": (494, 499),
    "scp51": (253, 255),
    "scp52": (302, 305),
    "scp53": (226, 228),
    "scpa1": (253, 255),
    "scpa2": (252, 254),
    "scpa3": (232, 234),
}

# Use the same seeds as HW1 if possible.
# Start with one seed if runtime is high.
SEEDS = [42]


BASE_GA_PARAMS = {
    "population_size": 300,
    "generations": 500,
    "crossover_rate": 0.91,
    "mutation_rate": 0.015,
    "elite_size": 10,
    "crossover_type": "uniform",
    "show_plots": False,
    "verbose": False,
    "patience": 80,
    "min_generations": 100,
}


FINAL_HW2_CONFIG = {
    "label": "HW2 novelty fitness alpha=0.7",
    "description": "Final selected HW2 configuration: novelty-based g(x,t), alpha=0.7",
    "adaptive_fitness_mode": "novelty",
    "adaptive_fitness_alpha": 0.7,
}


def short_instance_name(path):
    return os.path.basename(path).replace(".txt", "")


def avg_from_history(history, key):
    values = [row.get(key, 0.0) for row in history]
    values = [v for v in values if isinstance(v, (int, float))]
    return sum(values) / len(values) if values else 0.0


def target_gap(instance, cost):
    low, high = LECTURER_TARGETS[instance]

    if low <= cost <= high:
        return 0

    if cost > high:
        return cost - high

    return low - cost


def inside_target(instance, cost):
    low, high = LECTURER_TARGETS[instance]
    return low <= cost <= high


def run_single(instance_path, problem, seed):
    instance_name = short_instance_name(instance_path)

    result = run_ga(
        problem=problem,
        seed=seed,

        population_size=BASE_GA_PARAMS["population_size"],
        generations=BASE_GA_PARAMS["generations"],
        crossover_rate=BASE_GA_PARAMS["crossover_rate"],
        mutation_rate=BASE_GA_PARAMS["mutation_rate"],
        elite_size=BASE_GA_PARAMS["elite_size"],
        crossover_type=BASE_GA_PARAMS["crossover_type"],
        show_plots=BASE_GA_PARAMS["show_plots"],
        verbose=BASE_GA_PARAMS["verbose"],
        patience=BASE_GA_PARAMS["patience"],
        min_generations=BASE_GA_PARAMS["min_generations"],

        # Final HW2 config:
        # We do not use niching/speciation here because they were expensive
        # and did not consistently improve cost.
        niching_method="none",

        # Keep mutation fixed in the final config.
        mutation_control_mode="fixed",

        # Individual adaptive mutation is disabled.
        individual_mutation_mode="none",

        # Main HW2 improvement: novelty-based g(x,t).
        adaptive_fitness_mode=FINAL_HW2_CONFIG["adaptive_fitness_mode"],
        adaptive_fitness_alpha=FINAL_HW2_CONFIG["adaptive_fitness_alpha"],
        adaptive_fitness_distance="jaccard",
        adaptive_fitness_sample_size=30,
        adaptive_fitness_age_threshold=10,

        enable_diversity_injection=False,
    )

    history = result["history"]
    last = history[-1]
    cost = result["best_cost"]

    return {
        "instance": instance_name,
        "seed": seed,
        "config": FINAL_HW2_CONFIG["label"],

        "target_low": LECTURER_TARGETS[instance_name][0],
        "target_high": LECTURER_TARGETS[instance_name][1],
        "inside_target": inside_target(instance_name, cost),
        "target_gap": target_gap(instance_name, cost),

        "best_cost": cost,
        "feasible": is_feasible(result["best_solution"], problem),
        "runtime": result["total_time"],
        "generations_used": len(history),

        "final_jaccard_diversity": last.get("diversity_jaccard", 0.0),
        "avg_jaccard_diversity": avg_from_history(history, "diversity_jaccard"),
        "final_unique_ratio": last.get("diversity_unique_ratio", 0.0),

        "avg_adaptive_g_score": avg_from_history(history, "adaptive_g_avg"),
        "avg_adaptive_score": avg_from_history(history, "adaptive_score_avg"),
    }


def mean(values):
    return statistics.mean(values) if values else 0.0


def stdev(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def make_summary(rows):
    groups = {}

    for row in rows:
        groups.setdefault(row["instance"], []).append(row)

    summary_rows = []

    for instance, group_rows in groups.items():
        costs = [r["best_cost"] for r in group_rows]
        gaps = [r["target_gap"] for r in group_rows]
        runtimes = [r["runtime"] for r in group_rows]
        diversities = [r["final_jaccard_diversity"] for r in group_rows]
        generations = [r["generations_used"] for r in group_rows]

        summary_rows.append({
            "instance": instance,
            "target_low": LECTURER_TARGETS[instance][0],
            "target_high": LECTURER_TARGETS[instance][1],

            "avg_best_cost": mean(costs),
            "std_best_cost": stdev(costs),
            "best_cost_over_seeds": min(costs),
            "worst_cost_over_seeds": max(costs),

            "avg_target_gap": mean(gaps),
            "best_target_gap": min(gaps),
            "inside_target_any_seed": any(r["inside_target"] for r in group_rows),
            "inside_target_all_seeds": all(r["inside_target"] for r in group_rows),

            "avg_runtime": mean(runtimes),
            "avg_generations_used": mean(generations),
            "avg_final_jaccard_diversity": mean(diversities),

            "all_runs_feasible": all(r["feasible"] for r in group_rows),
            "num_seeds": len(group_rows),
        })

    return summary_rows


def write_csv(path, rows):
    if not rows:
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary_rows):
    print("\n\n================ FINAL HW2 CONFIG ON ALL 10 PROBLEMS ================")
    print(
        "instance | target | avg_cost | best | worst | gap | inside | "
        "div | gens | time | feasible | seeds"
    )
    print("-" * 170)

    total_gap = 0
    inside_count = 0

    for row in summary_rows:
        total_gap += row["avg_target_gap"]
        if row["inside_target_any_seed"]:
            inside_count += 1

        target_text = f"{row['target_low']}-{row['target_high']}"

        print(
            f"{row['instance']:6s} | "
            f"{target_text:9s} | "
            f"avg_cost={row['avg_best_cost']:.2f} | "
            f"best={row['best_cost_over_seeds']} | "
            f"worst={row['worst_cost_over_seeds']} | "
            f"gap={row['avg_target_gap']:.2f} | "
            f"inside={row['inside_target_any_seed']} | "
            f"div={row['avg_final_jaccard_diversity']:.4f} | "
            f"gens={row['avg_generations_used']:.1f} | "
            f"time={row['avg_runtime']:.2f}s | "
            f"feasible={row['all_runs_feasible']} | "
            f"seeds={row['num_seeds']}"
        )

    print("\nOverall:")
    print("Problems inside target:", inside_count, "/", len(summary_rows))
    print("Total target gap:", round(total_gap, 2))
    print("Average target gap:", round(total_gap / len(summary_rows), 2))


def main():
    all_rows = []
    start = time.time()

    print("Final HW2 configuration:")
    print(FINAL_HW2_CONFIG["description"])
    print("Seeds:", SEEDS)

    for instance_path in INSTANCES:
        print("\n====================================================")
        print("Instance:", instance_path)

        problem = parse_scp_file(instance_path)
        instance_name = short_instance_name(instance_path)
        target = LECTURER_TARGETS[instance_name]

        print(f"Problem size: m={problem.m}, n={problem.n}")
        print(f"Lecturer target: {target[0]}-{target[1]}")

        for seed in SEEDS:
            print(f"Running seed {seed}...", flush=True)

            row = run_single(instance_path, problem, seed)
            all_rows.append(row)

            print(
                f"seed={seed} | "
                f"cost={row['best_cost']} | "
                f"target={row['target_low']}-{row['target_high']} | "
                f"gap={row['target_gap']} | "
                f"inside={row['inside_target']} | "
                f"feasible={row['feasible']} | "
                f"time={row['runtime']:.2f}s | "
                f"div={row['final_jaccard_diversity']:.4f} | "
                f"gens={row['generations_used']}"
            )

    summary_rows = make_summary(all_rows)

    detailed_path = os.path.join(RESULTS_DIR, "final_hw2_all10_detailed.csv")
    summary_path = os.path.join(RESULTS_DIR, "final_hw2_all10_summary.csv")

    write_csv(detailed_path, all_rows)
    write_csv(summary_path, summary_rows)

    print_summary(summary_rows)

    print("\nSaved detailed results to:", detailed_path)
    print("Saved summary results to:", summary_path)
    print("Total runtime:", round(time.time() - start, 2), "seconds")


if __name__ == "__main__":
    main()