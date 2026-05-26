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
    "data/scp51.txt",
    "data/scpa3.txt",
]

# One seed is enough for the final check if runtime is high.
# If you have time later, change to [42, 123].
SEEDS = [42]


LECTURER_TARGETS = {
    "scp41": (429, 433),
    "scp51": (253, 255),
    "scpa3": (232, 234),
}


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


CONFIGS = [
    {
        "label": "HW2 novelty fitness alpha=0.6",
        "description": "Novelty-based g(x,t), alpha=0.6",
        "mutation_control_mode": "fixed",
        "nonlinear_min_rate": 0.005,
        "nonlinear_max_rate": 0.05,
        "nonlinear_power": 2.0,
        "individual_mutation_mode": "none",
        "adaptive_fitness_mode": "novelty",
        "adaptive_fitness_alpha": 0.6,
    },
    {
        "label": "HW2 novelty fitness alpha=0.7",
        "description": "Novelty-based g(x,t), alpha=0.7",
        "mutation_control_mode": "fixed",
        "nonlinear_min_rate": 0.005,
        "nonlinear_max_rate": 0.05,
        "nonlinear_power": 2.0,
        "individual_mutation_mode": "none",
        "adaptive_fitness_mode": "novelty",
        "adaptive_fitness_alpha": 0.7,
    },
    {
        "label": "HW2 fast nonlinear mutation",
        "description": "Nonlinear mutation with faster decay for 500 generations",
        "mutation_control_mode": "nonlinear",
        "nonlinear_min_rate": 0.005,
        "nonlinear_max_rate": 0.03,
        "nonlinear_power": 4.0,
        "individual_mutation_mode": "none",
        "adaptive_fitness_mode": "none",
        "adaptive_fitness_alpha": 0.8,
    },
    {
        "label": "HW2 fast nonlinear + novelty alpha=0.7",
        "description": "Fast nonlinear mutation + novelty-based g(x,t), alpha=0.7",
        "mutation_control_mode": "nonlinear",
        "nonlinear_min_rate": 0.005,
        "nonlinear_max_rate": 0.03,
        "nonlinear_power": 4.0,
        "individual_mutation_mode": "none",
        "adaptive_fitness_mode": "novelty",
        "adaptive_fitness_alpha": 0.7,
    },
    {
        "label": "HW2 age-based individual mutation",
        "description": "Age-based individual mutation, p=0.015-0.06, ageT=10",
        "mutation_control_mode": "fixed",
        "nonlinear_min_rate": 0.005,
        "nonlinear_max_rate": 0.05,
        "nonlinear_power": 2.0,
        "individual_mutation_mode": "age_based",
        "adaptive_fitness_mode": "none",
        "adaptive_fitness_alpha": 0.8,
    },
]


def short_instance_name(path):
    return os.path.basename(path).replace(".txt", "")


def avg_from_history(history, key):
    values = [row.get(key, 0.0) for row in history]
    values = [v for v in values if isinstance(v, (int, float))]
    return sum(values) / len(values) if values else 0.0


def target_gap(instance, cost):
    """
    Returns 0 if cost is inside lecturer target range.
    Otherwise returns how far above the upper bound it is.
    """
    if instance not in LECTURER_TARGETS:
        return None

    low, high = LECTURER_TARGETS[instance]

    if low <= cost <= high:
        return 0

    if cost > high:
        return cost - high

    return low - cost


def run_single(instance_path, problem, seed, config):
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

        # We isolate the best HW2 mechanisms.
        # Niching/speciation are not used here because they were expensive
        # and did not consistently improve cost.
        niching_method="none",

        mutation_control_mode=config["mutation_control_mode"],
        nonlinear_min_rate=config["nonlinear_min_rate"],
        nonlinear_max_rate=config["nonlinear_max_rate"],
        nonlinear_power=config["nonlinear_power"],

        # Individual adaptive mutation is disabled here because
        # nonlinear mutation and novelty fitness were stronger overall.
        individual_mutation_mode=config["individual_mutation_mode"],
        individual_mutation_min_rate=0.015,
        individual_mutation_max_rate=0.06,
        age_mutation_threshold=10,

        # Novelty-based g(x,t) candidate.
        adaptive_fitness_mode=config["adaptive_fitness_mode"],
        adaptive_fitness_alpha=config["adaptive_fitness_alpha"],
        adaptive_fitness_distance="jaccard",
        adaptive_fitness_sample_size=30,
        adaptive_fitness_age_threshold=10,

        enable_diversity_injection=False,
    )

    history = result["history"]
    last = history[-1]
    instance_name = short_instance_name(instance_path)

    return {
        "instance": instance_name,
        "seed": seed,
        "config": config["label"],
        "description": config["description"],

        "best_cost": result["best_cost"],
        "target_gap": target_gap(instance_name, result["best_cost"]),
        "feasible": is_feasible(result["best_solution"], problem),
        "runtime": result["total_time"],
        "generations_used": len(history),

        "final_jaccard_diversity": last.get("diversity_jaccard", 0.0),
        "avg_jaccard_diversity": avg_from_history(history, "diversity_jaccard"),
        "final_unique_ratio": last.get("diversity_unique_ratio", 0.0),

        "avg_mutation_rate": avg_from_history(history, "current_mutation_rate"),
        "max_mutation_rate": max(row.get("current_mutation_rate", 0.0) for row in history),

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
        key = (row["instance"], row["config"])
        groups.setdefault(key, []).append(row)

    summary_rows = []

    for key, group_rows in groups.items():
        instance, config = key

        costs = [r["best_cost"] for r in group_rows]
        gaps = [r["target_gap"] for r in group_rows if r["target_gap"] is not None]
        runtimes = [r["runtime"] for r in group_rows]
        diversities = [r["final_jaccard_diversity"] for r in group_rows]
        generations = [r["generations_used"] for r in group_rows]
        avg_mut_rates = [r["avg_mutation_rate"] for r in group_rows]
        adaptive_g_scores = [r["avg_adaptive_g_score"] for r in group_rows]

        summary_rows.append({
            "instance": instance,
            "config": config,

            "avg_best_cost": mean(costs),
            "std_best_cost": stdev(costs),
            "best_cost_over_seeds": min(costs),
            "worst_cost_over_seeds": max(costs),
            "avg_target_gap": mean(gaps) if gaps else None,

            "avg_runtime": mean(runtimes),
            "avg_generations_used": mean(generations),
            "avg_final_jaccard_diversity": mean(diversities),

            "avg_mutation_rate": mean(avg_mut_rates),
            "avg_adaptive_g_score": mean(adaptive_g_scores),

            "all_runs_feasible": all(r["feasible"] for r in group_rows),
            "num_seeds": len(group_rows),
        })

    return summary_rows


def make_overall_summary(summary_rows):
    groups = {}

    for row in summary_rows:
        groups.setdefault(row["config"], []).append(row)

    overall_rows = []

    for config, rows in groups.items():
        costs = [r["avg_best_cost"] for r in rows]
        gaps = [r["avg_target_gap"] for r in rows if r["avg_target_gap"] is not None]
        runtimes = [r["avg_runtime"] for r in rows]
        diversities = [r["avg_final_jaccard_diversity"] for r in rows]

        overall_rows.append({
            "config": config,
            "avg_cost_across_instances": mean(costs),
            "total_target_gap": sum(gaps),
            "avg_target_gap": mean(gaps),
            "avg_runtime": mean(runtimes),
            "avg_diversity": mean(diversities),
        })

    return overall_rows


def write_csv(path, rows):
    if not rows:
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary_rows):
    print("\n\n================ HW2 BEST CONFIG CANDIDATES ================")
    print(
        "instance | config | avg_cost | best | worst | target_gap | "
        "div | gens | avg_mut | avg_g | time | feasible | seeds"
    )
    print("-" * 190)

    for row in summary_rows:
        gap = row["avg_target_gap"]
        gap_text = f"{gap:.2f}" if gap is not None else "NA"

        print(
            f"{row['instance']:6s} | "
            f"{row['config']:38s} | "
            f"avg_cost={row['avg_best_cost']:.2f} | "
            f"best={row['best_cost_over_seeds']} | "
            f"worst={row['worst_cost_over_seeds']} | "
            f"gap={gap_text} | "
            f"div={row['avg_final_jaccard_diversity']:.4f} | "
            f"gens={row['avg_generations_used']:.1f} | "
            f"avg_mut={row['avg_mutation_rate']:.4f} | "
            f"avg_g={row['avg_adaptive_g_score']:.4f} | "
            f"time={row['avg_runtime']:.2f}s | "
            f"feasible={row['all_runs_feasible']} | "
            f"seeds={row['num_seeds']}"
        )


def print_overall_summary(overall_rows):
    print("\n\n================ OVERALL HW2 CANDIDATE RANKING ================")
    print("config | avg_cost | total_gap | avg_gap | avg_div | avg_time")
    print("-" * 140)

    overall_rows = sorted(
        overall_rows,
        key=lambda r: (r["total_target_gap"], r["avg_cost_across_instances"], r["avg_runtime"])
    )

    for row in overall_rows:
        print(
            f"{row['config']:38s} | "
            f"avg_cost={row['avg_cost_across_instances']:.2f} | "
            f"total_gap={row['total_target_gap']:.2f} | "
            f"avg_gap={row['avg_target_gap']:.2f} | "
            f"avg_div={row['avg_diversity']:.4f} | "
            f"avg_time={row['avg_runtime']:.2f}s"
        )

    print("\nRecommended HW2 best config:")
    print(overall_rows[0]["config"])


def main():
    all_rows = []
    start = time.time()

    for instance_path in INSTANCES:
        print("\n====================================================")
        print("Instance:", instance_path)

        problem = parse_scp_file(instance_path)
        print(f"Problem size: m={problem.m}, n={problem.n}")

        for config in CONFIGS:
            print("\n---", config["label"], "---")
            print(config["description"])

            for seed in SEEDS:
                print(f"Running seed {seed}...", flush=True)

                row = run_single(instance_path, problem, seed, config)
                all_rows.append(row)

                print(
                    f"seed={seed} | "
                    f"cost={row['best_cost']} | "
                    f"gap={row['target_gap']} | "
                    f"feasible={row['feasible']} | "
                    f"time={row['runtime']:.2f}s | "
                    f"div={row['final_jaccard_diversity']:.4f} | "
                    f"gens={row['generations_used']}"
                )

    summary_rows = make_summary(all_rows)
    overall_rows = make_overall_summary(summary_rows)

    detailed_path = os.path.join(RESULTS_DIR, "hw2_best_candidates_detailed.csv")
    summary_path = os.path.join(RESULTS_DIR, "hw2_best_candidates_summary.csv")
    overall_path = os.path.join(RESULTS_DIR, "hw2_best_candidates_overall.csv")

    write_csv(detailed_path, all_rows)
    write_csv(summary_path, summary_rows)
    write_csv(overall_path, overall_rows)

    print_summary(summary_rows)
    print_overall_summary(overall_rows)

    print("\nSaved detailed results to:", detailed_path)
    print("Saved summary results to:", summary_path)
    print("Saved overall results to:", overall_path)
    print("Total runtime:", round(time.time() - start, 2), "seconds")


if __name__ == "__main__":
    main()