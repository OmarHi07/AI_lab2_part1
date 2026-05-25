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

SEEDS = [42]


BASE_GA_PARAMS = {
    "population_size": 300,
    "generations": 200,
    "crossover_rate": 0.91,
    "mutation_rate": 0.015,
    "elite_size": 10,
    "crossover_type": "uniform",
    "show_plots": False,
    "verbose": False,
    "patience": 80,
    "min_generations": 100,
}


EXPERIMENT_CONFIGS = [
    {
        "variant": "No individual adaptive mutation",
        "individual_mutation_mode": "none",
    },
    {
        "variant": "Relative-fitness adaptive mutation p=0.015-0.06",
        "individual_mutation_mode": "relative_fitness",
    },
    {
        "variant": "Age-based adaptive mutation p=0.015-0.06 ageT=10",
        "individual_mutation_mode": "age_based",
    },
]


def short_instance_name(path):
    return os.path.basename(path).replace(".txt", "")


def avg_from_history(history, key):
    values = [row.get(key, 0.0) for row in history]
    values = [v for v in values if isinstance(v, (int, float))]
    return sum(values) / len(values) if values else 0.0


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

        niching_method="none",
        mutation_control_mode="fixed",

        individual_mutation_mode=config["individual_mutation_mode"],
        individual_mutation_min_rate=0.015,
        individual_mutation_max_rate=0.06,
        age_mutation_threshold=10,

        enable_diversity_injection=False,
    )

    history = result["history"]
    last = history[-1]

    return {
        "instance": short_instance_name(instance_path),
        "seed": seed,
        "variant": config["variant"],
        "individual_mutation_mode": config["individual_mutation_mode"],

        "best_cost": result["best_cost"],
        "feasible": is_feasible(result["best_solution"], problem),
        "runtime": result["total_time"],
        "generations_used": len(history),

        "final_jaccard_diversity": last.get("diversity_jaccard", 0.0),
        "avg_jaccard_diversity": avg_from_history(history, "diversity_jaccard"),
        "final_unique_ratio": last.get("diversity_unique_ratio", 0.0),

        "avg_individual_mutation_rate": avg_from_history(history, "avg_individual_mutation_rate"),
        "max_individual_mutation_rate": max(row.get("max_individual_mutation_rate", 0.0) for row in history),
        "avg_population_age": avg_from_history(history, "avg_population_age"),
        "max_population_age": max(row.get("max_population_age", 0) for row in history),
    }


def mean(values):
    return statistics.mean(values) if values else 0.0


def stdev(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def make_summary(rows):
    groups = {}

    for row in rows:
        key = (row["instance"], row["variant"], row["individual_mutation_mode"])
        groups.setdefault(key, []).append(row)

    summary_rows = []

    for key, group_rows in groups.items():
        instance, variant, mode = key

        costs = [r["best_cost"] for r in group_rows]
        runtimes = [r["runtime"] for r in group_rows]
        diversities = [r["final_jaccard_diversity"] for r in group_rows]
        avg_mut_rates = [r["avg_individual_mutation_rate"] for r in group_rows]
        max_mut_rates = [r["max_individual_mutation_rate"] for r in group_rows]
        ages = [r["avg_population_age"] for r in group_rows]
        max_ages = [r["max_population_age"] for r in group_rows]

        summary_rows.append({
            "instance": instance,
            "variant": variant,
            "mode": mode,

            "avg_best_cost": mean(costs),
            "std_best_cost": stdev(costs),
            "best_cost_over_seeds": min(costs),
            "worst_cost_over_seeds": max(costs),

            "avg_runtime": mean(runtimes),
            "avg_final_jaccard_diversity": mean(diversities),

            "avg_individual_mutation_rate": mean(avg_mut_rates),
            "max_individual_mutation_rate": max(max_mut_rates),

            "avg_population_age": mean(ages),
            "max_population_age": max(max_ages),

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
    print("\n\n================ INDIVIDUAL MUTATION SUMMARY ================")
    print(
        "instance | method | avg_cost | std | best | worst | div | "
        "avg_ind_mut | max_ind_mut | avg_age | max_age | time | feasible | seeds"
    )
    print("-" * 180)

    for row in summary_rows:
        print(
            f"{row['instance']:6s} | "
            f"{row['variant']:36s} | "
            f"avg_cost={row['avg_best_cost']:.2f} | "
            f"std={row['std_best_cost']:.2f} | "
            f"best={row['best_cost_over_seeds']} | "
            f"worst={row['worst_cost_over_seeds']} | "
            f"div={row['avg_final_jaccard_diversity']:.4f} | "
            f"avg_ind_mut={row['avg_individual_mutation_rate']:.4f} | "
            f"max_ind_mut={row['max_individual_mutation_rate']:.4f} | "
            f"avg_age={row['avg_population_age']:.2f} | "
            f"max_age={row['max_population_age']} | "
            f"time={row['avg_runtime']:.2f}s | "
            f"feasible={row['all_runs_feasible']} | "
            f"seeds={row['num_seeds']}"
        )


def print_best_methods(summary_rows):
    print("\n\n================ BEST INDIVIDUAL MUTATION METHOD PER INSTANCE ================")

    instances = sorted(set(row["instance"] for row in summary_rows))

    for instance in instances:
        rows = [row for row in summary_rows if row["instance"] == instance]

        best_cost_row = min(rows, key=lambda r: r["avg_best_cost"])
        best_diversity_row = max(rows, key=lambda r: r["avg_final_jaccard_diversity"])
        fastest_row = min(rows, key=lambda r: r["avg_runtime"])

        print(f"\nInstance: {instance}")
        print(
            "Best cost:",
            best_cost_row["variant"],
            "| cost =",
            round(best_cost_row["avg_best_cost"], 2),
            "| diversity =",
            round(best_cost_row["avg_final_jaccard_diversity"], 4),
            "| time =",
            round(best_cost_row["avg_runtime"], 2),
            "s"
        )

        print(
            "Highest diversity:",
            best_diversity_row["variant"],
            "| diversity =",
            round(best_diversity_row["avg_final_jaccard_diversity"], 4),
            "| cost =",
            round(best_diversity_row["avg_best_cost"], 2)
        )

        print(
            "Fastest:",
            fastest_row["variant"],
            "| time =",
            round(fastest_row["avg_runtime"], 2),
            "s",
            "| cost =",
            round(fastest_row["avg_best_cost"], 2)
        )


def main():
    all_rows = []

    start = time.time()

    for instance_path in INSTANCES:
        print("\n====================================================")
        print("Instance:", instance_path)

        problem = parse_scp_file(instance_path)
        print(f"Problem size: m={problem.m}, n={problem.n}")

        for config in EXPERIMENT_CONFIGS:
            print("\n---", config["variant"], "---")

            for seed in SEEDS:
                print(f"Running seed {seed}...", flush=True)

                row = run_single(
                    instance_path=instance_path,
                    problem=problem,
                    seed=seed,
                    config=config,
                )

                all_rows.append(row)

                print(
                    f"seed={seed} | "
                    f"cost={row['best_cost']} | "
                    f"feasible={row['feasible']} | "
                    f"time={row['runtime']:.2f}s | "
                    f"div={row['final_jaccard_diversity']:.4f} | "
                    f"avg_ind_mut={row['avg_individual_mutation_rate']:.4f} | "
                    f"max_ind_mut={row['max_individual_mutation_rate']:.4f} | "
                    f"avg_age={row['avg_population_age']:.2f} | "
                    f"max_age={row['max_population_age']}"
                )

    summary_rows = make_summary(all_rows)

    detailed_path = os.path.join(RESULTS_DIR, "individual_mutation_second_test_detailed.csv")
    summary_path = os.path.join(RESULTS_DIR, "individual_mutation_second_test_summary.csv")

    write_csv(detailed_path, all_rows)
    write_csv(summary_path, summary_rows)

    print_summary(summary_rows)
    print_best_methods(summary_rows)

    print("\nSaved detailed results to:", detailed_path)
    print("Saved summary results to:", summary_path)
    print("Total runtime:", round(time.time() - start, 2), "seconds")


if __name__ == "__main__":
    main()