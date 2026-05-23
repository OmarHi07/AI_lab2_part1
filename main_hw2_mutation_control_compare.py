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

# Start with one seed. Later you can add 123 if runtime is okay.
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
        "variant": "Fixed mutation rate",
        "mutation_control_mode": "fixed",
        "mutation_rate": 0.015,
        "nonlinear_min_rate": 0.005,
        "nonlinear_max_rate": 0.05,
        "hypermutation_rate": 0.08,
        "hypermutation_trigger_after": 20,
    },
    {
        "variant": "Nonlinear decreasing mutation",
        "mutation_control_mode": "nonlinear",
        "mutation_rate": 0.015,
        "nonlinear_min_rate": 0.005,
        "nonlinear_max_rate": 0.05,
        "hypermutation_rate": 0.08,
        "hypermutation_trigger_after": 20,
    },
    {
        "variant": "Triggered hypermutation",
        "mutation_control_mode": "triggered_hypermutation",
        "mutation_rate": 0.015,
        "nonlinear_min_rate": 0.005,
        "nonlinear_max_rate": 0.05,
        "hypermutation_rate": 0.08,
        "hypermutation_trigger_after": 20,
    },
]


def short_instance_name(path):
    return os.path.basename(path).replace(".txt", "")


def avg_from_history(history, key):
    values = [row.get(key, 0.0) for row in history]
    values = [v for v in values if isinstance(v, (int, float))]
    return sum(values) / len(values) if values else 0.0


def count_true_from_history(history, key):
    return sum(1 for row in history if row.get(key, False) is True)


def run_single(instance_path, problem, seed, config):
    result = run_ga(
        problem=problem,
        seed=seed,

        population_size=BASE_GA_PARAMS["population_size"],
        generations=BASE_GA_PARAMS["generations"],
        crossover_rate=BASE_GA_PARAMS["crossover_rate"],
        mutation_rate=config["mutation_rate"],
        elite_size=BASE_GA_PARAMS["elite_size"],
        crossover_type=BASE_GA_PARAMS["crossover_type"],
        show_plots=BASE_GA_PARAMS["show_plots"],
        verbose=BASE_GA_PARAMS["verbose"],
        patience=BASE_GA_PARAMS["patience"],
        min_generations=BASE_GA_PARAMS["min_generations"],

        # No niching here, because this experiment isolates mutation control.
        niching_method="none",

        mutation_control_mode=config["mutation_control_mode"],
        nonlinear_min_rate=config["nonlinear_min_rate"],
        nonlinear_max_rate=config["nonlinear_max_rate"],
        nonlinear_power=2.0,
        hypermutation_rate=config["hypermutation_rate"],
        hypermutation_trigger_after=config["hypermutation_trigger_after"],

        # Disable old diversity injection so triggered hypermutation is the main anti-stagnation tool.
        enable_diversity_injection=False,
    )

    history = result["history"]
    last = history[-1]

    row = {
        "instance": short_instance_name(instance_path),
        "seed": seed,
        "variant": config["variant"],
        "mutation_control_mode": config["mutation_control_mode"],

        "best_cost": result["best_cost"],
        "feasible": is_feasible(result["best_solution"], problem),
        "runtime": result["total_time"],
        "generations_used": len(history),

        "final_jaccard_diversity": last.get("diversity_jaccard", 0.0),
        "avg_jaccard_diversity": avg_from_history(history, "diversity_jaccard"),
        "final_unique_ratio": last.get("diversity_unique_ratio", 0.0),

        "final_mutation_rate": last.get("current_mutation_rate", 0.0),
        "avg_mutation_rate": avg_from_history(history, "current_mutation_rate"),
        "max_mutation_rate": max(row.get("current_mutation_rate", 0.0) for row in history),

        "hypermutation_active_generations": count_true_from_history(history, "hypermutation_active"),
        "final_stagnation_generations": last.get("stagnation_generations", 0),
    }

    return row


def mean(values):
    return statistics.mean(values) if values else 0.0


def stdev(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def make_summary(rows):
    groups = {}

    for row in rows:
        key = (row["instance"], row["variant"], row["mutation_control_mode"])
        groups.setdefault(key, []).append(row)

    summary_rows = []

    for key, group_rows in groups.items():
        instance, variant, mode = key

        costs = [r["best_cost"] for r in group_rows]
        runtimes = [r["runtime"] for r in group_rows]
        final_divs = [r["final_jaccard_diversity"] for r in group_rows]
        avg_mut_rates = [r["avg_mutation_rate"] for r in group_rows]
        max_mut_rates = [r["max_mutation_rate"] for r in group_rows]
        hyper_counts = [r["hypermutation_active_generations"] for r in group_rows]
        generations = [r["generations_used"] for r in group_rows]

        summary_rows.append({
            "instance": instance,
            "variant": variant,
            "mode": mode,

            "avg_best_cost": mean(costs),
            "std_best_cost": stdev(costs),
            "best_cost_over_seeds": min(costs),
            "worst_cost_over_seeds": max(costs),

            "avg_runtime": mean(runtimes),
            "avg_generations_used": mean(generations),

            "avg_final_jaccard_diversity": mean(final_divs),
            "avg_mutation_rate": mean(avg_mut_rates),
            "max_mutation_rate": max(max_mut_rates),
            "avg_hypermutation_active_generations": mean(hyper_counts),

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
    print("\n\n================ MUTATION CONTROL SUMMARY ================")
    print(
        "instance | method | avg_cost | std | best | worst | div | "
        "avg_mut | max_mut | hyper_gens | time | feasible | seeds"
    )
    print("-" * 170)

    for row in summary_rows:
        print(
            f"{row['instance']:6s} | "
            f"{row['variant']:32s} | "
            f"avg_cost={row['avg_best_cost']:.2f} | "
            f"std={row['std_best_cost']:.2f} | "
            f"best={row['best_cost_over_seeds']} | "
            f"worst={row['worst_cost_over_seeds']} | "
            f"div={row['avg_final_jaccard_diversity']:.4f} | "
            f"avg_mut={row['avg_mutation_rate']:.4f} | "
            f"max_mut={row['max_mutation_rate']:.4f} | "
            f"hyper_gens={row['avg_hypermutation_active_generations']:.1f} | "
            f"time={row['avg_runtime']:.2f}s | "
            f"feasible={row['all_runs_feasible']} | "
            f"seeds={row['num_seeds']}"
        )


def print_best_methods(summary_rows):
    print("\n\n================ BEST MUTATION METHOD PER INSTANCE ================")

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
                    f"avg_mut={row['avg_mutation_rate']:.4f} | "
                    f"max_mut={row['max_mutation_rate']:.4f} | "
                    f"hyper_gens={row['hypermutation_active_generations']}"
                )

    summary_rows = make_summary(all_rows)

    detailed_path = os.path.join(RESULTS_DIR, "mutation_control_detailed.csv")
    summary_path = os.path.join(RESULTS_DIR, "mutation_control_summary.csv")

    write_csv(detailed_path, all_rows)
    write_csv(summary_path, summary_rows)

    print_summary(summary_rows)
    print_best_methods(summary_rows)

    print("\nSaved detailed results to:", detailed_path)
    print("Saved summary results to:", summary_path)
    print("Total runtime:", round(time.time() - start, 2), "seconds")


if __name__ == "__main__":
    main()