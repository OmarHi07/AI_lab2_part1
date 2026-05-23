import csv
import os
import statistics
import time

from parser import parse_scp_file
from ga import run_ga
from fitness import is_feasible


RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# At least 3 different Set Cover problems.
# We use one from scp4x, one from scp5x, and one from scpa.
INSTANCES = [
    "data/scp41.txt",
    "data/scp51.txt",
    "data/scpa3.txt",
]



SEEDS = [42]


BASE_GA_PARAMS = {
    "population_size": 300,
    "generations": 200,
    "crossover_rate": 0.8,
    "mutation_rate": 0.05,
    "elite_size": 4,
    "crossover_type": "uniform",
    "show_plots": False,
    "verbose": False,
    "patience": 80,
    "min_generations": 100,
}


EXPERIMENT_CONFIGS = [
    {
        "method_group": "Original GA",
        "variant": "Original GA",
        "niching_method": "none",
        "sigma_share": 0.0,
        "speciation_threshold": 0.0,
    },

    # Fitness Sharing / Niching sensitivity to sigma_share
    {
        "method_group": "Fitness Sharing",
        "variant": "Fitness Sharing sigma=0.50",
        "niching_method": "fitness_sharing",
        "sigma_share": 0.50,
        "speciation_threshold": 0.0,
    },
    {
        "method_group": "Fitness Sharing",
        "variant": "Fitness Sharing sigma=0.70",
        "niching_method": "fitness_sharing",
        "sigma_share": 0.70,
        "speciation_threshold": 0.0,
    },
    {
        "method_group": "Fitness Sharing",
        "variant": "Fitness Sharing sigma=0.90",
        "niching_method": "fitness_sharing",
        "sigma_share": 0.90,
        "speciation_threshold": 0.0,
    },

    # Threshold Speciation sensitivity to threshold
    {
        "method_group": "Threshold Speciation",
        "variant": "Threshold Speciation threshold=0.30",
        "niching_method": "threshold_speciation",
        "sigma_share": 0.0,
        "speciation_threshold": 0.30,
    },
    {
        "method_group": "Threshold Speciation",
        "variant": "Threshold Speciation threshold=0.50",
        "niching_method": "threshold_speciation",
        "sigma_share": 0.0,
        "speciation_threshold": 0.50,
    },
    {
        "method_group": "Threshold Speciation",
        "variant": "Threshold Speciation threshold=0.70",
        "niching_method": "threshold_speciation",
        "sigma_share": 0.0,
        "speciation_threshold": 0.70,
    },
    {
        "method_group": "Threshold Speciation",
        "variant": "Threshold Speciation threshold=0.85",
        "niching_method": "threshold_speciation",
        "sigma_share": 0.0,
        "speciation_threshold": 0.85,
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

        niching_method=config["niching_method"],
        niche_distance="jaccard",

        sigma_share=config["sigma_share"],
        sharing_alpha=1.0,
        sharing_sample_size=30,

        speciation_threshold=config["speciation_threshold"],
    )

    history = result["history"]
    last = history[-1]

    row = {
        "instance": short_instance_name(instance_path),
        "seed": seed,

        "method_group": config["method_group"],
        "variant": config["variant"],
        "niching_method": config["niching_method"],

        "sigma_share": config["sigma_share"],
        "speciation_threshold": config["speciation_threshold"],

        "best_cost": result["best_cost"],
        "feasible": is_feasible(result["best_solution"], problem),
        "runtime": result["total_time"],
        "generations_used": len(history),

        "final_jaccard_diversity": last.get("diversity_jaccard", 0.0),
        "avg_jaccard_diversity": avg_from_history(history, "diversity_jaccard"),

        "final_unique_ratio": last.get("diversity_unique_ratio", 0.0),
        "avg_unique_ratio": avg_from_history(history, "diversity_unique_ratio"),

        "final_species_count": last.get("species_count", 0),
        "avg_species_count": avg_from_history(history, "species_count"),

        "final_avg_species_size": last.get("avg_species_size", 0.0),
        "avg_species_size": avg_from_history(history, "avg_species_size"),

        "final_sharing_niche_count": last.get("sharing_avg_niche_count", 0.0),
        "avg_sharing_niche_count": avg_from_history(history, "sharing_avg_niche_count"),
    }

    return row


def mean(values):
    return statistics.mean(values) if values else 0.0


def stdev(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def make_summary(rows):
    groups = {}

    for row in rows:
        key = (
            row["instance"],
            row["method_group"],
            row["variant"],
            row["sigma_share"],
            row["speciation_threshold"],
        )
        groups.setdefault(key, []).append(row)

    summary_rows = []

    for key, group_rows in groups.items():
        instance, method_group, variant, sigma_share, speciation_threshold = key

        costs = [r["best_cost"] for r in group_rows]
        runtimes = [r["runtime"] for r in group_rows]
        final_diversities = [r["final_jaccard_diversity"] for r in group_rows]
        avg_diversities = [r["avg_jaccard_diversity"] for r in group_rows]
        unique_ratios = [r["final_unique_ratio"] for r in group_rows]
        species_counts = [r["final_species_count"] for r in group_rows]
        species_sizes = [r["final_avg_species_size"] for r in group_rows]
        sharing_counts = [r["final_sharing_niche_count"] for r in group_rows]
        generations = [r["generations_used"] for r in group_rows]

        summary_rows.append({
            "instance": instance,
            "method_group": method_group,
            "variant": variant,
            "sigma_share": sigma_share,
            "speciation_threshold": speciation_threshold,

            "avg_best_cost": mean(costs),
            "std_best_cost": stdev(costs),
            "best_cost_over_seeds": min(costs),
            "worst_cost_over_seeds": max(costs),

            "avg_runtime": mean(runtimes),
            "avg_generations_used": mean(generations),

            "avg_final_jaccard_diversity": mean(final_diversities),
            "avg_jaccard_diversity_over_run": mean(avg_diversities),
            "avg_final_unique_ratio": mean(unique_ratios),

            "avg_final_species_count": mean(species_counts),
            "avg_final_species_size": mean(species_sizes),

            "avg_final_sharing_niche_count": mean(sharing_counts),

            "all_runs_feasible": all(r["feasible"] for r in group_rows),
            "num_seeds": len(group_rows),
        })

    return summary_rows


def write_csv(path, rows):
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary_rows):
    print("\n\n================ FINAL SUMMARY ================")
    print(
        "instance | method | avg_cost | std | best | worst | "
        "div | unique | species | species_size | share_niche | time | feasible | seeds"
    )
    print("-" * 160)

    for row in summary_rows:
        print(
            f"{row['instance']:6s} | "
            f"{row['variant']:38s} | "
            f"avg_cost={row['avg_best_cost']:.2f} | "
            f"std={row['std_best_cost']:.2f} | "
            f"best={row['best_cost_over_seeds']} | "
            f"worst={row['worst_cost_over_seeds']} | "
            f"div={row['avg_final_jaccard_diversity']:.4f} | "
            f"unique={row['avg_final_unique_ratio']:.4f} | "
            f"species={row['avg_final_species_count']:.2f} | "
            f"species_size={row['avg_final_species_size']:.2f} | "
            f"share_niche={row['avg_final_sharing_niche_count']:.4f} | "
            f"time={row['avg_runtime']:.2f}s | "
            f"feasible={row['all_runs_feasible']} | "
            f"seeds={row['num_seeds']}"
        )

def print_best_methods(summary_rows):
    print("\n\n================ BEST METHOD PER INSTANCE ================")

    instances = sorted(set(row["instance"] for row in summary_rows))

    for instance in instances:
        rows = [row for row in summary_rows if row["instance"] == instance]

        best_cost_row = min(rows, key=lambda r: r["avg_best_cost"])
        best_diversity_row = max(rows, key=lambda r: r["avg_final_jaccard_diversity"])
        fastest_row = min(rows, key=lambda r: r["avg_runtime"])

        print(f"\nInstance: {instance}")

        print(
            "Best average cost:",
            best_cost_row["variant"],
            "| avg_cost =",
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
            "| avg_cost =",
            round(best_diversity_row["avg_best_cost"], 2),
            "| time =",
            round(best_diversity_row["avg_runtime"], 2),
            "s"
        )

        print(
            "Fastest method:",
            fastest_row["variant"],
            "| time =",
            round(fastest_row["avg_runtime"], 2),
            "s",
            "| avg_cost =",
            round(fastest_row["avg_best_cost"], 2),
            "| diversity =",
            round(fastest_row["avg_final_jaccard_diversity"], 4)
        )

def main():
    all_rows = []

    overall_start = time.time()

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
                    f"unique={row['final_unique_ratio']:.4f} | "
                    f"species={row['final_species_count']} | "
                    f"share_niche={row['final_sharing_niche_count']:.4f}"
                )

    summary_rows = make_summary(all_rows)

    detailed_path = os.path.join(RESULTS_DIR, "niching_large_thresholds_detailed.csv")
    summary_path = os.path.join(RESULTS_DIR, "niching_large_thresholds_summary.csv")

    write_csv(detailed_path, all_rows)
    write_csv(summary_path, summary_rows)

    print_summary(summary_rows)
    print_best_methods(summary_rows)

    print("\nSaved detailed results to:", detailed_path)
    print("Saved summary results to:", summary_path)
    print("Total runtime:", round(time.time() - overall_start, 2), "seconds")


if __name__ == "__main__":
    main()