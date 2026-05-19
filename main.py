from parser import parse_scp_file
from ga import run_ga
from baseline import greedy_set_cover
from fitness import is_feasible
import time
import statistics
import csv
import os

def safe_name(text):
    return (
        text.replace("data/", "")
            .replace(".txt", "")
            .replace(" ", "_")
            .replace(":", "")
            .replace("=", "")
            .replace(".", "_")
            .replace("/", "_")
            .replace("\\", "_")
    )

def main():
    os.makedirs("results", exist_ok=True)

    instances = [
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

    configs = [
        {
            "label": "Required Config: Pop=300 Gen=200 C=0.8 M=0.05 one_point",
            "population_size": 300,
            "generations": 200,
            "crossover_rate": 0.8,
            "mutation_rate": 0.05,
            "elite_size": 10,
            "crossover_type": "one_point",
            "patience": 80,
            "min_generations": 100,
        },
        {
            "label": "Config: Pop=300 Gen=500 C=0.91 M=0.015 uniform",
            "population_size": 300,
            "generations": 500,
            "crossover_rate": 0.91,
            "mutation_rate": 0.015,
            "elite_size": 10,
            "crossover_type": "uniform",
            "patience": 80,
            "min_generations": 100,
        },
        {
            "label": "Crossover Only: Pop=300 Gen=200 C=0.9 M=0 uniform",
            "population_size": 300,
            "generations": 200,
            "crossover_rate": 0.9,
            "mutation_rate": 0.0,
            "elite_size": 10,
            "crossover_type": "uniform",
            "patience": 80,
            "min_generations": 100,
        },
        {
            "label": "Mutation Only: Pop=300 Gen=200 C=0 M=0.015 uniform",
            "population_size": 300,
            "generations": 200,
            "crossover_rate": 0.0,
            "mutation_rate": 0.015,
            "elite_size": 10,
            "crossover_type": "uniform",
            "patience": 80,
            "min_generations": 100,
        },
    ]

    seeds = [42, 123, 999, 2026, 7]
    all_results = []

    for instance_path in instances:
        print("\n========================================")
        print("Running instance:", instance_path, flush=True)

        problem = parse_scp_file(instance_path)

        print(f"Problem size: m={problem.m} elements, n={problem.n} sets")

        start = time.time()
        greedy_solution, greedy_cost = greedy_set_cover(problem)
        greedy_time = time.time() - start
        greedy_valid = is_feasible(greedy_solution, problem)

        print(f"Greedy: cost={greedy_cost}, time={greedy_time:.4f}s, feasible={greedy_valid}")

        for config in configs:
            print("\n---", config["label"], "---", flush=True)

            seed_results = []
            ga_costs = []
            ga_times = []
            ga_feasible_list = []
            ga_generations = []

            for seed in seeds:
                result = run_ga(
                    problem=problem,
                    population_size=config["population_size"],
                    generations=config["generations"],
                    crossover_rate=config["crossover_rate"],
                    mutation_rate=config["mutation_rate"],
                    elite_size=config["elite_size"],
                    crossover_type=config["crossover_type"],
                    seed=seed,
                    show_plots=False,
                    verbose=False,
                    patience=config["patience"],
                    min_generations=config["min_generations"],
                )

                ga_solution = result["best_solution"]
                ga_cost = result["best_cost"]
                ga_time = result["total_time"]
                generations_used = len(result["history"])
                ga_valid = is_feasible(ga_solution, problem)

                ga_costs.append(ga_cost)
                ga_times.append(ga_time)
                ga_feasible_list.append(ga_valid)
                ga_generations.append(generations_used)

                seed_results.append({
                    "seed": seed,
                    "cost": ga_cost,
                    "time": ga_time,
                    "feasible": ga_valid,
                })

                history_file = f"results/history_{safe_name(instance_path)}_seed{seed}_{safe_name(config['label'])}.csv"

                if result["history"]:
                    with open(history_file, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=result["history"][0].keys())
                        writer.writeheader()
                        writer.writerows(result["history"])

                print(
                    f"Seed {seed}: cost={ga_cost}, time={ga_time:.2f}s, "
                    f"generations={generations_used}, feasible={ga_valid}"
                )

            avg_ga_cost = statistics.mean(ga_costs)
            best_ga_cost = min(ga_costs)
            worst_ga_cost = max(ga_costs)
            std_ga_cost = statistics.stdev(ga_costs) if len(ga_costs) > 1 else 0.0

            avg_ga_time = statistics.mean(ga_times)
            avg_generations = statistics.mean(ga_generations)
            best_seed = seed_results[ga_costs.index(best_ga_cost)]["seed"]

            all_feasible = all(ga_feasible_list)
            better_method = "GA" if avg_ga_cost < greedy_cost else "Greedy"

            print("\nSummary for config:")
            print(f"Best GA cost: {best_ga_cost} achieved by seed {best_seed}")
            print(f"Average GA cost: {avg_ga_cost:.2f}")
            print(f"Worst GA cost: {worst_ga_cost}")
            print(f"Std GA cost: {std_ga_cost:.2f}")
            print(f"Average GA time: {avg_ga_time:.2f}s")
            print(f"All GA runs feasible: {all_feasible}")
            print(f"Better method on average: {better_method}")
            print(f"Average generations until stop: {avg_generations:.2f}")

            row = {
                "instance": instance_path,
                "config": config["label"],
                "greedy_cost": greedy_cost,
                "greedy_time": greedy_time,
                "best_ga_cost": best_ga_cost,
                "best_seed": best_seed,
                "avg_ga_cost": avg_ga_cost,
                "worst_ga_cost": worst_ga_cost,
                "std_ga_cost": std_ga_cost,
                "avg_ga_time": avg_ga_time,
                "all_ga_feasible": all_feasible,
                "better_method": better_method,
                "avg_generations": avg_generations,
            }

            all_results.append(row)

    print("\n========================================")
    print("FINAL SUMMARY")
    print("instance | config | greedy | best_ga | avg_ga | std | avg_time | best_seed | better")

    for row in all_results:
        print(
            f"{row['instance']} | "
            f"{row['config']} | "
            f"greedy={row['greedy_cost']} | "
            f"best_ga={row['best_ga_cost']} | "
            f"avg_ga={row['avg_ga_cost']:.2f} | "
            f"gens={row['avg_generations']:.2f} | "
            f"std={row['std_ga_cost']:.2f} | "
            f"time={row['avg_ga_time']:.2f}s | "
            f"seed={row['best_seed']} | "
            f"{row['better_method']}"
        )

    summary_file = "results/final_summary.csv"

    with open(summary_file, "w", newline="") as f:
        fieldnames = all_results[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nSaved final summary to {summary_file}")


if __name__ == "__main__":
    main()