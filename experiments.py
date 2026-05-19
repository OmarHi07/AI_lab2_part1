#This module runs experiments on multiple problem instances,
#configurations, and random seeds.
#את זה להוכיח שהאלגורית שלנו טוב
#הרצת האלגוריתם כמה פעמים והשוואת אותו לאלגוריתם greedy

from parser import parse_scp_file
from ga import run_ga
from baseline import greedy_set_cover


def run_single_experiment(instance_path, config, seed):
    #Run one GA experiment on one instance with one configuration and one seed.
    #Args:
    #    instance_path (str): Path to instance file.
    #    config (dict): GA configuration.
    #    seed (int): Random seed.
    #Returns:
    #    dict: Experiment result.
    #experiments = הרצות מסודרות של האלגוריתם
    #כדי לבדוק, להשוות, ולהסיק מסקנות

    problem = parse_scp_file(instance_path)

    result = run_ga(
        problem=problem,
        population_size=config["population_size"],
        generations=config["generations"],
        crossover_rate=config["crossover_rate"],
        mutation_rate=config["mutation_rate"],
        elite_size=config.get("elite_size", 1),
        seed=seed
    )

    return {
        "instance": instance_path,
        "seed": seed,
        "config": config,
        "best_cost": result["best_cost"],
        "total_time": result["total_time"],
        "history": result["history"],
    }


def run_baseline(instance_path):
    #Run the greedy baseline on one instance.
    #Args:
    #    instance_path (str): Path to instance file.
    #Returns:
    #    dict: Baseline result.


    problem = parse_scp_file(instance_path)
    solution, total_cost = greedy_set_cover(problem)

    return {
        "instance": instance_path,
        "baseline_cost": total_cost,
        "solution": solution
    }