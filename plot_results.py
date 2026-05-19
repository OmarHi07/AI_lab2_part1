import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates


RESULTS_DIR = "results"
PLOTS_DIR = "plots"

# ============================================================
# Settings
# ============================================================

PLOT_SEED = "42"

# 3 problems for best / average / worst fitness graphs
INTERESTING_PROBLEMS = ["scp41", "scp51", "scpa3"]

# 2 problems for diversity graphs
DIVERSITY_PROBLEMS = ["scp51", "scpa3"]

# 2 problems for representative generation graphs
REPRESENTATIVE_PROBLEMS = ["scp41", "scpa3"]

# Your best configuration
BEST_CONFIG_KEYWORD = "Pop300_Gen500_C0_91_M0_015_uniform"

# Required configuration from the assignment
REQUIRED_CONFIG_KEYWORD = "Required_Config_Pop300_Gen200_C0_8_M0_05_one_point"

# We use both configs for fitness, diversity, and representative graphs
CONFIGS_TO_PLOT = [
    ("best_config", BEST_CONFIG_KEYWORD),
    ("required_config", REQUIRED_CONFIG_KEYWORD),
]


# ============================================================
# Helpers
# ============================================================

def ensure_plots_folder():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def clean_name(text):
    return (
        str(text)
        .replace(" ", "_")
        .replace(":", "")
        .replace("=", "")
        .replace(".", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(",", "_")
        .replace("(", "")
        .replace(")", "")
    )


def normalize_series(series):
    min_val = series.min()
    max_val = series.max()

    if max_val == min_val:
        return series * 0

    return (series - min_val) / (max_val - min_val)


def pick_first_existing_column(df, candidate_columns):
    for col in candidate_columns:
        if col in df.columns:
            return col
    return None


def extract_config_value(config, key, default=0.0):
    """
    Supports both formats:
    Pop=300, Gen=500, C=0.91, M=0.015
    and:
    Pop300_Gen500_C0_91_M0_015
    """
    config = str(config)

    patterns = {
        "Pop": [r"Pop=(\d+)", r"Pop(\d+)"],
        "Gen": [r"Gen=(\d+)", r"Gen(\d+)"],
        "C": [r"C=(\d+(?:\.\d+)?)", r"C(\d+(?:_\d+)?)"],
        "M": [r"M=(\d+(?:\.\d+)?)", r"M(\d+(?:_\d+)?)"],
    }

    for pattern in patterns[key]:
        match = re.search(pattern, config)
        if match:
            value = match.group(1).replace("_", ".")
            return float(value)

    return default


def make_config_short_names(configs):
    unique_configs = list(pd.Series(configs).drop_duplicates())

    mapping = {
        config: f"C{i + 1}"
        for i, config in enumerate(unique_configs)
    }

    legend_df = pd.DataFrame([
        {"config_short": short, "config_full": full}
        for full, short in mapping.items()
    ])

    legend_path = os.path.join(PLOTS_DIR, "config_legend.csv")
    legend_df.to_csv(legend_path, index=False)
    print(f"Saved config legend: {legend_path}")

    return mapping


# ============================================================
# History file loading
# ============================================================

def history_candidate_score(filename, config_keyword=None):
    name = filename.lower()
    score = 0

    if config_keyword and config_keyword.lower() in name:
        score += 100

    if "gen500" in name:
        score += 20

    if "uniform" in name:
        score += 5

    return score


def collect_history_for_problem_seed(instance_name, seed, config_keyword):
    """
    Finds exactly one history file for:
    problem + seed + configuration keyword
    """
    files = glob.glob(os.path.join(RESULTS_DIR, "history_*.csv"))

    if not files:
        raise FileNotFoundError("No history_*.csv files found inside results/ folder.")

    candidates = []

    for path in files:
        name = os.path.basename(path).lower()

        same_instance = instance_name.lower() in name
        same_seed = f"seed{seed}" in name
        same_config = config_keyword.lower() in name

        if same_instance and same_seed and same_config:
            candidates.append(path)

    if not candidates:
        print(
            f"Warning: no exact history file found for "
            f"instance={instance_name}, seed={seed}, config={config_keyword}"
        )
        return None, None

    candidates = sorted(
        candidates,
        key=lambda p: history_candidate_score(os.path.basename(p), config_keyword),
        reverse=True
    )

    chosen_path = candidates[0]
    print(f"Using: {os.path.basename(chosen_path)}")

    history_df = pd.read_csv(chosen_path).sort_values("generation")
    return chosen_path, history_df


# ============================================================
# Graph 1:
# Best / Average / Worst fitness in ONE graph
# 3 problems × 2 configs = 6 graphs
# ============================================================

def plot_best_avg_worst_fitness_combined(history_df, instance_name, config_label):
    required_cols = ["generation", "best_fitness", "avg_fitness", "worst_fitness"]

    for col in required_cols:
        if col not in history_df.columns:
            print(f"Skipping fitness graph for {instance_name}, {config_label}: missing column {col}")
            return

    plt.figure(figsize=(10, 6))

    plt.plot(
        history_df["generation"],
        history_df["best_fitness"],
        label="Best Fitness"
    )

    plt.plot(
        history_df["generation"],
        history_df["avg_fitness"],
        label="Average Fitness"
    )

    plt.plot(
        history_df["generation"],
        history_df["worst_fitness"],
        label="Worst Fitness"
    )

    plt.title(f"Best / Average / Worst Fitness - {instance_name} - {config_label} - Seed {PLOT_SEED}")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_name = f"01_{instance_name}_{config_label}_seed{PLOT_SEED}_best_avg_worst_fitness.png"
    path = os.path.join(PLOTS_DIR, output_name)

    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved: {path}")


def plot_fitness_graphs():
    for config_label, config_keyword in CONFIGS_TO_PLOT:
        for instance_name in INTERESTING_PROBLEMS:
            _, history_df = collect_history_for_problem_seed(
                instance_name=instance_name,
                seed=PLOT_SEED,
                config_keyword=config_keyword
            )

            if history_df is None:
                continue

            plot_best_avg_worst_fitness_combined(
                history_df=history_df,
                instance_name=instance_name,
                config_label=config_label
            )


# ============================================================
# Graph 2:
# Population diversity over generations
# 2 problems × 2 configs = 4 graphs
# ============================================================

def plot_population_diversity_one_graph(history_df, instance_name, config_label):
    diversity_cols = [
        "diversity_hamming",
        "diversity_active_genes",
        "diversity_unique_ratio",
    ]

    found_any = False

    plt.figure(figsize=(10, 6))

    for col in diversity_cols:
        if col not in history_df.columns:
            continue

        series = history_df[col]

        # Normalize all diversity measures so they can appear together clearly.
        series = normalize_series(series)

        plt.plot(
            history_df["generation"],
            series,
            label=f"{col} normalized"
        )

        found_any = True

    if not found_any:
        plt.close()
        print(f"Skipping diversity graph for {instance_name}, {config_label}: no diversity columns found.")
        return

    plt.title(f"Population Diversity - {instance_name} - {config_label} - Seed {PLOT_SEED}")
    plt.xlabel("Generation")
    plt.ylabel("Normalized Diversity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_name = f"02_{instance_name}_{config_label}_seed{PLOT_SEED}_population_diversity.png"
    path = os.path.join(PLOTS_DIR, output_name)

    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved: {path}")


def plot_population_diversity_graphs():
    for config_label, config_keyword in CONFIGS_TO_PLOT:
        for instance_name in DIVERSITY_PROBLEMS:
            _, history_df = collect_history_for_problem_seed(
                instance_name=instance_name,
                seed=PLOT_SEED,
                config_keyword=config_keyword
            )

            if history_df is None:
                continue

            plot_population_diversity_one_graph(
                history_df=history_df,
                instance_name=instance_name,
                config_label=config_label
            )


# ============================================================
# Summary preparation
# Adds Greedy baseline + keeps full configuration names
# ============================================================

def prepare_summary_plot_df(summary_df):
    """
    Prepares exact values for summary graphs.

    It creates a long-format table with:
    - all GA configurations
    - Greedy Baseline for each problem, if greedy_cost exists

    The graph legend uses the full configuration names, not C1/C2.
    """
    required_cols = ["instance", "config", "avg_ga_cost", "avg_ga_time"]

    for col in required_cols:
        if col not in summary_df.columns:
            raise ValueError(f"Missing required column in final_summary.csv: {col}")

    plot_df = summary_df.copy()

    plot_df["instance_short"] = (
        plot_df["instance"]
        .astype(str)
        .str.replace("data/", "", regex=False)
        .str.replace(".txt", "", regex=False)
    )

    # GA rows
    ga_grouped = (
        plot_df
        .groupby(["instance_short", "config"], as_index=False)
        .agg({
            "avg_ga_cost": "mean",
            "avg_ga_time": "mean",
        })
    )

    ga_grouped = ga_grouped.rename(columns={
        "config": "method",
        "avg_ga_cost": "solution_cost",
        "avg_ga_time": "solution_time",
    })

    all_rows = [ga_grouped]

    # Greedy baseline rows for cost
    if "greedy_cost" in plot_df.columns:
        greedy_agg_dict = {
            "greedy_cost": "mean"
        }

        # Add greedy time only if it exists
        if "greedy_time" in plot_df.columns:
            greedy_agg_dict["greedy_time"] = "mean"

        greedy_grouped = (
            plot_df
            .groupby("instance_short", as_index=False)
            .agg(greedy_agg_dict)
        )

        greedy_grouped["method"] = "Greedy Baseline"
        greedy_grouped = greedy_grouped.rename(columns={
            "greedy_cost": "solution_cost",
            "greedy_time": "solution_time",
        })

        if "solution_time" not in greedy_grouped.columns:
            greedy_grouped["solution_time"] = None

        greedy_grouped = greedy_grouped[
            ["instance_short", "method", "solution_cost", "solution_time"]
        ]

        all_rows.append(greedy_grouped)
    else:
        print("Warning: greedy_cost column not found, so Greedy Baseline will not appear in cost summary graph.")

    grouped = pd.concat(all_rows, ignore_index=True)

    values_path = os.path.join(PLOTS_DIR, "summary_values_used_for_graphs.csv")
    grouped.to_csv(values_path, index=False)
    print(f"Saved summary values used for graphs: {values_path}")

    return grouped


# ============================================================
# Graph 3:
# Average solution cost over all problems and configurations
# including Greedy Baseline
# ============================================================

def plot_average_solution_cost_summary(summary_df):
    grouped = prepare_summary_plot_df(summary_df)

    pivot_df = grouped.pivot(
        index="instance_short",
        columns="method",
        values="solution_cost"
    )

    # Put Greedy Baseline last in the legend/order
    cols = [c for c in pivot_df.columns if c != "Greedy Baseline"]
    if "Greedy Baseline" in pivot_df.columns:
        cols.append("Greedy Baseline")
    pivot_df = pivot_df[cols]

    colors = [
        "#4C78A8",  # blue
        "#F58518",  # orange
        "#54A24B",  # green
        "#E45756",  # red
        "#72B7B2",  # teal
        "#B279A2",  # purple
        "#FF9DA6",  # pink
        "#9D755D",  # brown
        "#BAB0AC",  # gray
        "#000000",  # black, good for Greedy Baseline if last
    ]

    color_list = colors[:len(pivot_df.columns)]

    ax = pivot_df.plot(
        kind="bar",
        figsize=(15, 8),
        color=color_list
    )

    plt.title("Average Solution Cost by Problem and Configuration")
    plt.xlabel("Problem Instance")
    plt.ylabel("Average Solution Cost")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y")

    plt.legend(
        title="Configuration / Method",
        fontsize=8,
        title_fontsize=9,
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.tight_layout(rect=[0, 0, 0.72, 1])

    path = os.path.join(PLOTS_DIR, "03_average_solution_cost_summary.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved: {path}")


# ============================================================
# Graph 4:
# Average solution time over all problems and configurations
# including Greedy Baseline if greedy_time exists
# ============================================================

def plot_average_solution_time_summary(summary_df):
    grouped = prepare_summary_plot_df(summary_df)

    # If greedy_time does not exist, Greedy Baseline has None values here.
    # We remove empty columns to avoid a blank bar.
    time_df = grouped.dropna(subset=["solution_time"]).copy()

    pivot_df = time_df.pivot(
        index="instance_short",
        columns="method",
        values="solution_time"
    )

    cols = [c for c in pivot_df.columns if c != "Greedy Baseline"]
    if "Greedy Baseline" in pivot_df.columns:
        cols.append("Greedy Baseline")
    pivot_df = pivot_df[cols]

    colors = [
        "#4C78A8",
        "#F58518",
        "#54A24B",
        "#E45756",
        "#72B7B2",
        "#B279A2",
        "#FF9DA6",
        "#9D755D",
        "#BAB0AC",
        "#000000",
    ]

    color_list = colors[:len(pivot_df.columns)]

    pivot_df.plot(
        kind="bar",
        figsize=(15, 8),
        color=color_list
    )

    plt.title("Average Runtime by Problem and Configuration")
    plt.xlabel("Problem Instance")
    plt.ylabel("Average Time (seconds)")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y")

    plt.legend(
        title="Configuration / Method",
        fontsize=8,
        title_fontsize=9,
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.tight_layout(rect=[0, 0, 0.72, 1])

    path = os.path.join(PLOTS_DIR, "04_average_solution_time_summary.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved: {path}")

# ============================================================
# Graph 5:
# Representative generation graph
# 2 problems × 2 configs = 4 graphs
#
# Chosen graph: Best Cost Over Generations
# If best_cost does not exist, fallback to best_fitness.
# ============================================================

def plot_representative_generation_graph(history_df, instance_name, config_label):
    if "best_cost" in history_df.columns:
        y_col = "best_cost"
        y_label = "Best Cost"
        title_metric = "Best Cost"
    elif "best_fitness" in history_df.columns:
        y_col = "best_fitness"
        y_label = "Best Fitness"
        title_metric = "Best Fitness"
    else:
        print(
            f"Skipping representative graph for {instance_name}, {config_label}: "
            f"no best_cost or best_fitness column found."
        )
        return

    plt.figure(figsize=(10, 6))

    plt.plot(
        history_df["generation"],
        history_df[y_col],
        label=title_metric
    )

    plt.title(f"{title_metric} Over Generations - {instance_name} - {config_label} - Seed {PLOT_SEED}")
    plt.xlabel("Generation")
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_name = f"05_{instance_name}_{config_label}_seed{PLOT_SEED}_representative_{clean_name(title_metric)}.png"
    path = os.path.join(PLOTS_DIR, output_name)

    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved: {path}")


def plot_representative_generation_graphs():
    for config_label, config_keyword in CONFIGS_TO_PLOT:
        for instance_name in REPRESENTATIVE_PROBLEMS:
            _, history_df = collect_history_for_problem_seed(
                instance_name=instance_name,
                seed=PLOT_SEED,
                config_keyword=config_keyword
            )

            if history_df is None:
                continue

            plot_representative_generation_graph(
                history_df=history_df,
                instance_name=instance_name,
                config_label=config_label
            )




# ============================================================
# Main
# ============================================================

def main():
    ensure_plots_folder()

    summary_path = os.path.join(RESULTS_DIR, "final_summary.csv")

    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            "Could not find results/final_summary.csv. "
            "Run main.py first and make sure it saves the results."
        )

    summary_df = pd.read_csv(summary_path)

    # 1. 6 graphs:
    # 3 problems × 2 configs,
    # each graph contains best + average + worst fitness
    plot_fitness_graphs()

    # 2. 4 graphs:
    # 2 problems × 2 configs
    plot_population_diversity_graphs()

    # 3. Summary graph:
    # average solution cost over all problems and configurations
    plot_average_solution_cost_summary(summary_df)

    # 4. Summary graph:
    # average solution time over all problems and configurations
    plot_average_solution_time_summary(summary_df)

    # 5. 4 graphs:
    # 2 problems × 2 configs
    plot_representative_generation_graphs()


    print("\nDone. All requested plots were saved inside the plots/ folder.")
    print("\nExpected graph count:")
    print("- 6 fitness graphs")
    print("- 4 diversity graphs")
    print("- 4 representative generation graphs")
    print("- 2 summary graphs")



if __name__ == "__main__":
    main()