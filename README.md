# AI Lab 2 — Part 1: Evolution Control for Set Cover

This project extends the Genetic Algorithm from Lab 1 for the **Set Cover Problem** by adding several **evolution-control mechanisms**.

The goal of this part is to compare methods that fight premature convergence, preserve diversity, and improve the behavior of the genetic search.

The original Lab 1 GA code is kept as the baseline, and the Lab 2 mechanisms are implemented on top of it.

---

## Problem

The project solves benchmark **Set Cover** instances.

Each solution is represented as a binary chromosome:

```text
1 = selected set
0 = unselected set
```

The objective is to cover all elements while minimizing the total cost of the selected sets.

Because every chromosome is repaired before evaluation, all individuals are feasible. Therefore, the basic fitness is:

```python
fitness = -cost
```

So maximizing fitness is equivalent to minimizing solution cost.

---

## Main Project Files

| File | Purpose |
|---|---|
| `main.py` | Original Lab 1 experiment runner copied as the baseline |
| `ga.py` | Main Genetic Algorithm implementation, extended for Lab 2 |
| `distance.py` | Distance/similarity metrics: normalized Hamming and Jaccard distance |
| `niching.py` | Fitness Sharing / Niching and Threshold Speciation |
| `mutation_control.py` | Nonlinear mutation, triggered hypermutation, relative-fitness mutation, and age-based mutation |
| `adaptive_fitness.py` | Individual fitness with `g(x,t)` based on novelty or age |
| `fitness.py` | Cost and fitness calculation |
| `repair.py` | Repair mechanism for infeasible Set Cover chromosomes |
| `crossover.py` | One-point, two-point, and uniform crossover |
| `mutation.py` | Bit-flip mutation |
| `selection.py` | Selection helpers from the original implementation |
| `stats.py` | Statistics and diversity metrics |
| `parser.py` | Parser for SCP benchmark files |
| `problem.py` | `SetCoverProblem` class |
| `baseline.py` | Greedy Set Cover baseline |
| `plot_results.py` | Graph-generation file from the original Lab 1 project |
| `experiments.py` | Optional experiment helper file from the original project |

---

## Data Files

The benchmark files are stored in:

```text
data/
```

Used instances:

```text
scp41.txt
scp42.txt
scp43.txt
scp44.txt
scp51.txt
scp52.txt
scp53.txt
scpa1.txt
scpa2.txt
scpa3.txt
```

---

# Lab 2 Part 1 Sections

The assignment has two main groups of requirements:

1. **Fighting premature convergence**
2. **Population of genes / diversity methods**

Each implemented section has a matching `main_...py` file and a saved `.txt` result file.

---

# Section A1 — Nonlinear Mutation vs Triggered Hypermutation

## Requirement

Compare:

```text
Nonlinear mutation-rate control
vs
Triggered hypermutation
```

## Main file

```text
main_hw2_mutation_control_compare.py
```

## Implemented in

```text
mutation_control.py
ga.py
```

## What it compares

| Method | Description |
|---|---|
| Fixed mutation | Original constant mutation rate |
| Nonlinear decreasing mutation | Starts with higher mutation and decreases over time |
| Triggered hypermutation | Raises mutation rate when the GA stagnates |

## Result file

```text
results_a_1_lab2_part1.txt
```

## Main conclusion

Nonlinear mutation gave the best solution-quality behavior in this section, especially on `scpa3`. Triggered hypermutation produced the highest diversity, but it was sometimes too aggressive and could hurt exploitation.

---

# Section B1 — Relative-Fitness Mutation vs Age-Based Mutation

## Requirement

Compare:

```text
Individual adaptive mutation based on relative fitness
vs
Age-based individual mutation
```

## Main file

```text
main_hw2_individual_mutation_compare.py
```

## Implemented in

```text
mutation_control.py
ga.py
```

## What it compares

| Method | Description |
|---|---|
| No individual adaptive mutation | Baseline fixed mutation |
| Relative-fitness adaptive mutation | Better individuals get lower mutation, weaker individuals get higher mutation |
| Age-based adaptive mutation | Older individuals receive higher mutation |

## Result file

```text
results_lab2_part1_b1.txt
```

## Main conclusion

The first parameter setting was too conservative, so a second test was performed with stronger mutation bounds. With the stronger setting, age-based adaptive mutation improved diversity and achieved the best result on `scpa3` in that experiment.

---

# Section C1 — Individual Fitness with `g(x,t)` Based on Novelty vs Age

## Requirement

Compare:

```text
Individual fitness where g(x,t) is based on novelty
vs
Individual fitness where g(x,t) is based on age
```

## Main file

```text
main_hw2_adaptive_fitness_compare.py
```

## Implemented in

```text
adaptive_fitness.py
ga.py
```

## What it compares

| Method | Description |
|---|---|
| Objective fitness only | Uses only the original Set Cover cost |
| Novelty-based `g(x,t)` | Rewards chromosomes that are different from the rest of the population |
| Age-based `g(x,t)` | Adds a score based on how long an individual survived |

The selection score is computed as:

```text
F(x,t) = alpha * f(x) + (1 - alpha) * g(x,t)
```

Where:

```text
f(x) = objective quality based on cost
g(x,t) = novelty score or age score
```

## Result file

```text
results_c1_part1_lab2.txt
```

## Main conclusion

Novelty-based `g(x,t)` was more useful than age-based `g(x,t)`. With `alpha = 0.6`, novelty improved diversity and improved the result on `scp51`. With `alpha = 0.7`, novelty gave the best overall final HW2 configuration.

---

# Section A2 — Distance Function Between Set Cover Genes

## Requirement

Develop a function that measures distance or similarity between two genes/chromosomes for Set Cover.

## Implemented in

```text
distance.py
```

## Implemented metrics

| Metric | Description |
|---|---|
| Normalized Hamming distance | Measures bit-level difference between two chromosomes |
| Jaccard distance | Compares selected-set indices and ignores shared zero positions |

## Reasoning

Hamming distance is natural for binary chromosomes, but Set Cover chromosomes are sparse. Therefore, Jaccard distance is more meaningful because it focuses on which sets are actually selected.

This distance function is later used for:

- Diversity measurement
- Fitness Sharing / Niching
- Threshold Speciation
- Novelty-based `g(x,t)`

---

# Section B2 — Niching vs Threshold Speciation

## Requirement

Compare performance on Set Cover problems between:

```text
Niching / Fitness Sharing with a radius parameter
vs
Threshold Speciation with a variable number of species
```

Also check sensitivity to algorithm parameters.

## Main file

```text
main_hw2_niching_compare.py
```

## Implemented in

```text
niching.py
distance.py
ga.py
```

## What it compares

| Method | Main parameter | Description |
|---|---|---|
| Fitness Sharing / Niching | `sigma_share` | Reduces selection score in crowded regions |
| Threshold Speciation | `speciation_threshold` | Splits population into species according to distance from representatives |

## Result files

```text
results_b_2_lab2_part1.txt
second_results_B_2_part1.txt
```

## Main conclusion

Fitness Sharing preserved diversity but increased runtime. Threshold Speciation was highly sensitive to its threshold parameter. Small thresholds created too many tiny species, while large thresholds collapsed the population into one species. Therefore, these methods were useful for diversity analysis, but they were not selected as the final best HW2 configuration because they did not consistently improve cost.

---

# Final HW2 Configuration Selection

After testing the different Lab 2 mechanisms, the best general HW2 configuration selected was:

```text
Novelty-based individual fitness with alpha = 0.7
```

This configuration was selected because it gave the best total target-gap ranking among the tested HW2 candidates.

## Final configuration

```python
population_size = 300
generations = 500
crossover_rate = 0.91
mutation_rate = 0.015
elite_size = 10
crossover_type = "uniform"

niching_method = "none"
mutation_control_mode = "fixed"
individual_mutation_mode = "none"

adaptive_fitness_mode = "novelty"
adaptive_fitness_alpha = 0.7
adaptive_fitness_distance = "jaccard"
```

## Why Niching / Threshold Speciation were not selected for the final configuration

Niching and Threshold Speciation were useful for studying diversity, but they were not chosen as the final HW2 configuration because:

- They increased runtime significantly.
- They did not consistently improve solution cost.
- Threshold Speciation was very sensitive to its threshold parameter.
- Their main benefit was diversity preservation rather than better final cost.

Therefore, they are discussed in their own section, but the final selected HW2 configuration is based on novelty fitness.

---

# Final HW2 All-Problem Result

The final selected HW2 configuration was run on all 10 Set Cover benchmark instances.

The lecturer target ranges were:

```text
scp41: 429–433
scp42: 512–517
scp43: 516–521
scp44: 494–499
scp51: 253–255
scp52: 302–305
scp53: 226–228
scpa1: 253–255
scpa2: 252–254
scpa3: 232–234
```

The final HW2 configuration reached the target range on:

```text
7 / 10 problems
```

The total target gap was:

```text
10
```

The configuration reached the target range on:

```text
scp41
scp42
scp43
scp44
scp51
scp52
scp53
```

The remaining gaps were on:

```text
scpa1
scpa2
scpa3
```

---

# Important Result Files in the Repository

The main terminal outputs used for the report are saved as text files:

| Result file | Section |
|---|---|
| `results_a_1_lab2_part1.txt` | Section A1 — Nonlinear mutation vs triggered hypermutation |
| `results_lab2_part1_b1.txt` | Section B1 — Relative-fitness mutation vs age-based mutation |
| `results_c1_part1_lab2.txt` | Section C1 — Novelty-based fitness vs age-based fitness |
| `results_b_2_lab2_part1.txt` | Section B2 — Niching vs Threshold Speciation, first sensitivity test |
| `second_results_B_2_part1.txt` | Section B2 — Niching vs Threshold Speciation, second sensitivity test |

Some experiment files also save CSV files automatically inside:

```text
results/
```

Examples of possible CSV outputs:

```text
mutation_control_summary.csv
individual_mutation_summary.csv
individual_mutation_second_test_summary.csv
adaptive_fitness_summary.csv
niching_sensitivity_summary.csv
niching_large_thresholds_summary.csv
final_hw2_all10_summary.csv
```

---

# How to Run Each Section

Run commands from the project root.

## Section A1 — Mutation Control

```bash
python main_hw2_mutation_control_compare.py
```

Windows Python 3.13:

```bash
C:\Users\omar\AppData\Local\Programs\Python\Python313\python.exe main_hw2_mutation_control_compare.py
```

---

## Section B1 — Individual Adaptive Mutation

```bash
python main_hw2_individual_mutation_compare.py
```

Windows Python 3.13:

```bash
C:\Users\omar\AppData\Local\Programs\Python\Python313\python.exe main_hw2_individual_mutation_compare.py
```

---

## Section C1 — Individual Fitness with `g(x,t)`

```bash
python main_hw2_adaptive_fitness_compare.py
```

Windows Python 3.13:

```bash
C:\Users\omar\AppData\Local\Programs\Python\Python313\python.exe main_hw2_adaptive_fitness_compare.py
```

---

## Section B2 — Niching and Threshold Speciation

```bash
python main_hw2_niching_compare.py
```

Windows Python 3.13:

```bash
C:\Users\omar\AppData\Local\Programs\Python\Python313\python.exe main_hw2_niching_compare.py
```

---

## Original Lab 1 Baseline

```bash
python main.py
```

Windows Python 3.13:

```bash
C:\Users\omar\AppData\Local\Programs\Python\Python313\python.exe main.py
```

---

# Requirements

The project was tested with:

```text
Python 3.13
```

Required packages:

```text
numpy
matplotlib
pandas
```

Install using:

```bash
python -m pip install numpy matplotlib pandas
```

Or on Windows:

```bash
C:\Users\omar\AppData\Local\Programs\Python\Python313\python.exe -m pip install numpy matplotlib pandas
```

---

# Notes About Runtime

Some methods are much slower than the original GA because they calculate distances between chromosomes.

The slowest methods are usually:

- Novelty-based fitness
- Fitness Sharing
- Threshold Speciation

This is expected because they require extra population-level comparisons.

The faster methods are usually:

- Fixed mutation
- Age-based mutation
- Fast nonlinear mutation

---

# Notes About the Final Comparison

The original Lab 1 configuration was already strong. Therefore, Lab 2 does not dominate Lab 1 on every instance.

The best HW2 configuration improved or matched the original configuration on several problems, especially under same-seed comparison, but some Lab 1 configurations were still better on some instances.

The main contribution of Lab 2 is not only lower cost, but also analysis and control of the evolutionary process:

- Diversity preservation
- Stagnation handling
- Mutation-rate control
- Individual mutation policies
- Novelty-based selection pressure
- Species/niche behavior

---

# Main Conclusion

The Lab 2 extensions showed that evolution-control mechanisms can significantly change the behavior of the GA.

The strongest final mechanism was:

```text
Novelty-based individual fitness g(x,t), alpha = 0.7
```

It improved exploration and reached the target range on most benchmark instances.

However, the original Lab 1 GA was already highly competitive. The final conclusion is that the Lab 2 mechanisms are useful for controlling diversity and improving some instances, but they do not universally outperform the original GA on every Set Cover benchmark.
