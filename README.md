# Genetic Algorithm for Set Cover Problem

This project implements a Genetic Algorithm (GA) for solving the **Set Cover Problem** using benchmark SCP instances.

The goal is to choose a subset of available sets such that all elements are covered while minimizing the total cost.

The project also includes:

- Greedy baseline algorithm
- Genetic Algorithm with multiple configurations
- Repair mechanism for infeasible chromosomes
- Tournament selection
- One-point, two-point, and uniform crossover
- Bit-flip mutation
- Elitism
- Early stopping
- Diversity metrics
- CSV result saving
- Graph generation from experiment results

---

## Project Structure

```text
LAB1_Q2/
│
├── main.py              # Main experiment runner
├── ga.py                # Genetic Algorithm implementation
├── chromosome.py        # Chromosome representation and creation
├── crossover.py         # Crossover operators
├── mutation.py          # Mutation operator
├── repair.py            # Repair method for infeasible solutions
├── selection.py         # Parent selection methods
├── fitness.py           # Cost, feasibility, and fitness functions
├── baseline.py          # Greedy baseline algorithm
├── parser.py            # SCP file parser
├── problem.py           # SetCoverProblem class
├── stats.py             # Statistics and diversity metrics
├── plot_results.py      # Creates graphs from saved CSV results
├── experiments.py       # Optional experiment helper functions
│
├── data/                # SCP benchmark files should be placed here
│   ├── scp41.txt
│   ├── scp42.txt
│   ├── scp43.txt
│   ├── scp44.txt
│   ├── scp51.txt
│   ├── scp52.txt
│   ├── scp53.txt
│   ├── scpa1.txt
│   ├── scpa2.txt
│   └── scpa3.txt
│
├── results/             # Created automatically after running main.py
│   ├── final_summary.csv
│   └── history_*.csv
│
└── plots/               # Created automatically after running plot_results.py
    ├── 01_best_average_worst_fitness.png
    ├── 02_population_diversity.png
    ├── 03_selection_pressure.png
    ├── 04_configuration_cost_comparison.png
    ├── 05_parameter_sensitivity_parallel_coordinates.png
    └── 06_best_cost_over_generations.png
```

---

## Requirements

The project was tested with:

```text
Python 3.13
```

Required Python packages:

```text
numpy
matplotlib
pandas
```

Install them using:

```bash
python -m pip install numpy matplotlib pandas
```

If you are using Windows and Python 3.13 directly:

```bash
C:\Users\omar\AppData\Local\Programs\Python\Python313\python.exe -m pip install numpy matplotlib pandas
```

You can also create a `requirements.txt` file with:

```text
numpy
matplotlib
pandas
```

Then install using:

```bash
python -m pip install -r requirements.txt
```

---

## Dataset

The SCP benchmark files should be placed inside a folder named:

```text
data/
```

The project expects the following files:

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

The expected SCP format is:

```text
m n
c1 c2 c3 ... cn
k i1 i2 i3 ... ik
...
```

Where:

- `m` is the number of elements that must be covered
- `n` is the number of available sets
- `cj` is the cost of set `j`
- each following row describes which sets cover each element

The parser converts the 1-based set indices from the file into 0-based Python indices.

---

## Solution Representation

Each solution is represented as a binary chromosome:

```text
1 = the set is selected
0 = the set is not selected
```

Example:

```python
[1, 0, 1, 0, 0]
```

This means that set 1 and set 3 are selected.

---

## Fitness Function

The algorithm minimizes the total cost of the selected sets.

Since every chromosome is repaired before evaluation, all individuals in the population are feasible. Therefore, the fitness is defined as:

```python
fitness = -cost
```

This means that maximizing fitness is equivalent to minimizing cost.

Example:

```text
cost = 228  -> fitness = -228
cost = 300  -> fitness = -300
```

Since `-228 > -300`, the lower-cost solution is considered better.

---

## Repair Mechanism

The GA uses a repair method instead of only penalizing infeasible solutions.

The repair method:

1. Adds useful sets until all elements are covered.
2. Removes redundant selected sets while preserving feasibility.

The optimized repair method keeps coverage counts:

```python
cover_count[e]
```

This stores how many selected sets cover each element, allowing faster feasibility checks during repair.

---

## Genetic Algorithm Components

The GA includes the following components:

### Initial Population

The population is initialized using:

- One greedy solution
- Greedy-based variants
- Random repaired chromosomes

This gives the population a strong starting point while preserving diversity.

### Selection

The project uses tournament selection.

### Crossover

The project supports:

- One-point crossover
- Two-point crossover
- Uniform crossover

Uniform crossover is used in the best configuration because gene order is not naturally meaningful in Set Cover.

### Mutation

The project uses optimized bit-flip mutation.

Instead of checking every gene one by one, the number of mutated bits is sampled using a binomial distribution.

### Replacement / Survival

The algorithm uses elitism, meaning the best chromosomes are copied to the next generation.

### Stopping Condition

The GA supports early stopping.

If no improvement is found for a fixed number of generations, the run stops early.

---

## Experiment Configurations

The main experiment compares four configurations:

1. Required configuration
2. Improved / best configuration
3. Crossover-only configuration
4. Mutation-only configuration

The seeds used are:

```python
[42, 123, 999, 2026, 7]
```

The benchmark instances are:

```python
[
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
```

---

## How to Run the Experiments

From the project folder, run:

```bash
python main.py
```

On Windows with Python 3.13:

```bash
C:\Users\omar\AppData\Local\Programs\Python\Python313\python.exe main.py
```

The program will:

1. Read all SCP instances from the `data/` folder.
2. Run the greedy baseline.
3. Run the GA on each instance, configuration, and seed.
4. Print the results.
5. Save per-generation history files.
6. Save the final summary file.

---

## Output Files

After running `main.py`, the program creates:

```text
results/
```

Inside it:

```text
final_summary.csv
history_*.csv
```

### `final_summary.csv`

Contains summary results for each instance and configuration:

- Instance name
- Greedy cost
- Greedy runtime
- Best GA cost
- Average GA cost
- Worst GA cost
- Standard deviation
- Average runtime
- Average number of generations
- Best seed
- Better method

### `history_*.csv`

Each history file contains per-generation data for one run:

- Best fitness
- Average fitness
- Worst fitness
- Standard deviation
- Best cost
- Average cost
- Worst cost
- Selection pressure
- Diversity metrics
- Elapsed runtime

---

## How to Generate Graphs

After running `main.py`, run:

```bash
python plot_results.py
```

Or on Windows with Python 3.13:

```bash
C:\Users\omar\AppData\Local\Programs\Python\Python313\python.exe plot_results.py
```

This will create a folder named:

```text
plots/
```

The generated graphs include:

```text
01_best_average_worst_fitness.png
02_population_diversity.png
03_selection_pressure.png
04_configuration_cost_comparison.png
05_parameter_sensitivity_parallel_coordinates.png
06_best_cost_over_generations.png
```

---

## Graph Descriptions

### 1. Best / Average / Worst Fitness Over Generations

Shows how the best, average, and worst fitness values change over generations.

Since fitness is negative cost, higher fitness means better solution.

### 2. Population Diversity Over Generations

Shows diversity metrics across generations, including:

- Sampled Hamming distance
- Active genes standard deviation
- Unique chromosome ratio

This graph helps explain whether the population is converging or maintaining diversity.

### 3. Selection Pressure Over Generations

Shows how strongly the best individual differs from the average population.

Higher selection pressure means the best individuals are much better than the average.

### 4. Average Solution Cost by Configuration

Compares the greedy baseline and GA configurations across all benchmark instances.

### 5. Parameter Sensitivity Graph

Shows how different configurations relate to solution quality, runtime, and convergence.

### 6. Best Cost Over Generations

Shows how the best solution cost improves over generations.

This graph is often easier to understand than the fitness graph because Set Cover is a minimization problem.

---

## Notes About Runtime

The greedy baseline is very fast, but it usually produces worse solutions.

The GA is slower, but it produces significantly better solutions.

This is expected because the GA is an offline optimization algorithm. Early stopping is used to avoid wasting time after convergence.

---

## Example Result Interpretation

If the output says:

```text
Seed 42: cost=237, time=129.63s, generations=212, feasible=True
```

It means:

- The solution cost found by the GA is 237
- The run took 129.63 seconds
- The algorithm stopped after 212 generations
- The solution is feasible and covers all elements

---

## Main Conclusion

The Genetic Algorithm consistently improves the greedy baseline in solution quality.

The best-performing configurations use:

- Greedy-based initialization
- Repair for infeasible solutions
- Tournament selection
- Uniform crossover
- Bit-flip mutation
- Elitism
- Early stopping
- Diversity tracking

The tradeoff is higher runtime compared to greedy, but the improvement in solution cost makes the GA useful for the Set Cover optimization task.
