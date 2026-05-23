# mutation_control.py
# Dynamic mutation-rate control methods.


def nonlinear_mutation_rate(
    generation,
    total_generations,
    p_min=0.005,
    p_max=0.05,
    power=2.0,
):
    """
    Nonlinear decreasing mutation rate.

    Early generations:
        high mutation -> exploration

    Late generations:
        low mutation -> exploitation / fine tuning

    Formula:
        p(t) = p_min + (p_max - p_min) * (1 - t/T)^power
    """
    if total_generations <= 1:
        return p_min

    progress = generation / (total_generations - 1)
    progress = max(0.0, min(1.0, progress))

    rate = p_min + (p_max - p_min) * ((1.0 - progress) ** power)

    return max(p_min, min(p_max, rate))


def triggered_hypermutation_rate(
    base_rate,
    hyper_rate,
    stagnation_generations,
    trigger_after=20,
):
    """
    Triggered Hypermutation.

    If the best solution does not improve for 'trigger_after' generations,
    mutation rate jumps temporarily to hyper_rate.

    Once improvement happens, stagnation becomes 0, so the rate returns to base_rate.
    """
    if stagnation_generations >= trigger_after:
        return hyper_rate, True

    return base_rate, False


def get_controlled_mutation_rate(
    mode,
    generation,
    total_generations,
    base_rate,
    stagnation_generations,
    nonlinear_min_rate=0.005,
    nonlinear_max_rate=0.05,
    nonlinear_power=2.0,
    hypermutation_rate=0.08,
    hypermutation_trigger_after=20,
):
    """
    Return the mutation rate for this generation according to the selected mode.

    Supported modes:
    - fixed
    - nonlinear
    - triggered_hypermutation
    """
    if mode == "fixed":
        return base_rate, False

    if mode == "nonlinear":
        rate = nonlinear_mutation_rate(
            generation=generation,
            total_generations=total_generations,
            p_min=nonlinear_min_rate,
            p_max=nonlinear_max_rate,
            power=nonlinear_power,
        )
        return rate, False

    if mode == "triggered_hypermutation":
        return triggered_hypermutation_rate(
            base_rate=base_rate,
            hyper_rate=hypermutation_rate,
            stagnation_generations=stagnation_generations,
            trigger_after=hypermutation_trigger_after,
        )

    raise ValueError("Unknown mutation control mode: " + str(mode))