
#This module reads Set Cover instances from SCP format files.

#Expected file format:
# First two integers: m n
# Next n integers: costs of the sets
# Then for each of the m elements:
#    k s1 s2 ... sk
#  where k is the number of sets that cover the element,
#  and s1..sk are 1-based set indices.
from problem import SetCoverProblem


def parse_scp_file(path):
    """
    Parse a Set Cover instance from a file.
    Args:
        path (str): Path to the SCP instance file.
    Returns:
        SetCoverProblem: Parsed problem object.
    """
    with open(path, "r") as f:
        tokens = f.read().split()

    tokens = list(map(int, tokens))
    idx = 0

    m = tokens[idx]
    idx += 1
    n = tokens[idx]
    idx += 1

    costs = tokens[idx:idx + n]
    idx += n

    element_to_sets = []
    for _ in range(m):
        k = tokens[idx]
        idx += 1

        covering_sets = [tokens[idx + i] - 1 for i in range(k)]  # convert to 0-based
        idx += k
        element_to_sets.append(covering_sets)

    set_to_elements = [[] for _ in range(n)]
    for element, sets_list in enumerate(element_to_sets):
        for s in sets_list:
            set_to_elements[s].append(element)

    return SetCoverProblem(
        m=m,
        n=n,
        costs=costs,
        element_to_sets=element_to_sets,
        set_to_elements=set_to_elements,
        name=path
    )