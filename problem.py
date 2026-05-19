
#This module defines the SetCoverProblem class.
#The class stores all the information for a Set Cover instance:
# number of elements to cover
# number of available sets
# cost of each set
# mapping from each element to the sets that cover it
# mapping from each set to the elements it covers

class SetCoverProblem:
    #Represents a single Set Cover problem instance.

    def __init__(self, m, n, costs, element_to_sets, set_to_elements, name=""):
        """
        Initialize a SetCoverProblem object.
        Args:
            m (int): Number of elements that must be covered.
            n (int): Number of available sets.
            costs (list[float]): Cost of each set.
            element_to_sets (list[list[int]]): For each element, a list of set indices that cover it.
            set_to_elements (list[list[int]]): For each set, a list of element indices covered by it.
            name (str): Optional problem name.
        """
        self.m = m
        self.n = n
        self.costs = costs
        self.element_to_sets = element_to_sets
        self.set_to_elements = set_to_elements
        self.name = name

    def __repr__(self):
        return (
            f"SetCoverProblem(name={self.name}, m={self.m}, n={self.n})"
        )