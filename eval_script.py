import os
import pickle
import sys
import time

import networkx as nx
import numpy as np

def main():

    with open('datasets/pareto_front_osdpm.pickle', 'rb') as f:
        pareto_front = pickle.load(f)
    with open('datasets/graph_dict_osdpm.pickle', 'rb') as f:
        graph = pickle.load(f)

    total_time, exact_path_matches, undominated_not_in_pareto = evaluate_track1(graph, pareto_front[0:5], pf_algorithm)

    print(f"Score 1: {exact_path_matches}, Time 1: {total_time}", undominated_not_in_pareto)

    num_samples = 100000

    # Parameters for track 2
    num_solutions = 3
    max_time = 5

    param_dict = {
        0: {'a_mean': 0.05, 'a_std': 0.01, 'b_mean': 150, 'b_std': 10},  # First cost
        1: {'a_mean': 0.1, 'a_std': 0.02, 'b_mean': 10, 'b_std': 2},  # Second cost (kept low)
        2: {'a_mean': 0.05, 'a_std': 0.01, 'b_mean': 100, 'b_std': 10},  # Third cost (similar to first)
        3: {'a_mean': 0.05, 'a_std': 0.01, 'b_mean': 100, 'b_std': 10}  # Fourth cost (similar to first)
    }

    score2, time2 = evaluate_track2(num_solutions, pf_algorithm, graph, pareto_front[5:10], twopl_utility_func, param_dict, num_samples, max_time)

    print(f"Score 2: {score2}, Time 2: {time2}")

def pf_algorithm(graph, node_a, node_b):
    pass


def twopl_utility_func(v_list, param_dict, num_samples,
                       default_params={'a_mean': 1.0, 'a_std': 1.0, 'b_mean': 0.0, 'b_std': 1.0}):
    """
    Monte Carlo integration to compute the two-parameter logistic utility.

    Parameters:
    v_list: List of tuples, each containing (cost_tuple, path)
      - cost_tuple is a tuple of 4 costs
      - path can be ignored
    param_dict: Dictionary of parameter settings for each of the 4 costs
    num_samples: Number of Monte Carlo samples
    default_params: Default parameters if not specified

    Returns:
    List of utility estimates for each input vector
    """

    def twoPL(v, a, b):
        """Two-parameter logistic function."""
        return 1.0 / (1.0 + np.exp(-a * (v - b)))

    # Initialize max product
    max_product = 0

    # Process each input vector
    for cost_tuple, _ in v_list:
        # Process each cost in the 4-tuple
        for i, cost in enumerate(cost_tuple):
            # Get parameters for this specific cost
            params = param_dict.get(i, default_params)
            a_mean = params.get('a_mean', default_params['a_mean'])
            a_std = params.get('a_std', default_params['a_std'])
            b_mean = params.get('b_mean', default_params['b_mean'])
            b_std = params.get('b_std', default_params['b_std'])

            # Sample parameters
            a_samples = np.random.normal(loc=a_mean, scale=a_std, size=num_samples)
            b_samples = np.random.normal(loc=b_mean, scale=b_std, size=num_samples)

            # Compute transformations for this cost
            trans_samples_i = twoPL(cost, a_samples, b_samples)

            for j, other_cost in enumerate(cost_tuple):
                if i == j:
                    continue

                # Get parameters for other cost
                other_params = param_dict.get(j, default_params)
                a_mean_j = other_params.get('a_mean', default_params['a_mean'])
                a_std_j = other_params.get('a_std', default_params['a_std'])
                b_mean_j = other_params.get('b_mean', default_params['b_mean'])
                b_std_j = other_params.get('b_std', default_params['b_std'])

                # Sample parameters for other cost
                a_samples_j = np.random.normal(loc=a_mean_j, scale=a_std_j, size=num_samples)
                b_samples_j = np.random.normal(loc=b_mean_j, scale=b_std_j, size=num_samples)

                # Compute transformations
                trans_samples_j = twoPL(other_cost, a_samples_j, b_samples_j)

                # Compute and update max product
                max_product = max(max_product, np.mean(trans_samples_i * trans_samples_j))

    # If only one vector, return its results directly
    return max_product

def is_valid_path(graph, path):
    for i in range(len(path) - 1):
        current_node = path[i]
        next_node = path[i + 1]
        if current_node not in graph:
            return False, f"Node {current_node} not found in graph."
        if next_node not in graph[current_node]:
            return False, f"No edge between {current_node} and {next_node}."
    return True, "Path is valid."

def evaluate_track1(graph, pareto_fronts, algorithm):

    # pareto_found returns the count of matched or undominated solutions
    # total_time returns the total time for the provided algorithm to solve all instances
    # exact_path_matches returns the number of paths in algorithm output that exactly match any pareto path
    # undominated_not_in_pareto returns the list of (cost_tuple, path) from algorithm output that are undominated but not in pareto

    pareto_found = 0
    total_time = 0
    exact_path_matches = 0

    for instance in pareto_fronts:

        undominated_not_in_pareto = []
        valid_solutions= []
        invalid_paths = []

        node_a = instance['source']
        node_b = instance['target']
        pareto_set = instance['pareto_set']

        # extract pareto costs and paths
        pareto_costs = [cost for cost, _ in pareto_set]
        pareto_paths = [path for _, path in pareto_set]

        # run algorithm and time it
        start_time = time.time()
        solutions = algorithm(graph, node_a, node_b)
        elapsed = time.time() - start_time
        total_time += elapsed

        for sol_cost, sol_path in solutions:

            # check if path is valid and adheres to graph structure
            if not is_valid_path(graph, sol_path):
                invalid_paths.append(sol_path)
                continue
            valid_solutions.append((sol_cost, sol_path))

            # recalculate costs to validate cost matches graph data
            calc_costs = [0, 0, 0, 0]
            valid = True

            for i in range(len(sol_path) - 1):
                u, v = sol_path[i], sol_path[i + 1]
                if v not in graph.get(u, {}):
                    valid = False
                    break
                edge_costs = graph[u][v]  
                for j in range(4):
                    calc_costs[j] += edge_costs[j]

            if not valid or not all(abs(calc_costs[i] - sol_cost[i]) < 1e-5 for i in range(4)):
                continue 

            # path match check
            if sol_path in pareto_paths:
                exact_path_matches += 1

            # check for exact cost match 
            if sol_cost in pareto_costs:
                pareto_found += 1
            else:
                # dominance check
                undominated = True
                for pareto_cost in pareto_costs:
                    if all(p <= s for p, s in zip(pareto_cost, sol_cost)) and any(p < s for p, s in zip(pareto_cost, sol_cost)):
                        undominated = False
                        break

                if undominated:
                    pareto_found += 1
                    # dump solutions not found in pareto front but undominated
                    undominated_not_in_pareto.append((sol_cost, sol_path))

    return total_time, exact_path_matches, undominated_not_in_pareto

def evaluate_track2(num_solutions, algorithm, graph, pareto_fronts, utility_func, param_dict, num_samples, max_time):

    # Call the algorithm, time the call, get the time and the set of solutions
    # if the algorithm exceeds the time limit, return inf (disqualified)
    # Check if the set of solutions is the correct number, calculate
    # the expected utility of the solutions
    # return sum_i (expected utility), total_time
    # the best one will be max expected utility and if equal min total_time
    
    total_time = 0
    total_utility = 0

    for index in range(len(pareto_fronts)):

        node_a = pareto_fronts[index]['source']
        node_b = pareto_fronts[index]['target']

        current_time = time.time()
        solutions = algorithm(graph, node_a, node_b, utility_func, param_dict, max_time)
        total_time += time.time() - current_time

        if total_time > max_time:
            return float('inf'), float('inf')

        if len(solutions) != num_solutions:
            return float('inf'), total_time

        total_utility += utility_func(solutions, param_dict, num_samples)

    return total_utility, total_time


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
