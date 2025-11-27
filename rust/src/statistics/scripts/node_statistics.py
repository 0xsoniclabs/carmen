# %%
# Read the CSV and process data
import itertools
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pulp
from sortedcontainers import SortedDict, SortedSet

CSV_PATH = './carmen_stats_node_counts_by_kind.csv'

df = pd.read_csv(CSV_PATH)
# Remove all empty nodes
df = df[df['Node Kind'] != 'Empty']
# Group by `Node Kind` and calculate the prefix sum of `Count` within each group
grouped_df = df.groupby('Node Kind').apply(lambda x: x.assign(
    PrefixSum=x['Count'].cumsum())).reset_index(drop=True)
# Collect the node info into a nested dictionary
node_info = dict()
for name, group in grouped_df.groupby('Node Kind'):
    node_info[name] = dict()
    node_info[name]['total_count'] = group['Count'].sum()
    node_info[name]['subtype_count'] = dict()
    node_info[name]['prefix_sum'] = dict()
    for index, row in group.iterrows():
        id = int(row['Node Size'])
        node_info[name]['subtype_count'][id] = row['Count']
        node_info[name]['prefix_sum'][id] = row['PrefixSum']

print(node_info['Inner']['subtype_count'])
print(node_info['Leaf']['subtype_count'])

# %%
# Define the sizes of the nodes
# NOTE: These sizes are based on the current implementation of the trie nodes in Carmen and needs to be manually updated if the implementation changes.

# commitment_size = 64 + 256 / 8 + 2
commitment_size = 32
id_size = 6
id_index = 1
full_inner_node_size = commitment_size + 256 * id_size


def sparse_inner_node_size(num_children):
    return commitment_size + num_children * (id_size + id_index)


inner_node_sizes = [sparse_inner_node_size(
    element) for element in range(256)]
inner_node_sizes.append(full_inner_node_size)

value_size = 32
value_index = 1
stem_size = 31
full_leaf_node_size = commitment_size + 256 * value_size + stem_size


def sparse_leaf_node_size(num_children): return commitment_size + \
    num_children * (value_size + value_index) + stem_size


leaf_node_sizes = [sparse_leaf_node_size(
    element) for element in range(256)]
leaf_node_sizes.append(full_leaf_node_size)

print(leaf_node_sizes)
# %% [markdown]
# # Brute force approach

# %%


def calculate_size(num_nodes_to_use, prefix_sum: dict, max_node: int, node_sizes: dict):
    assert num_nodes_to_use > 0

    # All combination of num-nodes - 1 indexes, with the biggest node always included
    available_nodes = [i for i in prefix_sum.keys() if i != max_node]
    initial_indexes = SortedSet()  # The set of indexes of the selected node sizes
    initial_indexes.add(max_node)  # always include the biggest node

    def calculate_size_for_indexes(index_combination):
        cur_indexes = initial_indexes.copy()
        cur_indexes.update(index_combination)
        space_occupied = 0
        already_covered_nodes = 0
        cur_solution = SortedDict()
        for value in cur_indexes:
            num_nodes_covered = prefix_sum[value] - already_covered_nodes
            space_occupied += node_sizes[value] * num_nodes_covered
            already_covered_nodes += num_nodes_covered
            cur_solution[value] = num_nodes_covered
        return cur_solution, space_occupied

    # Initial minimum solution to the worst case (only the biggest node)
    min_space_occupied = prefix_sum[max_node] * node_sizes[max_node]
    min_solution = SortedDict()
    min_solution[max_node] = prefix_sum[max_node]

    index_combinations = list(itertools.combinations(
        available_nodes, num_nodes_to_use - 1))
    max_concurrent = 100
    prev = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        while prev < len(index_combinations):
            futures = []
            for i in range(prev, min(prev + max_concurrent, len(index_combinations))):
                futures.append(executor.submit(
                    calculate_size_for_indexes, index_combinations[i]))
            for future in as_completed(futures):
                cur_solution, space_occupied = future.result()
                if space_occupied < min_space_occupied:
                    min_space_occupied = space_occupied
                    min_solution = cur_solution.copy()
            prev += max_concurrent
    return min_solution, min_space_occupied


# %% [markdown]
# # Greedy approach

# %%


def calculate_size_for_indexes(indices, node_prefix_sum: dict, node_sizes: dict):
    space_occupied = 0
    already_covered_nodes = 0
    cur_solution = SortedDict()
    for value in indices:
        num_nodes_covered = node_prefix_sum[value] - already_covered_nodes
        space_occupied += node_sizes[value] * num_nodes_covered
        already_covered_nodes += num_nodes_covered
        cur_solution[value] = num_nodes_covered
    return cur_solution, space_occupied


def solve_greedy(num_nodes_to_use, node_prefix_sum: dict, max_node: int, node_sizes: dict):
    assert num_nodes_to_use > 0

    # All combination of num-nodes - 1 indexes, with the biggest node always included
    available_nodes = [i for i in node_prefix_sum.keys() if i != max_node]
    initial_indexes = SortedSet()  # The set of indexes of the selected node sizes
    initial_indexes.add(max_node)  # always include the biggest node

    # Initial minimum solution to the worst case (only the biggest node)
    min_size = node_prefix_sum[max_node] * node_sizes[max_node]
    min_solution = SortedDict()
    min_solution[max_node] = node_prefix_sum[max_node]

    while len(min_solution) < num_nodes_to_use:
        min_tmp_solution = min_solution.copy()
        min_tmp_size = sys.maxsize
        for candidate_key in available_nodes:
            if candidate_key in min_solution:
                continue
            current_indexes = SortedSet(min_solution.keys())
            current_indexes.add(candidate_key)
            current_solution, current_size = calculate_size_for_indexes(
                current_indexes, node_prefix_sum, node_sizes)
            if current_size < min_tmp_size:
                min_tmp_size = current_size
                min_tmp_solution = current_solution
        min_solution = min_tmp_solution
        min_size = min_tmp_size

    # solution = dict()
    # for key in min_solution:
    #     solution[key] = min_solution[key]
    return min_solution, min_size

# %% [markdown]
# # Mixed Integer Programming


# %%


def solve_mip(num_nodes_to_use, node_count_by_type: dict, node_prefix_sum: dict, max_node: int, node_sizes: dict, greedy_solution: dict, node_pruning_threshold: float):
    greedy_dict, _ = greedy_solution
    total_node_count = max(node_prefix_sum.values())
    assert num_nodes_to_use > 0

    problem = pulp.LpProblem("Trie size problem", pulp.LpMinimize)
    nodes_range = [i for i in node_prefix_sum.keys()]
    # Upper bound is the prefix sum of each Node Kind
    nodes_lp_variable = {
        i: pulp.LpVariable(f"n_{i}", lowBound=0,
                      upBound=node_prefix_sum[i], cat=pulp.LpInteger)
        for i in nodes_range
    }
    # Use the greedy solution as initial solution
    for i in greedy_dict.keys():
        nodes_lp_variable[i - 1].setInitialValue(greedy_dict[i])
    # Setup linked binary variables
    node_exist_binary = pulp.LpVariable.dicts(
        "n_binary", nodes_range, 0, 1, cat='Binary')
    # Objective function
    problem += pulp.lpSum([nodes_lp_variable[i] * node_sizes[i]
                     for i in nodes_range]), "Minimize trie size"
    # Constraints
    # ## Sum of number of inner nodes must be equals to the total number of inner nodes
    problem += pulp.lpSum(nodes_lp_variable[i]
                     for i in nodes_range) == total_node_count, "Must cover all nodes"
    # ## Exclude all variable with less than threshold% of the total number of nodes
    threshold_count = (node_pruning_threshold * total_node_count) / 100.0
    for i in nodes_range:
        var_size = node_count_by_type[i]
        if var_size < threshold_count:
            problem += nodes_lp_variable[i] == 0
    # ## Must cover all nodes minus the one covered by previous nodes
    for i in nodes_range:
        problem += nodes_lp_variable[i] <= node_prefix_sum[i] - pulp.lpSum(nodes_lp_variable[j] for j in range(
            0, i) if j in nodes_range), f"node {i} cover at at least its node minus the one covered by the previous nodes"
    # ## Special case: biggest variable must cover the remaining nodes
    last_node_index = max(node_count_by_type.keys())
    problem += node_count_by_type[last_node_index] <= nodes_lp_variable[max_node], "Last node must cover remaining nodes"
    # ## Link the variables to the binary ones
    for i in nodes_range:
        problem += nodes_lp_variable[i] <= (node_prefix_sum[i] + 1) * \
            node_exist_binary[i], f"Link variable {i} existence to usage"
    # ## Limit number of variables to use
    problem += pulp.lpSum([node_exist_binary[i] for i in nodes_range]
                     ) == num_nodes_to_use, "Limit number of variables to use"

    # Solve the problem
    solver = pulp.getSolver("SCIP_PY", msg=0, threads=os.cpu_count())
    problem.solve(solver)

    solution = dict()
    for j in nodes_range:
        if nodes_lp_variable[j].varValue >= 1:
            solution[j] = nodes_lp_variable[j].varValue
    size = problem.objective.value()

    return solution, size

# %%


def print_results(solution_type, solution, size, prev_solution_size, writer):
    writer.write(f"{solution_type} solution:\n")
    writer.write(f"    {len(solution)}: ")
    writer.write(f"{solution}\n")
    writer.write(f"       Size in MiB: {size / (1024 * 1024)}\n")
    if prev_solution_size is not None:
        writer.write(
            f"       Saved space in MiB: {(prev_solution_size - size) / (1024 * 1024)}\n")
        writer.write(
            f"       Saved space in percentage: {100.0 * (prev_solution_size - size) / prev_solution_size}%\n")


# %%


def plot_solution_sizes(node_name, node_range, solution_sizes_greedy, solution_sizes_mip):
    def _plot(node_name, node_range, solution_sizes_greedy, solution_sizes_mip):
        sns.set_theme()
        sns.set_context("talk")
        plt.figure(figsize=(10, 6))
        plt.plot(node_range, [size / (1024 * 1024) for size in solution_sizes_greedy],
                 marker='o', label='Greedy Solution Size (MiB)')
        if len(solution_sizes_mip) > 0:
            plt.plot(node_range, [size / (1024 * 1024)
                     for size in solution_sizes_mip], marker='o', label='MIP Solution Size (MiB)')
        plt.title(f'{node_name}')
        plt.xlabel('Number of specialization')
        plt.ylabel('Storage size (MiB)')
        plt.xticks(node_range)
        # increase font size of labels and thicks
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        # plt.legend()
        plt.grid(True)
        plt.show()

    _plot(node_name, node_range, solution_sizes_greedy, solution_sizes_mip)
    # plot starting from position 1
    solution_sizes_greedy = solution_sizes_greedy[1:]
    solution_sizes_mip = solution_sizes_mip[1:]
    node_range = node_range[1:]
    _plot(node_name + " (zoomed, without specialization size 1)", node_range,
          solution_sizes_greedy, solution_sizes_mip)

# Get the optimal number of node sizes to use for `node_name` with `node_range` available specializations


def solve(node_name, node_range, node_info: dict, max_node: int, node_sizes: dict, node_pruning_threshold: float, writer, ignore_mip=True):
    solution_sizes_greedy = []
    solution_sizes_mip = []
    # add max_node to the node_info if not present
    if max_node not in node_info['subtype_count']:
        node_info['subtype_count'][max_node] = 0
    if max_node not in node_info['prefix_sum']:
        node_info['prefix_sum'][max_node] = node_info['prefix_sum'][max(
            node_info['prefix_sum'].values())]

    writer.write("\n-------------------------------\n")
    writer.write(f"--------- {node_name} nodes ---------\n")
    prev_greedy_solution_size = None
    prev_ilp_solution_size = None
    for i in node_range:
        greedy_solution = solve_greedy(
            i, node_info['prefix_sum'], max_node, node_sizes)
        print_results(
            "Greedy", greedy_solution[0], greedy_solution[1], prev_greedy_solution_size, writer)
        solution_sizes_greedy.append(greedy_solution[1])
        prev_greedy_solution_size = greedy_solution[1]
        if not ignore_mip:
            ilp_solution, ilp_size = solve_mip(
                i, node_info['subtype_count'], node_info['prefix_sum'], 255, node_sizes, greedy_solution, node_pruning_threshold)
            print_results("MIP", ilp_solution, ilp_size,
                          prev_ilp_solution_size, writer)
            solution_sizes_mip.append(ilp_size)
            writer.write(
                f"Difference between Greedy and MIP: {(greedy_solution[1] - ilp_size) / (ilp_size) * 100}%\n")
            prev_ilp_solution_size = ilp_size

    plot_solution_sizes(node_name, node_range,
                        solution_sizes_greedy, solution_sizes_mip)
    writer.write("-------------------------------\n\n")


with open("node_optimization_results.txt", "w") as writer:
    solve("Inner", range(1, 10),
          node_info['Inner'], 256, inner_node_sizes, 0, writer)
    solve("Leaf", range(1, 10),
          node_info['Leaf'], 256, leaf_node_sizes, 0.002, writer)
    writer.flush()

# %% Minimum possible storage size with one specialization per node size


def min_storage_size(node_info: dict, node_sizes: dict, node_type: str):
    min_size = 0
    for i in node_info[node_type]['subtype_count'].keys():
        num_nodes_covered = node_info[node_type]['subtype_count'][i]
        min_size += node_sizes[i] * num_nodes_covered
    return min_size


with open("node_optimization_results.txt", "a") as writer:
    writer.write("Minimum possible storage sizes:\n")
    inner_min_size = min_storage_size(node_info, inner_node_sizes, 'Inner')
    leaf_min_size = min_storage_size(node_info, leaf_node_sizes, 'Leaf')
    writer.write(
        f"  Inner nodes minimum size: {inner_min_size / (1024 * 1024)} MiB\n")
    writer.write(
        f"  Leaf nodes minimum size: {leaf_min_size / (1024 * 1024)} MiB\n")
    writer.write(
        f"  Total minimum size: {(inner_min_size + leaf_min_size) / (1024 * 1024)} MiB\n")
    writer.write("-------------------------------\n\n")
    writer.flush()

# %% Storage size for the selected specialization

# Change this to the selected specializations
specializations = []

res = calculate_size_for_indexes(
    SortedSet(specializations), node_info['Leaf']['prefix_sum'], leaf_node_sizes)
for key in res[0].keys():
    num_nodes = res[0][key]
    folder_size = leaf_node_sizes[key] * num_nodes
    print(
        f"Leaf node specialization {key + 1}: {num_nodes} nodes, size {folder_size / (1024 * 1024)} MiB")

# %%
