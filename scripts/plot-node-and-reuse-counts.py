#!/bin/python3

# This script reads a log file containing JSON entries of node counts and reuse counts,
# and generates a plot showing the total and reusable node counts over time.
#
# Usage: python plot-usage.py <log_file>
# The log file should contain JSON objects with total and reusable node counts for each node variant for every 10k blocks.
# The lines should look like:
# {"node1": {"total": 1000, "reuse": 200}, "node2": {"total": 1500, "reuse": 300}, ...}
# Lines which do not start with '{' are ignored.

import sys
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import colorsys
import os

def parse_data(filename):
    data = {}

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()

            if not line.startswith('{'):
                continue

            entry = json.loads(line)

            for node_name, stats in entry.items():
                if node_name not in data:
                    data[node_name] = {'total': [], 'reuse': []}

                data[node_name]['total'].append(stats['total'])
                data[node_name]['reuse'].append(stats['reuse'])
    return data

def plot_data(data):
    fig, [ax_total_reuse, ax_used] = plt.subplots(2, 1, figsize=(24, 12))

    colormap = plt.get_cmap('tab20')
    node_keys = list(data.keys())

    for idx, node in enumerate(node_keys):
        color = colormap(idx % len(data))

        y_total = data[node]['total']
        y_reuse = data[node]['reuse']
        y_used = [t - r for t, r in zip(y_total, y_reuse)]

        x_block_heights = [x / 100 for x in range(len(y_total))]

        ax_total_reuse.plot(x_block_heights, y_total,
                color=color,
                linestyle='-',
                linewidth=1.5,
                label=f'{node} (Total)')

        ax_total_reuse.plot(x_block_heights, y_reuse,
                color=color,
                linestyle=':',
                linewidth=2.5,
                label=f'{node} (Reuse)')

        ax_used.plot(x_block_heights, y_used,
                color=color,
                linestyle='-',
                linewidth=1.5,
                label=f'{node}')

    ax_total_reuse.set_xlabel('Block Height in 1M')
    ax_total_reuse.set_xlim(left=0, right=len(data[node_keys[0]]['total']) / 100)
    ax_total_reuse.set_ylim(bottom=100)
    ax_total_reuse.set_ylabel('Count')
    ax_total_reuse.set_yscale('log')
    ax_total_reuse.set_title('Total Node Count vs Reuse')
    ax_total_reuse.grid(True, which="both", ls="-", alpha=0.2)
    ax_total_reuse.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0.)

    ax_used.set_xlabel('Block Height in 1M')
    ax_used.set_xlim(left=0, right=len(data[node_keys[0]]['total']) / 100)
    ax_used.set_ylabel('Count')
    ax_used.set_yscale('log')
    ax_used.set_title('Used Node Count')
    ax_used.grid(True, which="both", ls="-", alpha=0.2)
    ax_used.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0.)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./plot-node-and-reuse-counts.py <log_file>")
        sys.exit(1)

    file = sys.argv[1]
    data = parse_data(file)
    plot_data(data)
