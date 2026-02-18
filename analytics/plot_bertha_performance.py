#!/bin/python3

# %% Setup

from operator import itemgetter
import re
import matplotlib.pyplot as plt


def parse_performance_metrics(log_file_path):
    """Parses a Bertha log file to extract MGas/s and realtime factors over time."""

    block_numbers = []
    mgas_values = []
    rt_factors = []
    pattern = (
        r"Processing block (\d+).*?, [\d.]+ txs/s, ([\d.]+) MGas/s, ([\d.]+)x realtime"
    )

    with open(log_file_path, "r") as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                block = int(match.group(1))
                mgas = float(match.group(2))
                rt_factor = float(match.group(3))
                block_numbers.append(block)
                mgas_values.append(mgas)
                rt_factors.append(rt_factor)

    print(f"Parsed {len(block_numbers)} log entries.")
    return block_numbers, mgas_values, rt_factors


def plot_mgas_comparison(mgas_with_labels, title):
    plt.figure(figsize=(10, 5))
    for (block_numbers, mgas_values), label in mgas_with_labels:
        plt.plot(block_numbers, mgas_values, label=label)
    plt.xlabel("Block Number")
    plt.ylabel("MGas/s")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


# %% Comparison between S5 and S6 performance


data = []
data.append(
    (
        itemgetter(0, 1)(
            parse_performance_metrics("./data/nightly_bench_9b783b82_5_go-file.log")
        ),
        "go-file (S5)",
    )
)
data.append(
    (
        itemgetter(0, 1)(
            parse_performance_metrics(
                "./data/nightly_bench_ac707b44_5_go-file-flat.log"
            )
        ),
        "go-file-flat (S5)",
    )
)
data.append(
    (
        itemgetter(0, 1)(
            parse_performance_metrics("./data/nightly_bench_50f881d9_6_rust-file.log")
        ),
        "rust-file (S6)",
    )
)
data.append(
    (
        itemgetter(0, 1)(
            parse_performance_metrics(
                "./data/nightly_bench_50f881d9_6_rust-file-flat.log"
            )
        ),
        "rust-file-flat (S6)",
    )
)
plot_mgas_comparison(data, "S5 vs S6 - First 1M Blocks Sonic Mainnet")

# %% Full history run S5 vs S6

# Note that these were not obtained on the same machine, and the S5 run only goes up to 51M blocks

data = []
data.append(
    (
        itemgetter(0, 1)(
            parse_performance_metrics("./data/2025_12_17_S5_51M_blocks.log")
        ),
        "go-file (S5)",
    )
)
data.append(
    (
        itemgetter(0, 1)(
            parse_performance_metrics("./data/2026_01_23_S6_55M_blocks.log")
        ),
        "rust-file (S6)",
    )
)
plot_mgas_comparison(data, "S5 vs S6 - Full history run")

# %% Plot realtime factor for S5 full history run (55M blocks)

block_numbers, _, rt_factors = parse_performance_metrics(
    "./data/2026_01_12_S5_55M_blocks_disk_usage.log"
)

plt.figure(figsize=(10, 5))
plt.plot(block_numbers, rt_factors, label="Realtime Factor")
plt.xlabel("Block Number")
plt.ylabel("x Realtime")
plt.title("S5 Realtime Factor - Archive Full History Run (55M Blocks)")
plt.grid(True)
plt.yscale("log")
plt.axhline(y=24, color="r", linestyle="--", label="24x Realtime")
plt.legend()

# %% Plot live, archive, and geth performance for S5 up to block 4050000
data = []
data.append(
    (
        itemgetter(0, 1)(
            parse_performance_metrics("./data/live-for-gas-55M-main-056907.log")
        ),
        "Live (S6)",
    )
)
data.append(
    (
        itemgetter(0, 1)(
            parse_performance_metrics("./data/archive-for-gas-55M-main-056907.log")
        ),
        "Archive (S6)",
    )
)
data.append(
    (
        itemgetter(0, 1)(
            parse_performance_metrics(
                "./data/live-for-mgas-go-geth-memory-main-2eab53d5f4ec.log"
            )
        ),
        "Geth Verkle (in-memory)",
    )
)

for i in range(len(data)):
    block_numbers, mgas_values = data[i][0]
    data[i] = (block_numbers[:405], mgas_values[:405]), data[i][1]

plot_mgas_comparison(
    data,
    "Live vs Archive vs Geth Verkle (in-memory) - First 4M Blocks Sonic Mainnet",
)
