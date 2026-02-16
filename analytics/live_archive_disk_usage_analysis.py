# %% Setup

from matplotlib.transforms import offset_copy
import re
import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# The default CSV to load. When running non-interactively, this can be overridden by a command-line argument.
# Note that the titles of plots below need to be adjusted accordingly if different data is used.
CSV_PATH = "./data/2026_01_12_S5_55M_blocks_disk_usage.log"


def parse_db_size(log_file_path):
    """Parses a Bertha log file to extract live and archive DB sizes over time. Sizes are in GiB."""

    block_numbers = []
    live_sizes = []
    archive_sizes = []
    pattern = r"Processing block (\d+).*?, [\d.]+ txs/s, [\d.]+ MGas/s, [\d.]+x realtime, live DB size: ([\d.]+) ([GM])iB, archive DB size: ([\d.]+) [GM]iB"

    with open(log_file_path, "r", encoding="utf-8") as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                block = int(match.group(1))
                live_size = float(match.group(2))
                unit = match.group(3)
                archive_size = float(match.group(4))

                # NOTE: Earlier versions of Bertha logged in MiB instead of GiB, normalize to GiB
                if unit == "M":
                    live_size /= 1024
                    archive_size /= 1024

                block_numbers.append(block)
                live_sizes.append(live_size)
                archive_sizes.append(archive_size)

    print(f"Parsed {len(block_numbers)} log entries.")
    return block_numbers, live_sizes, archive_sizes


is_interactive = "IPython" in sys.modules
if not is_interactive and len(sys.argv) > 1:
    path = sys.argv[1]
else:
    path = CSV_PATH

db_sizes = parse_db_size(path)

# %% Plot live versus archive DB sizes over time

plt.figure(figsize=(10, 5))
plt.plot(db_sizes[0], db_sizes[1], label="Live DB")
plt.plot(db_sizes[0], db_sizes[2], label="Archive DB")
plt.xlabel("Block Number")
plt.ylabel("DB Size [GiB]")
plt.title("S5 DB Size Growth - Full History Run (55M Blocks)")
plt.grid(True)
plt.legend()
plt.show()

# %% Compute growth rates and associated statistics

live_growth_rates = [
    db_sizes[1][i] - db_sizes[1][i - 1] for i in range(1, len(db_sizes[1]))
]
archive_growth_rates = [
    db_sizes[2][i] - db_sizes[2][i - 1] for i in range(1, len(db_sizes[2]))
]

# We assume logs every 10K blocks in messages / labels below
assert db_sizes[0][1] - db_sizes[0][0] == 10000


def print_statistics(block_indices, growth_rates, db_type):
    mean_growth = sum(growth_rates) / len(growth_rates)
    median_growth = sorted(growth_rates)[len(growth_rates) // 2]
    max_growth = max(growth_rates)
    p95_growth = sorted(growth_rates)[int(0.95 * len(growth_rates))]

    print(
        f"{db_type} DB growth rate (GiB / 10K blocks):\n\tMean={mean_growth:.3f}\n\tMedian={median_growth:.3f}\n\tMax={max_growth:.3f}\n\tP95={p95_growth:.3f}"
    )

    # Print outliers outside of fence*IQR
    fence = 5
    q1 = sorted(growth_rates)[len(growth_rates) // 4]
    q3 = sorted(growth_rates)[3 * len(growth_rates) // 4]
    iqr = q3 - q1
    lower_bound = q1 - fence * iqr
    upper_bound = q3 + fence * iqr
    outliers = [
        (i, gr)
        for i, gr in enumerate(growth_rates)
        if gr < lower_bound or gr > upper_bound
    ]
    print(f"{len(outliers)} Outliers (outside {fence}*IQR):")
    for i, outlier in outliers:
        print(
            f"\t\tBlocks: {block_indices[i]}-{block_indices[i] + 10000}, Growth Rate: {outlier:.3f} GiB / 10K blocks"
        )


print_statistics(db_sizes[0], live_growth_rates, "Live")
print_statistics(db_sizes[0], archive_growth_rates, "Archive")

# %% Plot growth rate comparison

fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

axs[0].plot(db_sizes[0], db_sizes[1], label="Live DB")
axs[0].set_ylabel("DB Size [GiB]")
axs[0].set_title("S5 DB Size Growth - Full History Run (55M Blocks) - Live Only")
axs[0].grid(True)

axs[1].plot(db_sizes[0][1:], live_growth_rates, label="Live DB")
axs[1].set_xlabel("Block Number")
axs[1].set_ylabel("GiB / 10K Blocks")
axs[1].set_title("S5 DB Size Growth Rate - Full History Run (55M Blocks) - Live Only")
axs[1].grid(True)

fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

axs[0].plot(db_sizes[0], db_sizes[2], label="Archive DB", color="orange")
axs[0].set_ylabel("DB Size [GiB]")
axs[0].set_title("S5 DB Size Growth - Full History Run (55M Blocks) - Archive Only")
axs[0].grid(True)

axs[1].plot(db_sizes[0][1:], archive_growth_rates, label="Archive DB", color="orange")
axs[1].set_xlabel("Block Number")
axs[1].set_ylabel("GiB / 10K Blocks")
axs[1].set_title(
    "S5 DB Size Growth Rate - Full History Run (55M Blocks) - Archive Only"
)
axs[1].grid(True)

# %% Plot empirical CDFs of growth rates


def plot_empirical_cdf(data, title):
    sorted_data = sorted(data)
    n = len(sorted_data)
    y_values = [(i + 1) / n for i in range(n)]

    plt.figure(figsize=(10, 5))
    plt.plot([x * 1024 for x in sorted_data], y_values, marker=".", linestyle="none")
    plt.xlabel("MiB / 10K Blocks")
    plt.ylabel("Empirical CDF")
    plt.title(title)
    plt.grid(True)

    median_value = sorted_data[n // 2] * 1024
    max_value = sorted_data[-1] * 1024
    p95_value = sorted_data[int(0.95 * n)] * 1024

    plt.axvline(x=median_value, color="red")
    plt.axvline(x=max_value, color="red")
    plt.axvline(x=p95_value, color="red")
    text_transform = offset_copy(
        plt.gca().transData, fig=plt.gcf(), x=10, y=0, units="points"
    )
    plt.text(
        median_value,
        0.3,
        f"Median={round(median_value)}",
        rotation=90,
        transform=text_transform,
        color="red",
    )
    plt.text(
        max_value,
        0.3,
        f"Max={round(max_value)}",
        rotation=90,
        verticalalignment="top",
        transform=text_transform,
        color="red",
    )
    plt.text(
        p95_value,
        0.6,
        f"P95={round(p95_value)}",
        rotation=90,
        verticalalignment="top",
        transform=text_transform,
        color="red",
    )

    plt.show()


def plot_empirical_cdf_broken_axis(data, break_at, outlier_region, title):
    sorted_data = sorted(data)
    n = len(sorted_data)
    y_values = [(i + 1) / n for i in range(n)]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10, 5), sharey=True, width_ratios=[4, 1]
    )
    fig.subplots_adjust(wspace=0.03)

    ax1.set_xlim(0, break_at * 1024)
    ax2.set_xlim(outlier_region[0] * 1024, outlier_region[1] * 1024)

    median_value = sorted_data[n // 2] * 1024
    max_value = sorted_data[-1] * 1024
    p95_value = sorted_data[int(0.95 * n)] * 1024

    for ax in [ax1, ax2]:
        ax.plot([x * 1024 for x in sorted_data], y_values, marker=".", linestyle="none")

        ax.axvline(x=median_value, color="red")
        ax.axvline(x=max_value, color="red")
        ax.axvline(x=p95_value, color="red")

        text_transform = offset_copy(
            ax.transData, fig=ax.figure, x=5, y=0, units="points"
        )

        ax.text(
            median_value,
            0.3,
            f"Median={round(median_value)}",
            rotation=90,
            transform=text_transform,
            color="red",
            clip_on=True,
        )
        ax.text(
            max_value,
            0.3,
            f"Max={round(max_value)}",
            rotation=90,
            transform=text_transform,
            color="red",
            clip_on=True,
        )
        ax.text(
            p95_value,
            0.6,
            f"P95={round(p95_value)}",
            rotation=90,
            transform=text_transform,
            color="red",
            clip_on=True,
        )

        ax.grid(True)

    fig.subplots_adjust(left=0.075, top=0.92)
    fig.supylabel("Empirical CDF", fontsize=10)
    fig.supxlabel("MiB / 10K Blocks", fontsize=10)
    fig.suptitle(title)

    ax1.spines.right.set_visible(False)
    ax2.spines.left.set_visible(False)
    ax1.yaxis.tick_left()
    ax2.yaxis.tick_right()
    ax1.tick_params(labelright=False)
    ax2.tick_params(labelleft=False)

    # draw broken axes indicators
    d = 1.5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(
        marker=[(-1, -d), (1, d)],
        markersize=12,
        linestyle="none",
        color="k",
        mec="k",
        mew=1,
        clip_on=False,
    )
    ax1.plot([1, 1], [1, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 0], [1, 0], transform=ax2.transAxes, **kwargs)


# plot_empirical_cdf(live_slope, 'Empirical CDF of S5 Live DB Growth Rates - Full History Run (55M Blocks)')
plot_empirical_cdf(
    archive_growth_rates,
    "Empirical CDF of S5 Archive DB Growth Rates - Full History Run (55M Blocks)",
)

plot_empirical_cdf_broken_axis(
    live_growth_rates,
    0.08,
    (0.4, 0.55),
    "Empirical CDF of S5 Live DB Growth Rates - Full History Run (55M Blocks)",
)
# plot_empirical_cdf_broken_axis(archive_slope, 0.7, (0.8, 1), 'Empirical CDF of S5 Archive DB Growth Rates - Full History Run (55M Blocks)')

# %%

archive_delta_logs = dict(
    {
        "Baseline": "./data/archive-for-delta-size-baseline-5M-31503252cba7c5724ebaabc8051f18f43b6d34b7.log",
        "Inner delta": "./data/archive-for-delta-size-inner-only-bfb4f86eeda2695b57a2305f1d0691e6c06c19fb.log",
        "Leaf delta": "./data/archive-for-delta-size-leaf-only-7b45a1cbd651fa8cb3177a90adeb010ef5215cd3.log",
        "Inner and leaf delta": "./data/archive-for-delta-size-inner-leaf-f5c99b82239cf4819c2ee8c48755ebeddfc55c3c.log",
    }
)


def parse_archive_delta_log(archive_delta_logs):
    """
    Parse multiple Bertha log files corresponding to different delta node variants and combine the results into a single DataFrame. The DataFrame will have columns for the variant label, block number, archive DB size in GiB, and growth rate in GiB per 10K blocks.
    """
    df = pd.DataFrame(columns=["Variant", "Block Number", "DB Size [GiB]"])
    for label, log_path in archive_delta_logs.items():
        db_sizes = parse_db_size(log_path)
        archive_growth_rates = [
            db_sizes[2][i] - db_sizes[2][i - 1] for i in range(1, len(db_sizes[2]))
        ]
        temp_df = pd.DataFrame(
            {
                "Variant": [label] * len(db_sizes[0]),
                "Block Number": db_sizes[0],
                "DB Size [GiB]": db_sizes[2],
                "Growth Rate": [0] + archive_growth_rates,
            }
        )
        df = pd.concat([df, temp_df], ignore_index=True)
    return df


def plot_archive_delta_comparison(df):
    """
    Plot the archive DB size and growth rate over time for different delta node variants.
    """
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    plot = sns.lineplot(
        data=df,
        x="Block Number",
        y="DB Size [GiB]",
        hue="Variant",
    )
    plot.set_xlabel("Block Number")
    plot.set_ylabel("DB Size [GiB]")
    plt.show()

    plt.figure(figsize=(12, 6))
    plot = sns.lineplot(
        data=df,
        x="Block Number",
        y="Growth Rate",
        hue="Variant",
    )
    plot.set_xlabel("Block Number")
    plot.set_ylabel("GiB / 10K Blocks")
    plt.show()


df = parse_archive_delta_log(archive_delta_logs)
plot_archive_delta_comparison(df)
