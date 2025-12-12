# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

CSV_PATH = "./carmen_stats_node_counts_by_kind.csv"


def load_and_prepare_data(csv_path):
    """
    Load the node statistics CSV and add total rows for each Node Kind.
    Returns a DataFrame with both subtype and total rows.
    """
    df = pd.read_csv(csv_path)
    node_kinds = df["Node Kind"].unique()
    total_rows = []
    for kind in node_kinds:
        kind_rows = df[df["Node Kind"] == kind]
        total_count = kind_rows["Count"].sum()
        total_rows.append({"Node Kind": kind, "Node Size": kind, "Count": total_count})
    total_df = pd.DataFrame(total_rows)
    return pd.concat([df, total_df], ignore_index=True)


def pie_plot(data, values_col, labels_col, title, pad=20):
    plt.figure(figsize=(10, 7))
    plt.pie(
        data[values_col],
        labels=data[labels_col],
        autopct="%1.1f%%",
        startangle=140,
    )
    plt.title(title, pad=pad)
    plt.axis("equal")
    plt.show()


def bar_plot(data, x_col, y_col, title, order=None, pad=20, rotation=45):
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=x_col,
        y=y_col,
        data=data,
        order=order,
    )
    plt.title(title, pad=pad)
    plt.xticks(rotation=rotation)
    plt.show()


def filter_and_group(df, kind, size_col="Node Size", count_col="Count", threshold=0.01):
    """
    Filter subtypes of a given kind by a minimum percentage threshold.
    Returns a DataFrame with subtypes above the threshold.
    """
    sub_df = df[(df["Node Kind"] == kind) & (df[size_col] != kind)]
    total = sub_df[count_col].sum()
    return sub_df[sub_df[count_col] / total >= threshold], total


def pie_with_other(df, kind, threshold, label_col="Node Size", count_col="Count"):
    """
    Pie plot for a node kind, grouping subtypes below threshold into 'Other'.
    """
    sub_df = df[(df["Node Kind"] == kind) & (df[label_col] != kind)].copy()
    total = sub_df[count_col].sum()
    sub_df["Percentage"] = sub_df[count_col] / total
    above = sub_df[sub_df["Percentage"] >= threshold]
    below = sub_df[sub_df["Percentage"] < threshold]
    if not below.empty:
        other_count = below[count_col].sum()
        above = pd.concat(
            [above, pd.DataFrame([{label_col: "Other", count_col: other_count}])],
            ignore_index=True,
        )
    pie_plot(
        above,
        count_col,
        label_col,
        f"Distribution of {kind} node children ({threshold * 100:.1f}% threshold)",
    )


def set_plot_params():
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16
    plt.rcParams["legend.fontsize"] = 16
    plt.rcParams["axes.titlesize"] = 20


# %%  --- Main analysis and plotting ---

set_plot_params()
df = load_and_prepare_data(CSV_PATH)

# Pie: All node kinds (with totals)
all_kinds_df = df[df["Node Kind"] == df["Node Size"]]
pie_plot(all_kinds_df, "Count", "Node Kind", "Distribution of Node Kinds")

# Pie: All node kinds, excluding "Empty"
non_empty_df = all_kinds_df[all_kinds_df["Node Kind"] != "Empty"]
pie_plot(
    non_empty_df,
    "Count",
    "Node Kind",
    "Distribution of Node Kinds (excluding Empty nodes)",
)

# Pie: Leaf node subtypes above threshold
leaf_threshold = 0.01
leaf_subtypes_df, _ = filter_and_group(df, "Leaf", threshold=leaf_threshold)
pie_plot(
    leaf_subtypes_df,
    "Count",
    "Node Size",
    f"Distribution of Leaf node children ({leaf_threshold * 100:.2f}% threshold)",
)

# Pie: Inner node subtypes above threshold
inner_threshold = 0.01
inner_subtypes_df, _ = filter_and_group(df, "Inner", threshold=inner_threshold)
pie_plot(
    inner_subtypes_df,
    "Count",
    "Node Size",
    f"Distribution of Inner node children ({inner_threshold * 100:.1f}% threshold)",
)

# Bar: Inner node subtypes above 2% threshold
inner_bar_threshold = 0.02
inner_bar_df, _ = filter_and_group(df, "Inner", threshold=inner_bar_threshold)
bar_plot(
    inner_bar_df,
    "Node Size",
    "Count",
    "Inner node children count",
    order=inner_bar_df.sort_values("Count", ascending=False)["Node Size"],
    pad=20,
    rotation=45,
)

# Pie: Inner node subtypes, group below 0.5% as "Other"
pie_with_other(df, "Inner", threshold=0.005)

# Pie: Leaf node subtypes, group below 2% as "Other"
pie_with_other(df, "Leaf", threshold=0.02)
