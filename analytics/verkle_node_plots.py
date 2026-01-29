# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import sys

sns.set_theme(style="whitegrid")

CSV_PATH = "rust/carmen_stats_node_counts_by_kind.csv"


def load_and_prepare_data(csv_path):
    """
    Load the node statistics CSV and add total rows for each Node Kind.
    Returns a DataFrame with both per-size and total counts.
    """
    df = pd.read_csv(csv_path)
    node_kinds = df["Node Kind"].unique()
    total_rows = []
    for kind in node_kinds:
        kind_rows = df[df["Node Kind"] == kind]
        total_count = kind_rows["Count"].sum()
        total_rows.append({"Node Kind": kind, "Node Size": kind, "Count": total_count})
    total_df = pd.DataFrame(total_rows)
    return (df, total_df)


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


def group_below_threshold(df, kind, threshold=0.01):
    """
    Group all counts below the given threshold into an "Other" category.
    Returns a DataFrame with the grouped data.
    """
    label_col = "Node Size"
    count_col = "Count"
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
    return above


def set_plot_params():
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16
    plt.rcParams["legend.fontsize"] = 16
    plt.rcParams["axes.titlesize"] = 20


# %%  --- Main analysis and plotting ---

set_plot_params()

is_interactive = "IPython" in sys.modules
if not is_interactive and len(sys.argv) > 1:
    path = sys.argv[1]
else:
    path = CSV_PATH

df, total_df = load_and_prepare_data(path)


# Pie: All node kinds, excluding "Empty"
non_empty_df = total_df[total_df["Node Kind"] != "Empty"]
pie_plot(
    non_empty_df,
    "Count",
    "Node Kind",
    "Distribution of Node Kinds (excluding Empty nodes)",
)

# Bar: Inner node children count above 2% threshold
inner_bar_threshold = 0.02
inner_bar_df = group_below_threshold(df, "Inner", threshold=inner_bar_threshold)
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
threshold = 0.005
inner_pie_df = group_below_threshold(df, "Inner", threshold=threshold)
pie_plot(
    inner_pie_df,
    "Count",
    "Node Size",
    f"Distribution of Inner node children ({threshold * 100}% threshold)",
)

# Pie: Leaf node subtypes, group below 2% as "Other"
threshold = 0.02
leaf_pie_df = group_below_threshold(df, "Leaf", threshold=threshold)
pie_plot(
    leaf_pie_df,
    "Count",
    "Node Size",
    f"Distribution of Leaf node values ({threshold * 100}% threshold)",
)
