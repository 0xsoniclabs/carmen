# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")


class TimeResolution:
    """
    Helper class to manage time resolutions for plotting timestamps.
    It takes a time resolution (e.g. seconds, milliseconds, microseconds) and converts it to microseconds, as well as provides a name for the resolution.
    """

    _resolution_name: str

    def __init__(self, resolution_name: str):
        self._resolution_name = resolution_name

    def name(self) -> str:
        return self._resolution_name

    def as_microseconds(self) -> int:
        if self._resolution_name == "seconds":
            return 1_000_000
        elif self._resolution_name == "milliseconds":
            return 1_000
        elif self._resolution_name == "microseconds":
            return 1
        else:
            raise ValueError(f"Unsupported time resolution: {self._resolution_name}")


CSV_PATH = "../../../carmen_storage_op_with_timestamp.csv"

TIME_RESOLUTION = TimeResolution("seconds")
GROUPING_TIME_INTERVAL_US = 10 * TIME_RESOLUTION.as_microseconds()


def load_data(csv_path):
    """
    Load the storage operation timestamps CSV into a DataFrame and rename operation types.
    """
    df = pd.read_csv(csv_path)
    # Rename values in `Op` column.
    df["Op"] = df["Op"].replace(
        {
            "R": "Reserve",
            "S": "Set",
            "G": "Get",
            "D": "Delete",
        }
    )
    return df


def count_op_by_time_interval(df, time_interval_us: int):
    """
    Group the DataFrame by time intervals and count operations per type.
    """
    df_grouped = df.copy()
    df_grouped["Time Interval"] = (
        df_grouped["Timestamp"] // time_interval_us
    ) * time_interval_us
    df_grouped = (
        df_grouped.groupby(["Time Interval", "Op"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    df_grouped["Operation Count"] = df_grouped.drop(columns=["Time Interval"]).sum(
        axis=1
    )

    return df_grouped


df = load_data(CSV_PATH)
df_total_ops = count_op_by_time_interval(df, GROUPING_TIME_INTERVAL_US)

# Convert interval time to the specified time resolution for plotting
df_total_ops["Time Interval"] = (
    df_total_ops["Time Interval"] / TIME_RESOLUTION.as_microseconds()
)

# Operation count over time intervals
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=df_total_ops,
    x="Time Interval",
    y="Operation Count",
    marker="o",
)
plt.title(
    f"Operation count over time (grouped by {GROUPING_TIME_INTERVAL_US / TIME_RESOLUTION.as_microseconds()} {TIME_RESOLUTION.name()})"
)
plt.xlabel(f"Time ({TIME_RESOLUTION.name()})")
plt.ylabel("Operation Count")
plt.grid(True)
plt.show()

# Operation count over time intervals by operation type
plt.figure(figsize=(12, 6))
df_no_total = df_total_ops.drop(columns=["Operation Count"])
operation_types = df_no_total.columns.drop("Time Interval")
sns.lineplot(
    data=df_no_total.melt(id_vars=["Time Interval"], value_vars=operation_types),
    x="Time Interval",
    y="value",
    hue="Op",
    marker="o",
)
plt.title(
    f"Operation count by operation type over time (grouped by {GROUPING_TIME_INTERVAL_US / TIME_RESOLUTION.as_microseconds()} {TIME_RESOLUTION.name()})"
)
plt.xlabel(f"Time ({TIME_RESOLUTION.name()})")
plt.ylabel("Operation Count")
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
df_no_total = df_total_ops.drop(columns=["Operation Count"])
df_no_total = df_no_total.iloc[:-1]
operation_types = df_no_total.columns.drop("Time Interval")
sns.lineplot(
    data=df_no_total.melt(id_vars=["Time Interval"], value_vars=operation_types),
    x="Time Interval",
    y="value",
    hue="Op",
    marker="o",
)
plt.title(
    f"Zoomed - operation count by operation type over time (grouped by {GROUPING_TIME_INTERVAL_US / TIME_RESOLUTION.as_microseconds()} {TIME_RESOLUTION.name()})"
)
plt.xlabel(f"Time ({TIME_RESOLUTION.name()})")
plt.ylabel("Operation Count")
plt.grid(True)
plt.show()

# Using df, for each NodeType, print a stacked bar chart of operation count by operation type
df_node_op_count = df.groupby(["NodeType", "Op"]).size().unstack(fill_value=0)
df_node_op_count.plot(kind="bar", stacked=True)
plt.title("Operation count by node type")
plt.xlabel("Node Type")
plt.ylabel("Operation Count")
plt.grid(True)
plt.show()

# %%
