# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

CSV_PATH = './carmen_stats_node_counts_by_kind.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(CSV_PATH)

# Get the sum of counts for each Node Kind
node_types = df['Node Kind'].unique()
summed_rows = []
for node_type in node_types:
    subtype_rows = df[df['Node Kind'] == node_type]
    total_count = subtype_rows['Count'].sum()
    summed_rows.append({'Node Kind': node_type,
                        'Node Size': node_type, 'Count': total_count})
summed_df = pd.DataFrame(summed_rows)
df = pd.concat([df, summed_df], ignore_index=True)


plt.rcParams['axes.labelsize'] = 16           # x/y label size
plt.rcParams['xtick.labelsize'] = 16          # x-tick font
plt.rcParams['ytick.labelsize'] = 16          # y-tick font
plt.rcParams['legend.fontsize'] = 16          # legend font size
plt.rcParams['axes.titlesize'] = 20          # title font size

# Distribution of all Node Kinds
filtered_df = df[df['Node Kind'] == df['Node Size']]
print(filtered_df)
node_type_counts = filtered_df['Count']
plt.figure(figsize=(10, 7))
plt.pie(node_type_counts,
        labels=filtered_df['Node Kind'], autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Node Kinds', pad=20)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# Distribution without empty nodes
df_non_empty = filtered_df[filtered_df['Node Kind'] != 'Empty']
node_type_counts_non_empty = df_non_empty['Count']
plt.figure(figsize=(10, 7))
plt.pie(node_type_counts_non_empty,
        labels=df_non_empty['Node Kind'], autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Node Kinds (excluding Empty nodes)', pad=20)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# Pie plot of Node Kinds for Leaf nodes (only if their number is within % of leaf nodes)
percentage_threshold = 0.01
leaf_nodes_df = df[(df['Node Kind'] == 'Leaf') &
                   (df['Node Size'] != 'Leaf')]
leaf_node_count = leaf_nodes_df['Count'].sum()
# Filter subtypes with at least 0.4% of leaf nodes
leaf_nodes_df = leaf_nodes_df[leaf_nodes_df['Count'] /
                              leaf_node_count >= percentage_threshold]
plt.figure(figsize=(10, 7))
plt.pie(leaf_nodes_df['Count'], labels=leaf_nodes_df['Node Size'],
        autopct='%1.1f%%', startangle=140)
plt.title(
    f'Distribution of Leaf node children {percentage_threshold*100:.2f}% threshold', pad=20)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# Pie plot of Node Kinds for Inner nodes (only if their number is within % of inner nodes)
percentage_threshold = 0.01
inner_nodes_df = df[(df['Node Kind'] == 'Inner') &
                    (df['Node Size'] != 'Inner')]
inner_node_count = inner_nodes_df['Count'].sum()
# Filter subtypes with at least 3% of inner nodes
inner_nodes_df = inner_nodes_df[inner_nodes_df['Count'] /
                                inner_node_count >= percentage_threshold]
plt.figure(figsize=(10, 7))
plt.pie(inner_nodes_df['Count'],
        labels=inner_nodes_df['Node Size'], autopct='%1.1f%%', startangle=140)
plt.title(
    f'Distribution of Inner node children {percentage_threshold*100:.1f}% threshold', pad=20)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# Bar plot of Count for Inner nodes (only if their number is within % of inner nodes)
inner_nodes_df = df[(df['Node Kind'] == 'Inner') &
                    (df['Node Size'] != 'Inner')]
inner_node_count = inner_nodes_df['Count'].sum()
# Filter subtypes with at least 2% of inner nodes
inner_nodes_df = inner_nodes_df[inner_nodes_df['Count'] /
                                inner_node_count >= 0.02]
plt.figure(figsize=(12, 6))
sns.barplot(x='Node Size', y='Count', data=inner_nodes_df,
            order=inner_nodes_df.sort_values('Count', ascending=False)['Node Size'])
plt.title('Inner node children count', pad=20)
plt.xticks(rotation=45)
plt.show()

# Pie plot of nodes types for Inner nodes (only if their number is within % of inner nodes, the rest go to "Other")
percentage_threshold = 0.028
inner_nodes_df = df[(df['Node Kind'] == 'Inner') &
                    (df['Node Size'] != 'Inner')]
inner_node_count = inner_nodes_df['Count'].sum()
inner_nodes_df = inner_nodes_df.assign(
    Percentage=inner_nodes_df['Count'] / inner_node_count)
inner_nodes_df_other = inner_nodes_df[inner_nodes_df['Percentage']
                                      < percentage_threshold]
inner_nodes_df = inner_nodes_df[inner_nodes_df['Percentage']
                                >= percentage_threshold]
other_count = inner_nodes_df_other['Count'].sum()
other_row = pd.DataFrame(
    [{'Node Size': 'Other', 'Count': other_count}])
inner_nodes_df = pd.concat([inner_nodes_df, other_row], ignore_index=True)
plt.figure(figsize=(10, 7))
plt.pie(
    inner_nodes_df['Count'],
    labels=inner_nodes_df['Node Size'],
    autopct='%1.1f%%',
    startangle=140,
)

plt.title(
    f'Distribution of Inner node children ({percentage_threshold*100:.1f}% threshold)', pad=20)
plt.axis('equal')  # Equal aspect ratio for a perfect circle
plt.show()

# Pie plot of Node Kinds for Leaf nodes (only if their number is within % of leaf nodes, the rest go to "Other")
percentage_threshold = 0.02
leaf_nodes_df = df[(df['Node Kind'] == 'Leaf') &
                   (df['Node Size'] != 'Leaf')]
leaf_node_count = leaf_nodes_df['Count'].sum()
leaf_nodes_df = leaf_nodes_df.assign(
    Percentage=leaf_nodes_df['Count'] / leaf_node_count)
leaf_nodes_df_other = leaf_nodes_df[leaf_nodes_df['Percentage']
                                    < percentage_threshold]
leaf_nodes_df = leaf_nodes_df[leaf_nodes_df['Percentage']
                              >= percentage_threshold]
other_count = leaf_nodes_df_other['Count'].sum()
other_row = pd.DataFrame(
    [{'Node Size': 'Other', 'Count': other_count}])
leaf_nodes_df = pd.concat([leaf_nodes_df, other_row], ignore_index=True)
plt.figure(figsize=(10, 7))
plt.pie(
    leaf_nodes_df['Count'],
    labels=leaf_nodes_df['Node Size'],
    autopct='%1.1f%%',
    startangle=140,
)
# Set the spec 2 to blue, spec 3 to orange, spec other to green, and 256 to something else
plt.title(
    f'Distribution of Leaf node values ({percentage_threshold*100:.1f}% threshold)', pad=20)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
